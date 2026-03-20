import asyncio
import logging
import random
import re
import sys
import time
from dataclasses import asdict, field, is_dataclass

import hydra

from nemo_skills.inference.generate import GenerationTask, GenerationTaskConfig, InferenceConfig
from nemo_skills.inference.model import server_params
from nemo_skills.inference.model.utils import is_context_window_exceeded_error
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


def extract_code_block(text: str):
    # todo (sean): this is a hack to prevent catching report tags in the CoT, causing parsing errors for gpt-oss.
    text = text.split("<|end|><|start|>assistant<|channel|>final<|message|>")[-1]
    matches = re.findall(r"```cpp(.*?)```", text, re.DOTALL)
    return matches[-1].strip() if matches else None


def extract_failure_summary(text: str):
    text = text.split("<|end|><|start|>assistant<|channel|>final<|message|>")[-1]
    matches = re.findall(r"```report(.*?)```", text, re.DOTALL)
    return matches[-1].strip() if matches else None


def extract_verdict(text: str):
    """Extract a single memory update verdict from a fenced verdict block.

    The model should output a single boxed verdict in a block like:

    ```verdict
    no
    ```
    or
    ```verdict
    3
    ```
    """
    text = text.split("<|end|><|start|>assistant<|channel|>final<|message|>")[-1]
    matches = re.findall(r"```verdict(.*?)```", text, re.DOTALL)
    return matches[-1].strip().lower() if matches else None


def extract_steps(text: str):
    """Extract integer step count from a fenced steps block:

    ```steps
    17
    ```
    """
    text = text.split("<|end|><|start|>assistant<|channel|>final<|message|>")[-1]
    matches = re.findall(r"```steps(.*?)```", text, re.DOTALL)
    if not matches:
        return None
    content = matches[-1].strip()
    m = re.search(r"-?\d+", content)
    try:
        return int(m.group(0)) if m else None
    except Exception:
        return None


@nested_dataclass(kw_only=True)
class IOIExecutionConfig(GenerationTaskConfig):
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: dict = field(default_factory=dict)
    prompt_config: str = "eval/ioi/agent/solver"
    improve_after_verify_prompt_config: str = "eval/ioi/agent/improve_with_private_test"
    self_improve_prompt_config: str = "eval/ioi/agent/self_correct/self_improve"
    memory_verdict_prompt_config: str = "eval/ioi/agent/self_correct/memory_verdict"
    model_choose_steps_prompt_config: str = "eval/ioi/agent/self_correct/choose_steps"
    total_steps: int = 60
    time_limit: str | None = None  # Optional wall-clock time limit in 'HH:MM:SS'
    show_k_solutions: int = 0  # Number of previous solutions to include in improve prompt
    retry_solution: int = 5  # Retry count when code block extraction fails
    max_char_for_stored_solution: int = 20000
    randomize_k_solutions: bool = False
    min_random_k_solutions: int = 1
    model_choose_steps: bool = False
    average_model_choose_steps: int = 1
    sample_tests_attempts: int = 0
    per_step_evaluate: bool = True
    only_sample_tests: bool = False
    context_window_retry_token_reduction: int = 10000
    save_intermediate_checkpoints: bool = True


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_ioi_generation_config", node=IOIExecutionConfig)


class IOIExecutionGenerationTask(GenerationTask):
    # Shared formats for composing solution blocks
    SOLUTION_BLOCK_TEMPLATE = "## Solution{title_suffix} ##\n{code}"
    PREVIOUS_PREAMBLE = (
        "Here are some previous solutions that were submitted to help you design and improve the solution:"
    )

    def __init__(self, cfg: IOIExecutionConfig):
        super().__init__(cfg)
        prompt_kwargs = {
            "examples_type": cfg.examples_type,
        }
        self.prompts = {
            "initial": get_prompt(cfg.prompt_config, **prompt_kwargs),
            "self_improve": get_prompt(cfg.self_improve_prompt_config, **prompt_kwargs),
            "memory_verdict": get_prompt(cfg.memory_verdict_prompt_config, **prompt_kwargs),
            "choose_steps": get_prompt(cfg.model_choose_steps_prompt_config, **prompt_kwargs),
        }
        # Parse time limit locally for this task only
        self._deadline_ts = None
        if cfg.time_limit:
            parts = cfg.time_limit.split(":")
            if len(parts) != 3:
                raise ValueError("time_limit must be in 'HH:MM:SS' format")
            h, m, s = map(int, parts)
            self._deadline_ts = time.time() + (h * 3600 + m * 60 + s)

        # Keep a sliding window of evaluated solutions with their scores and feedback
        # Each entry: {"solution": str, "score": float, "feedback": str} (order denotes recency)
        self.saved_solutions: list[dict] = []

    def _should_save_intermediate_checkpoints(self) -> bool:
        return bool(getattr(self.cfg, "save_intermediate_checkpoints", True) or self._deadline_ts is not None)

    def _load_saved_state(self, async_pos: int):
        if not self._should_save_intermediate_checkpoints():
            return None
        return self.load_intermediate_state(async_pos)

    def _save_checkpoint_state(self, async_pos: int, step: int, state: dict) -> None:
        if not self._should_save_intermediate_checkpoints():
            return
        print(f"[Checkpoint] Saving intermediate pos={async_pos}, step={step}")
        self.save_intermediate_state(async_pos, state)

    def log_example_prompt(self, data):
        pass

    def _build_generation_params(self, data_point, all_data, prompt):
        if is_dataclass(self.cfg.inference):
            inference_params = asdict(self.cfg.inference)
        else:
            inference_params = dict(self.cfg.inference)

        return {
            **inference_params,
            **self.extra_generate_params,
            "prompt": self.fill_prompt(data_point=data_point, data=all_data, prompt=prompt),
            "stop_phrases": [self.cfg.stop_phrase] if self.cfg.stop_phrase else None,
        }

    async def _generate_with_context_retry(self, generation_params, async_pos=None):
        reduction = max(1, int(self.cfg.context_window_retry_token_reduction))
        while True:
            try:
                return await self.generate_with_semaphore(**generation_params)
            except Exception as error:
                if not is_context_window_exceeded_error(error):
                    raise

                current_budget = generation_params.get("tokens_to_generate")
                if not isinstance(current_budget, int):
                    raise

                next_budget = max(1, current_budget - reduction)
                if next_budget == current_budget:
                    raise

                prefix = f"Async Pos : {async_pos} " if async_pos is not None else ""
                print(
                    f"[ContextRetry] {prefix}Reducing tokens_to_generate from {current_budget} to {next_budget}"
                )
                generation_params = {**generation_params, "tokens_to_generate": next_budget}

    async def _call_llm(self, data_point, all_data, prompt_key, **extra_data):
        combined_dp = {**data_point, **extra_data}
        prompt = self.prompts[prompt_key]
        generation_params = self._build_generation_params(combined_dp, all_data, prompt)
        filled_prompt = generation_params["prompt"]
        start_t = time.time()
        llm_out = await self._generate_with_context_retry(
            generation_params,
            async_pos=combined_dp.get(self.cfg.async_position_key),
        )
        gen_time = time.time() - start_t
        return filled_prompt, llm_out, gen_time

    async def _call_llm_with_code_retry(self, data_point, all_data, prompt_key, async_pos: int, **extra_data):
        """Call LLM and ensure a code block exists, retrying up to cfg.retry_solution times."""
        attempts = int(self.cfg.retry_solution)
        last = "```cpp\n # failed to create a solution```"
        for attempt in range(1, attempts + 1):
            filled_prompt, llm_out, gen_time = await self._call_llm(data_point, all_data, prompt_key, **extra_data)
            last = (filled_prompt, llm_out, gen_time)
            if extract_code_block(llm_out["generation"]) is not None:
                return filled_prompt, llm_out, gen_time
            if attempt < attempts:
                print(f"Retry {attempt}/{attempts} at async position {async_pos}.")
        print(f"Failed to create a solution after {attempts} attempts at async position {async_pos}.")
        # Return last even if no block; caller may decide how to handle
        return last

    async def _decide_steps_with_retry(self, data_point, all_data, async_pos: int) -> int | None:
        """Ask the model to choose number of improvement steps. Retries on parse failure."""
        attempts = int(self.cfg.retry_solution)
        chosen = None
        for attempt in range(1, attempts + 1):
            _, llm_out, _ = await self._call_llm(
                data_point,
                all_data,
                "choose_steps",
                total_steps=int(self.cfg.total_steps),
            )
            chosen = extract_steps(llm_out["generation"])
            if isinstance(chosen, int):
                return chosen
            if attempt < attempts:
                print(f"Retry choose-steps {attempt}/{attempts} at async position {async_pos}.")
        return chosen

    async def _decide_steps_average_parallel(
        self, data_point, all_data, async_pos: int, num_samples: int
    ) -> int | None:
        """Run multiple choose-steps queries in parallel and return the rounded average."""
        n = max(1, int(num_samples or 1))
        tasks = [self._decide_steps_with_retry(data_point, all_data, async_pos) for _ in range(n)]
        results = await asyncio.gather(*tasks)
        ints = [r for r in results if isinstance(r, int)]
        if not ints:
            return None
        return int(round(sum(ints) / len(ints)))

    async def process_single_datapoint(self, data_point, all_data, prompt_format=None):
        chat_history = []
        num_steps_completed = 0

        # ICPC does not have a subtask score, we add it manually (max score is 1)
        if data_point.get("subtask_score") is None:
            data_point["subtask_score"] = "1"

        # Capture original and effective K for this run (may be randomized)
        original_show_k = int(self.cfg.show_k_solutions or 0)
        selected_k = original_show_k
        decided_total_steps = int(self.cfg.total_steps)

        # Attempt to resume from latest intermediate state
        async_pos = data_point[self.cfg.async_position_key]
        saved_state = self._load_saved_state(async_pos)
        if saved_state:
            chat_history = saved_state["steps"]
            cur_generation_response = saved_state["generation"]
            num_steps_completed = int(saved_state["num_steps_completed"])
            self.saved_solutions = list(saved_state["saved_solutions"])
            print(f"[Resume] Restoring pos={async_pos} from step {num_steps_completed}")
            if original_show_k > 0:
                selected_k = int(saved_state.get("selected_k", selected_k))
            if getattr(self.cfg, "model_choose_steps", False):
                decided_total_steps = int(saved_state.get("decided_total_steps", decided_total_steps))
        else:
            prompt_txt, solution_response, gen_time = await self._call_llm_with_code_retry(
                data_point, all_data, "initial", async_pos
            )
            cur_generation_response = solution_response["generation"]
            chat_history.append(
                {"num_generated_tokens": solution_response["num_generated_tokens"], "generation_time": gen_time}
            )

            print("[Initial] Generated initial solution.")
            # Save checkpoint after initial solution
            # Determine effective K for this run (optionally randomized)
            if original_show_k > 0:
                if getattr(self.cfg, "randomize_k_solutions", False):
                    min_k = int(getattr(self.cfg, "min_random_k_solutions", 1) or 1)
                    min_k = max(1, min_k)
                    if min_k > original_show_k:
                        min_k = original_show_k
                    selected_k = random.randint(min_k, original_show_k)
                else:
                    selected_k = original_show_k
            # Decide number of improvement steps (optional)
            if getattr(self.cfg, "model_choose_steps", False):
                num_samples = max(1, int(getattr(self.cfg, "average_model_choose_steps", 1) or 1))
                if num_samples == 1:
                    decided = await self._decide_steps_with_retry(data_point, all_data, async_pos)
                else:
                    decided = await self._decide_steps_average_parallel(
                        data_point, all_data, async_pos, num_samples=num_samples
                    )
                if isinstance(decided, int):
                    # Clip to [1, total_steps]
                    decided_total_steps = max(1, min(int(self.cfg.total_steps), decided))
                else:
                    decided_total_steps = int(self.cfg.total_steps)
                print(
                    f"Async Pos : {async_pos} Step : {num_steps_completed} Problem {data_point['id']}: Decided improvement steps = {decided_total_steps} (max {int(self.cfg.total_steps)})"
                )
            self._save_checkpoint_state(
                async_pos,
                num_steps_completed,
                {
                    "id": data_point["id"],
                    "generation": cur_generation_response,
                    "steps": chat_history,
                    "num_steps_completed": num_steps_completed,
                    "saved_solutions": self.saved_solutions,
                    "selected_k": selected_k,
                    "decided_total_steps": decided_total_steps,
                },
            )

        for step_num in range(num_steps_completed, decided_total_steps):
            is_icpc = False
            sample_avg = 0.0
            if self.cfg.per_step_evaluate:
                eval_start_t = time.time()
                eval_results = await self.evaluator.eval_single(
                    {
                        **data_point,
                        "generation": cur_generation_response,
                        "only_sample_tests": self.cfg.only_sample_tests,
                    }
                )
                eval_time = time.time() - eval_start_t
                print(
                    f"Async Pos : {async_pos} Step : {step_num} Problem {data_point['id']}: Evaluation time = {eval_time:.3f}s"
                )
                if chat_history:
                    chat_history[-1]["evaluation_time"] = eval_time
                test_case_results = eval_results["test_case_results"]
                normalized_results = self._normalize_test_case_results(test_case_results)
                if chat_history:
                    chat_history[-1]["subtask_scores"] = {k: v["score"] for k, v in normalized_results.items()}
                outputs_all = test_case_results.get("outputs", [])
                sample_outputs = [o for o in outputs_all if o.get("test_type") == "sample"]

                def _avg(outputs_list):
                    if not outputs_list:
                        return 0.0
                    try:
                        total = len(outputs_list)
                        passed = sum(1.0 if float(o.get("score", 0.0)) == 1.0 else 0.0 for o in outputs_list)
                        return float(passed / total)
                    except Exception:
                        return 0.0

                sample_avg = _avg(sample_outputs)
                is_icpc = (
                    isinstance(test_case_results, dict)
                    and "outputs" in test_case_results
                    and "score" in test_case_results
                    and isinstance(test_case_results.get("outputs"), list)
                )
                if is_icpc and chat_history:
                    chat_history[-1]["sample_score"] = sample_avg

            # Update memory with current code solution and score (use sample_avg)
            if int(selected_k) > 0:
                current_code_block = extract_code_block(cur_generation_response)
                await self._maybe_update_memory(
                    code=current_code_block,
                    step_index=step_num,
                    sample_score=sample_avg,
                    chat_history_ref=chat_history,
                    async_pos=async_pos,
                    problem_id=data_point["id"],
                    k_override=selected_k,
                )

            # Early termination: if configured attempts reached and samples are not perfect (ICPC only)
            attempts_limit = int(getattr(self.cfg, "sample_tests_attempts", 0) or 0)
            if (
                self.cfg.per_step_evaluate
                and is_icpc
                and attempts_limit > 0
                and step_num >= attempts_limit
                and float(sample_avg) < 1.0
            ):
                print(
                    f"[Terminate] Problem {data_point['id']}: Sample tests not perfect after {attempts_limit} steps; stopping."
                )
                break

            # Check if all subtasks passed fully (score == 1 for every output)
            if (
                self.cfg.per_step_evaluate
                and not self.cfg.only_sample_tests
                and all(all(float(o["score"]) == 1.0 for o in v["outputs"]) for v in normalized_results.values())
            ):
                print(f"[Success] Problem {data_point['id']}: All test cases passed at step {step_num}.")

            print(f"[Step {step_num + 1}/{self.cfg.total_steps}] Self-improving solution.")

            # Build the 'solution' payload for the next prompt using only code
            current_code_block = extract_code_block(cur_generation_response)
            if int(selected_k) > 0:
                prev_text = self._build_previous_solutions_text(selected_k)
                current_text = self._build_solution_block(current_code_block)
                solution_payload = prev_text + ("\n\n" if prev_text else "") + current_text
            else:
                solution_payload = current_code_block

            # Ask the LLM to improve the solution without evaluator feedback
            prompt_txt, improve_resp, gen_time = await self._call_llm_with_code_retry(
                data_point,
                all_data,
                "self_improve",
                async_pos=async_pos,
                solution=solution_payload,
            )

            num_steps_completed += 1
            cur_generation_response = improve_resp["generation"]

            chat_history.append(
                {
                    "prompt": prompt_txt,
                    "response": cur_generation_response,
                    "generation_time": gen_time,
                    "num_generated_tokens": improve_resp["num_generated_tokens"],
                }
            )

            # Save checkpoint after each improvement step
            self._save_checkpoint_state(
                async_pos,
                num_steps_completed,
                {
                    "id": data_point["id"],
                    "generation": cur_generation_response,
                    "steps": chat_history,
                    "num_steps_completed": num_steps_completed,
                    "saved_solutions": self.saved_solutions,
                    "selected_k": selected_k,
                },
            )
            # Time limit check only inside the loop after checkpoint save
            if self._deadline_ts is not None and time.time() >= self._deadline_ts:
                print("[TimeLimit] Reached limit after step save; exiting cleanly.")
                sys.exit(0)

        return {
            "id": data_point["id"],
            "generation": cur_generation_response,
            "steps": chat_history,
            "num_steps_completed": num_steps_completed,
            "num_generated_tokens": sum(step.get("num_generated_tokens", 0) for step in chat_history),
        }

    def _normalize_test_case_results(self, test_case_results: dict) -> dict:
        """Normalize evaluator outputs to a common shape:
        {subtask: {"score": float, "outputs": list}}.

        Supports both IOI-style per-subtask dicts and ICPC-style flat dict with
        keys {"score": bool|float, "outputs": list} by mapping the latter to
        a single "overall" subtask.
        """
        # ICPC-style: flat dict with outputs list
        if (
            isinstance(test_case_results, dict)
            and "outputs" in test_case_results
            and "score" in test_case_results
            and isinstance(test_case_results.get("outputs"), list)
        ):
            # ICPC-style: preserve fractional score (proportion of tests passed)
            return {"overall": {"score": float(test_case_results["score"]), "outputs": test_case_results["outputs"]}}

        # IOI-style: dict of subtasks
        return {
            k: {"score": float(v.get("score", 0.0)), "outputs": list(v.get("outputs", []))}
            for k, v in test_case_results.items()
        }

    def _update_saved_solutions(self, solution: str) -> None:
        """Add current solution to sliding window and keep only k most recent."""
        k = int(self.cfg.show_k_solutions or 0)
        if k <= 0:
            return

        self.saved_solutions.append({"solution": solution})

        # Evict to maintain size k+1 (K previous + current)
        while len(self.saved_solutions) > k + 1:
            del self.saved_solutions[0]

    async def _maybe_update_memory(
        self,
        code: str,
        step_index: int,
        sample_score: float,
        chat_history_ref: list,
        async_pos: int,
        problem_id,
        k_override: int | None = None,
    ) -> None:
        """Populate and manage memory of candidate solutions.

        - Initially fill up to K solutions without verdicts.
        - After K filled, ask LLM for a verdict whether to replace an existing one or skip.
        - Only consider code shorter than max_char_for_stored_solution.
        - Store step index and sample score alongside code.
        - Snapshot current candidate solutions (just code strings) into chat history entry.
        """
        k = int((k_override if k_override is not None else self.cfg.show_k_solutions) or 0)
        if k <= 0 or not code:
            return

        # Enforce character limit
        if int(self.cfg.max_char_for_stored_solution) > 0 and len(code) > int(self.cfg.max_char_for_stored_solution):
            # Snapshot existing candidates for visibility
            if chat_history_ref:
                chat_history_ref[-1]["memory_candidates"] = [e.get("solution") for e in self.saved_solutions]
            return

        # If memory not yet full, just append
        if len(self.saved_solutions) < k:
            self.saved_solutions.append(
                {
                    "solution": code,
                    "step": int(step_index),
                    "sample_score": float(sample_score),
                }
            )
            if chat_history_ref:
                chat_history_ref[-1]["memory_candidates"] = [e.get("solution") for e in self.saved_solutions]
            return

        # Already have K solutions; ask for verdict whether to replace
        memory_text_lines = [
            "You are a competitive programmer curating a small memory of promising, diverse solutions.",
            "Sample tests are basic compared to the private suite, but solutions must pass samples.",
            f"Memory capacity K = {k}.",
            "Here are the current memory candidates with their sample test scores:",
        ]
        # Randomize display order to minimize positional bias
        shuffled_indices = list(range(len(self.saved_solutions)))
        random.shuffle(shuffled_indices)
        display_to_actual_index = {
            display_idx + 1: actual_idx for display_idx, actual_idx in enumerate(shuffled_indices)
        }
        for display_idx, actual_idx in enumerate(shuffled_indices, start=1):
            e = self.saved_solutions[actual_idx]
            memory_text_lines.append("")
            memory_text_lines.append(f"[#{display_idx}] sample_score={e.get('sample_score', 0.0):.3f}")
            memory_text_lines.append(
                self._build_solution_block(e.get("solution", ""), title_suffix=f" #{display_idx}")
            )
        memory_text = "\n".join(memory_text_lines)

        # Compose current solution block
        current_text = self._build_solution_block(code, title_suffix=" (current)")

        # Ask LLM for verdict (no retries needed)
        _, llm_out, _ = await self._call_llm(
            data_point={},
            all_data={},
            prompt_key="memory_verdict",
            memory_list=memory_text,
            current_solution=current_text,
            current_sample_score=f"{float(sample_score):.3f}",
            k=k,
        )
        verdict = extract_verdict(llm_out["generation"])
        if verdict and verdict != "no":
            try:
                display_idx = int(verdict)
                if 1 <= display_idx <= len(self.saved_solutions):
                    actual_idx = display_to_actual_index.get(display_idx)
                    self.saved_solutions[actual_idx] = {
                        "solution": code,
                        "step": int(step_index),
                        "sample_score": float(sample_score),
                    }
                    print(
                        f"Async Pos : {async_pos} Step : {step_index} Problem {problem_id}: Overwritten display position {display_idx} (actual {actual_idx + 1}) in memory"
                    )
                else:
                    print(
                        f"Async Pos : {async_pos} Step : {step_index} Problem {problem_id}: Failed to generate a verdict for memory decision."
                    )
            except Exception:
                print(
                    f"Async Pos : {async_pos} Step : {step_index} Problem {problem_id}: Failed to generate a verdict for memory decision."
                )
        elif verdict is None:
            print(
                f"Async Pos : {async_pos} Step : {step_index} Problem {problem_id}: Failed to generate a verdict for memory decision."
            )

        # Snapshot current candidates
        if chat_history_ref:
            chat_history_ref[-1]["memory_candidates"] = [e.get("solution") for e in self.saved_solutions]

    def _compute_subtask_score(self, subtask_data: dict) -> float:
        """Return fractional score when per-test outputs are available, otherwise fallback to provided score.

        For ICPC-style results, each output contains a per-test score (0/1). We compute the
        proportion of passed tests. For IOI-style, we respect the provided subtask score.
        """
        outputs = subtask_data.get("outputs")
        if isinstance(outputs, list) and outputs:
            try:
                total = len(outputs)
                passed = sum(1.0 if float(o.get("score", 0.0)) == 1.0 else 0.0 for o in outputs)
                return float(passed / total)
            except Exception:
                # Fallback to provided score on any unexpected shape
                pass
        return float(subtask_data.get("score", 0.0))

    def _build_previous_solutions_text(self, k: int) -> str:
        """Create a formatted text block listing up to k previous solutions.

        Excludes the most recent solution (current) when possible so it can be
        provided separately via the 'solution' field, avoiding duplication.
        """
        if not k or k <= 0:
            return ""
        if not self.saved_solutions:
            return ""

        # Exclude the latest/current solution from the listed previous ones
        prev = self.saved_solutions[:-1]
        if not prev:
            return ""

        # Take up to k most recent from prev, in reverse-chronological order
        tail = prev[-k:]
        tail = list(reversed(tail))

        lines = [self.PREVIOUS_PREAMBLE]
        for idx, e in enumerate(tail, start=1):
            lines.append("")
            # Use shared block template with numbered title suffix
            lines.append(
                self.SOLUTION_BLOCK_TEMPLATE.format(
                    title_suffix=f" {idx}",
                    code=e["solution"],
                )
            )

        return "\n".join(lines)

    def _build_solution_block(self, code: str, title_suffix: str = "") -> str:
        """Return a single solution block using shared template."""
        return self.SOLUTION_BLOCK_TEMPLATE.format(
            title_suffix=title_suffix,
            code=code,
        )


GENERATION_TASK_CLASS = IOIExecutionGenerationTask


@hydra.main(version_base=None, config_name="base_ioi_generation_config")
def ioi_generation(cfg: IOIExecutionConfig):
    cfg = IOIExecutionConfig(_init_nested=True, **cfg)
    LOG.info("Note: IOI Module is being used.")
    LOG.info("Config used: %s", cfg)
    task = IOIExecutionGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(IOIExecutionConfig, server_params=server_params())

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        ioi_generation()
