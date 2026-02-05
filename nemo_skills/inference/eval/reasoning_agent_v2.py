import asyncio
import json
import logging
import re
import sys
import time
from dataclasses import asdict, field
from pathlib import Path

import hydra
from omegaconf import ListConfig

from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.inference.eval.bfcl import ClientMessageParser, ServerMessageParser
from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.model import get_model, server_params
from nemo_skills.inference.model.utils import is_context_window_exceeded_error
from nemo_skills.prompt.utils import get_prompt, get_token_count
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class ReasoningAgentConfig(GenerateSolutionsConfig):
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # agent (orchestrator)
    inference_reasoner: InferenceConfig = field(default_factory=InferenceConfig)  # reasoner
    server: dict = field(default_factory=dict)
    use_client_parsing: bool = False
    model_name: str | None = None
    max_steps: int = 10
    max_time: str | None = None  # Format: "hh:mm:ss" (e.g., "03:45:00")
    explicit_feedback: bool = False
    avg_score: bool = True
    max_n: int = 5  # Maximum number of parallel solutions the reasoner can generate
    agent_prompt_config: str = "eval/ioi/agent/orchestrator"
    reasoner_prompt_config: str = "eval/ioi/agent/agent_tools_solver"
    reasoner_improve_prompt_config: str = "eval/ioi/agent/self_improve_feedback"

    def __post_init__(self):
        base_url = self.server.get("base_url")
        model = self.server.get("model")
        server_type = self.server.get("server_type", "vllm")

        if isinstance(base_url, ListConfig):
            base_url = list(base_url)
        if isinstance(model, ListConfig):
            model = list(model)
        if isinstance(server_type, ListConfig):
            server_type = list(server_type)

        if not isinstance(base_url, list):
            base_url = [base_url] if base_url else []
        if not isinstance(model, list):
            model = [model] if model else []
        if not isinstance(server_type, list):
            server_type = [server_type] if server_type else ["vllm"]

        if len(base_url) != 2:
            raise ValueError(
                f"ReasoningAgent requires exactly 2 models (agent, reasoner) via server.base_url, got {len(base_url)}. "
                f"Example: ++server.base_url=[http://host1:port1,http://host2:port2]"
            )
        if len(model) != len(base_url):
            raise ValueError(
                f"Number of server.model ({len(model)}) must match number of server.base_url ({len(base_url)})"
            )
        if len(server_type) == 1:
            server_type = server_type * len(base_url)
        elif len(server_type) != len(base_url):
            raise ValueError(
                f"Number of server.server_type ({len(server_type)}) must match number of server.base_url ({len(base_url)}) or be 1 (broadcast)"
            )

        original_server = self.server.copy()
        self.server["base_url"] = base_url
        self.server["model"] = model
        self.server["server_type"] = server_type

        self.server = {"base_url": base_url[0], "model": model[0], "server_type": server_type[0]}
        try:
            super().__post_init__()
        finally:
            self.server = original_server
            self.server["base_url"] = base_url
            self.server["model"] = model
            self.server["server_type"] = server_type


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_reasoning_agent_config", node=ReasoningAgentConfig)


class ReasoningAgentGenerationTask(GenerationTask):
    def __init__(self, cfg: ReasoningAgentConfig):
        self.server_addresses = cfg.server["base_url"]
        self.model_names = cfg.server["model"]
        self.server_types = cfg.server["server_type"]
        self.reasoner_llm = None
        self.reasoner_semaphore = None
        super().__init__(cfg)
        self.message_parser = ClientMessageParser(cfg) if cfg.use_client_parsing else ServerMessageParser(cfg)
        self.agent_prompt = get_prompt(cfg.agent_prompt_config)
        self.reasoner_prompt = get_prompt(cfg.reasoner_prompt_config)
        self.reasoner_improve_prompt = get_prompt(cfg.reasoner_improve_prompt_config)

    def dp_print(self, data_point, *args):
        dp_id = data_point.get("id", "?") if isinstance(data_point, dict) else "?"
        print(f"[{dp_id}]", *args)

    def setup_prompt(self):
        return None

    def log_example_prompt(self, data):
        return

    def setup_llm(self):
        self.sandbox = get_sandbox(**self.cfg.sandbox) if self.cfg.sandbox else None
        output_dir = str(Path(self.cfg.output_file).parent)

        clients = []
        for idx, (address, model_name, server_type) in enumerate(
            zip(self.server_addresses, self.model_names, self.server_types)
        ):
            if not isinstance(address, str) or not address:
                raise ValueError(f"Invalid base_url for server {idx}: {address}")
            if not address.startswith(("http://", "https://")):
                address = f"http://{address}"
            if not address.endswith("/v1"):
                address = f"{address}/v1"
            clients.append(
                get_model(server_type=server_type, model=model_name, base_url=address, output_dir=output_dir)
            )

        self.llm, self.reasoner_llm = clients
        self.reasoner_semaphore = asyncio.Semaphore(self.cfg.max_concurrent_requests)
        return self.llm

    def _extract_cpp(self, text: str | None) -> str | None:
        """Extract C++ code from content."""
        if not text:
            return None

        # Extract from final output channel
        text_final = text.split("<|end|><|start|>assistant<|channel|>final<|message|>")[-1]
        m = re.findall(r"```(?:cpp|c\+\+)\s*(.*?)```", text_final, re.DOTALL | re.IGNORECASE)
        if m:
            return m[-1].strip()

        return None

    def _normalize_scores(self, test_case_results: dict) -> dict:
        """Normalize evaluator outputs to a common shape."""
        # ICPC-style: flat dict with outputs list
        if (
            isinstance(test_case_results, dict)
            and "outputs" in test_case_results
            and "score" in test_case_results
            and isinstance(test_case_results.get("outputs"), list)
        ):
            return {"overall": {"score": float(test_case_results["score"]), "outputs": test_case_results["outputs"]}}
        # IOI-style: dict of subtasks
        return {
            k: {"score": float(v.get("score", 0.0)), "outputs": list(v.get("outputs", []))}
            for k, v in test_case_results.items()
        }

    def _calculate_avg_score(self, normalized_results: dict) -> float:
        """Calculate average score across all test outputs."""
        all_outputs = []
        for subtask_data in normalized_results.values():
            all_outputs.extend(subtask_data.get("outputs", []))

        if not all_outputs:
            return 0.0

        try:
            total = len(all_outputs)
            passed = sum(1.0 if float(o.get("score", 0.0)) == 1.0 else 0.0 for o in all_outputs)
            return float(passed / total)
        except Exception:
            return 0.0

    def _parse_max_time(self, max_time_str: str | None) -> float | None:
        """Parse max_time string (hh:mm:ss) into seconds."""
        if not max_time_str:
            return None
        try:
            parts = max_time_str.split(":")
            if len(parts) != 3:
                raise ValueError(f"Invalid max_time format: {max_time_str}. Expected hh:mm:ss")
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        except Exception as e:
            raise ValueError(f"Invalid max_time format: {max_time_str}. Expected hh:mm:ss. Error: {e}")

    def _build_tools(self):
        """Build tools available to the orchestrator agent."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "generate_solution",
                    "description": "Ask the reasoner to generate or improve a C++17 solution. Use feedback parameter to describe what went wrong with previous solution. Use n parameter to specify how many solutions you would like to generate in parallel (useful when the model finds it difficult to generate 1 solution).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "feedback": {
                                "type": "string",
                                "description": "Brief feedback about what went wrong with previous solution (optional, only for improvements)",
                            },
                            "n": {
                                "type": "integer",
                                "description": "Number of solutions to generate in parallel. Useful when the model finds it difficult to generate 1 solution.",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "submit_solution",
                    "description": "Submit a C++17 solution for evaluation. Set sample=true to run only sample tests first.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "C++17 source code to submit"},
                            "sample": {"type": "boolean", "description": "Run only sample tests", "default": False},
                        },
                        "required": ["code"],
                    },
                },
            },
        ]

    async def _agent_turn(self, messages: list[dict], tools: list[dict]) -> dict:
        """Call agent (orchestrator) model."""
        input_dict = self.message_parser.construct_input_dict(messages, tools)
        return_dict = {}

        if self.cfg.count_prompt_tokens:
            return_dict["num_input_tokens"] = get_token_count(
                self.hf_tokenizer, messages=input_dict["prompt"], tools=input_dict.get("tools", None)
            )

        try:
            out = await self.generate_with_semaphore(**input_dict)
        except Exception as e:
            if is_context_window_exceeded_error(e):
                return {"message": None, "generation": "", **return_dict}
            raise

        parsed = self.message_parser.parse_output_dict(out)
        parsed.update(return_dict)
        return parsed

    async def _call_reasoner(
        self, problem: str, previous_solution: str | None, feedback: str | None, data_point: dict
    ) -> dict:
        """Call reasoner to generate or improve solution."""
        async with self.reasoner_semaphore:
            # Build prompt based on context
            if previous_solution:
                # Use improve prompt with previous solution
                messages = self.reasoner_improve_prompt.fill(
                    {
                        "subtask_score": data_point.get("subtask_score", "1"),
                        "question": problem,
                        "solution": previous_solution,
                        "feedback": feedback or "",
                    }
                )
            else:
                # Use default generate prompt
                messages = self.reasoner_prompt.fill(
                    {"subtask_score": data_point.get("subtask_score", "1"), "question": problem}
                )

            result = await self.reasoner_llm.generate_async(
                prompt=messages, include_response=False, **asdict(self.cfg.inference_reasoner)
            )

            return {
                "generation": result.get("generation", ""),
                "reasoning_content": result.get("reasoning_content", ""),
                "num_generated_tokens": result.get("num_generated_tokens", 0),
            }

    async def _summarize_progress(self, agent_messages: list[dict]) -> str:
        """Generate a concise summary of attempts when context is exceeded."""
        summary_prompt = (
            "Summarize what we tried and what didn't work in 2-3 sentences. "
            "Focus on key failures and patterns observed. Be concise."
        )
        summary_messages = agent_messages + [{"role": "user", "content": summary_prompt}]

        try:
            result = await self.llm.generate_async(
                prompt=summary_messages,
                include_response=False,
                max_tokens=200,
                temperature=0.0,
            )
            return result.get("generation", "").strip()
        except Exception:
            return "Previous attempts exhausted context window."

    async def process_single_datapoint(self, data_point, all_data):
        """Process a single datapoint using tool-based orchestration.

        The agent controls the entire flow via tool calls:
        1. Agent calls generate_solution() to get initial solution
        2. Agent calls submit_solution() to test
        3. Agent calls generate_solution(feedback=...) to improve
        4. Repeat until success

        This design makes the agent a true orchestrator with full control.
        """
        if self.evaluator is None:
            raise ValueError(
                "ReasoningAgent requires an evaluator supporting eval_single (set ++eval_type and ++eval_config)."
            )

        if data_point.get("subtask_score") is None:
            data_point["subtask_score"] = "1"

        problem = data_point["question"]

        # Parse max_time and track start time
        max_time_seconds = self._parse_max_time(self.cfg.max_time)
        start_time = time.time()

        # Load intermediate state if available
        async_position = data_point.get(self.cfg.async_position_key)
        saved_state = self.load_intermediate_state(async_position) if async_position is not None else None

        if saved_state:
            self.dp_print(data_point, f"resuming from iteration {saved_state['iteration']}")
            # Restore state
            agent_messages = saved_state["agent_messages"]
            trace = saved_state["trace"]
            num_agent_tokens = saved_state["num_agent_tokens"]
            num_reasoner_tokens = saved_state["num_reasoner_tokens"]
            previous_solution = saved_state["previous_solution"]
            final_code = saved_state["final_code"]
            out_of_context = saved_state["out_of_context"]
            start_iteration = saved_state["iteration"]
        else:
            self.dp_print(data_point, "start orchestration")
            # Initialize agent conversation
            agent_messages = [
                {"role": "system", "content": self.agent_prompt.config.system},
                {"role": "user", "content": f"Problem:\n{problem}"},
            ]
            trace = []
            num_agent_tokens, num_reasoner_tokens = [], []
            final_code = ""
            out_of_context = False
            previous_solution = None
            start_iteration = 0

        tools = self._build_tools()

        # Agent orchestration loop
        for iteration in range(start_iteration, self.cfg.max_steps):
            # Check if max_time exceeded
            if max_time_seconds is not None:
                elapsed = time.time() - start_time
                if elapsed >= max_time_seconds:
                    self.dp_print(data_point, f"max_time reached ({elapsed:.1f}s >= {max_time_seconds}s)")
                    if async_position is not None:
                        state = {
                            "agent_messages": agent_messages,
                            "trace": trace,
                            "num_agent_tokens": num_agent_tokens,
                            "num_reasoner_tokens": num_reasoner_tokens,
                            "previous_solution": previous_solution,
                            "final_code": final_code,
                            "out_of_context": out_of_context,
                            "iteration": iteration,
                        }
                        self.save_intermediate_state(async_position, state)
                    break

            self.dp_print(data_point, f"iteration {iteration + 1}/{self.cfg.max_steps}")

            # Agent decides what to do
            a = await self._agent_turn(agent_messages, tools)
            if a.get("message") is None:
                self.dp_print(data_point, "agent: out_of_context, generating summary and restarting")

                # Generate summary of what we tried
                summary = await self._summarize_progress(agent_messages)
                self.dp_print(data_point, f"summary: {summary[:100]}...")

                # Restart conversation with summary as feedback
                agent_messages = [
                    {"role": "system", "content": self.agent_prompt.config.system},
                    {
                        "role": "user",
                        "content": f"Problem:\n{problem}\n\nPrevious attempt summary:\n{summary}",
                    },
                ]
                trace.append(
                    {
                        "source": "system",
                        "role": "user",
                        "content": f"Context reset. Summary: {summary}",
                        "iteration": iteration,
                    }
                )
                continue

            num_agent_tokens.append(a.get("num_generated_tokens", 0))
            msg = a["message"]
            if hasattr(msg, "model_dump"):
                msg = msg.model_dump()

            agent_messages.append(msg)
            trace.append({"source": "agent", **msg})

            # Process tool calls
            tool_calls = a.get("generation", [])
            tool_call_ids = a.get("tool_call_ids", [])

            if not isinstance(tool_calls, list) or len(tool_calls) == 0:
                self.dp_print(data_point, "agent: no tool calls")
                break

            # Execute each tool call
            for gen, tool_call_id in zip(tool_calls, tool_call_ids or [None] * len(tool_calls)):
                (name, raw_args) = next(iter(gen.items()))

                # Parse arguments
                args = raw_args
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except Exception:
                        args = {}

                # ============================================================
                # TOOL: generate_solution
                # ============================================================
                if name == "generate_solution":
                    feedback = args.get("feedback")
                    n = args.get("n", 1)

                    # Validate n is within valid range [1, max_n]
                    if not isinstance(n, int) or n < 1 or n > self.cfg.max_n:
                        tool_out = json.dumps(
                            {
                                "error": f"Invalid n parameter: {n}. Must be an integer between 1 and {self.cfg.max_n}.",
                                "status": "error",
                            }
                        )
                        agent_messages.append({"role": "tool", "content": tool_out, "tool_call_id": tool_call_id})
                        trace.append(
                            {"source": "tool", "role": "tool", "content": tool_out, "tool_call_id": tool_call_id}
                        )
                        self.dp_print(data_point, f"error: invalid n={n}, must be 1-{self.cfg.max_n}")
                        continue

                    self.dp_print(
                        data_point, f"tool: generate_solution(feedback={'yes' if feedback else 'no'}, n={n})"
                    )

                    # Call reasoner n times in parallel
                    results = await asyncio.gather(
                        *[self._call_reasoner(problem, previous_solution, feedback, data_point) for _ in range(n)]
                    )

                    # Track total tokens
                    for result in results:
                        num_reasoner_tokens.append(result.get("num_generated_tokens", 0))

                    # Extract code from all results
                    solutions = []
                    for idx, result in enumerate(results):
                        code = self._extract_cpp(result.get("generation", ""))
                        solutions.append(
                            {
                                "code": code if code else None,
                                "generation": result.get("generation", ""),
                                "reasoning_content": result.get("reasoning_content", ""),
                            }
                        )

                        # Log each reasoner call to trace
                        trace.append(
                            {
                                "source": "reasoner",
                                "role": "assistant",
                                "content": result.get("generation", ""),
                                "reasoning_content": result.get("reasoning_content", ""),
                                "feedback": feedback,
                                "previous_solution": previous_solution,
                                "solution_index": idx + 1,
                                "total_solutions": n,
                            }
                        )

                    # Filter valid solutions
                    valid_solutions = [s for s in solutions if s["code"] is not None]

                    if valid_solutions:
                        # Use the first valid solution as previous_solution
                        previous_solution = valid_solutions[0]["code"]
                        # Return all solutions
                        tool_out = json.dumps(
                            {
                                "solutions": [s["code"] for s in valid_solutions],
                                "status": "success",
                                "count": len(valid_solutions),
                            }
                        )
                        self.dp_print(
                            data_point,
                            f"reasoner: generated {len(valid_solutions)}/{n} valid solutions "
                            f"({len(valid_solutions[0]['code'])} chars in first)",
                        )
                    else:
                        tool_out = json.dumps(
                            {"error": f"No cpp code block found in any of {n} generations", "status": "error"}
                        )
                        self.dp_print(data_point, f"reasoner: failed to generate code block in {n} attempts")

                    agent_messages.append({"role": "tool", "content": tool_out, "tool_call_id": tool_call_id})
                    trace.append({"source": "tool", "role": "tool", "content": tool_out, "tool_call_id": tool_call_id})

                # ============================================================
                # TOOL: submit_solution
                # ============================================================
                elif name == "submit_solution":
                    code = args.get("code", "")
                    sample = bool(args.get("sample", False))

                    if not code:
                        tool_out = json.dumps({"error": "No code provided"})
                        agent_messages.append({"role": "tool", "content": tool_out, "tool_call_id": tool_call_id})
                        trace.append(
                            {"source": "tool", "role": "tool", "content": tool_out, "tool_call_id": tool_call_id}
                        )
                        continue

                    self.dp_print(data_point, f"tool: submit_solution(sample={sample}, code_len={len(code)})")

                    # Execute evaluation
                    eval_payload = {
                        **data_point,
                        "generation": f"```cpp\n{code}\n```",
                        "only_sample_tests": sample,
                    }
                    eval_result = await self.evaluator.eval_single(eval_payload)
                    test_case_results = eval_result.get("test_case_results", {})

                    # For IOI, extract the relevant subtask
                    is_ioi = "ioi_id" in data_point
                    if is_ioi and "subtask" in data_point:
                        subtask_name = data_point["subtask"]
                        if subtask_name in test_case_results:
                            test_case_results = {subtask_name: test_case_results[subtask_name]}

                    normalized = self._normalize_scores(test_case_results)

                    # Build tool response
                    if self.cfg.explicit_feedback:
                        tool_out_dict = eval_result
                    else:
                        if is_ioi and "subtask_score" in data_point:
                            # For IOI, report score as "actual/max"
                            max_score = data_point["subtask_score"]
                            subtask_scores = {k: f"{v['score']}/{max_score}" for k, v in normalized.items()}
                        else:
                            subtask_scores = {k: v["score"] for k, v in normalized.items()}
                        tool_out_dict = {"subtask_scores": subtask_scores}

                    if self.cfg.avg_score:
                        tool_out_dict["avg_score"] = self._calculate_avg_score(normalized)

                    success = bool(normalized) and all(float(v["score"]) == 1.0 for v in normalized.values())
                    tool_out_dict["success"] = success

                    tool_out = json.dumps(tool_out_dict)
                    agent_messages.append({"role": "tool", "content": tool_out, "tool_call_id": tool_call_id})
                    trace.append({"source": "tool", "role": "tool", "content": tool_out, "tool_call_id": tool_call_id})

                    self.dp_print(data_point, f"result: success={success}")

                    # Check if we're done
                    if success and not sample:
                        final_code = code
                        self.dp_print(data_point, "✓ solution accepted")

                # Unknown tool
                else:
                    tool_out = json.dumps({"error": f"Unknown tool: {name}"})
                    agent_messages.append({"role": "tool", "content": tool_out, "tool_call_id": tool_call_id})
                    trace.append({"source": "tool", "role": "tool", "content": tool_out, "tool_call_id": tool_call_id})

            # Save intermediate state after each iteration
            if async_position is not None:
                state = {
                    "agent_messages": agent_messages,
                    "trace": trace,
                    "num_agent_tokens": num_agent_tokens,
                    "num_reasoner_tokens": num_reasoner_tokens,
                    "previous_solution": previous_solution,
                    "final_code": final_code,
                    "out_of_context": out_of_context,
                    "iteration": iteration + 1,  # Save next iteration to start from
                }
                self.save_intermediate_state(async_position, state)

            # Check if we have a final solution
            if final_code:
                break

        # Use last generated solution if available
        if not final_code and previous_solution:
            final_code = previous_solution
            self.dp_print(data_point, "using last generated solution")

        out = {
            "id": data_point["id"],
            "generation": "```cpp\n" + final_code + "\n```",
            "messages": trace,
            "num_generated_tokens": sum(num_agent_tokens) + sum(num_reasoner_tokens),
            "num_generated_tokens_list": {"agent": num_agent_tokens, "reasoner": num_reasoner_tokens},
        }

        # Clear intermediate state when done (success or failure)
        if async_position is not None:
            self.clear_intermediate_state(async_position)

        return out

    def wait_for_server(self):
        LOG.info(f"Waiting for {len(self.server_addresses)} server(s) to be ready...")
        original_server_config = self.cfg.server.copy()
        try:
            for idx, (address, model_name) in enumerate(zip(self.server_addresses, self.model_names)):
                if not address.startswith(("http://", "https://")):
                    address = f"http://{address}"
                LOG.info(f"Waiting for Server {idx} ({model_name}) @ {address}...")
                self.cfg.server = {"base_url": address}
                super().wait_for_server()
        finally:
            self.cfg.server = original_server_config


GENERATION_TASK_CLASS = ReasoningAgentGenerationTask


@hydra.main(version_base=None, config_name="base_reasoning_agent_config")
def reasoning_agent_generation(cfg: ReasoningAgentConfig):
    cfg = ReasoningAgentConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)
    task = ReasoningAgentGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(ReasoningAgentConfig, server_params=server_params())

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        reasoning_agent_generation()
