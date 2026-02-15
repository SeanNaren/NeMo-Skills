import asyncio
import json
import logging
import re
import sys
import time
from dataclasses import asdict, field

import hydra

from nemo_skills.inference.generate import GenerationTask, GenerationTaskConfig, InferenceConfig
from nemo_skills.inference.model import server_params
from nemo_skills.inference.model.utils import is_context_window_exceeded_error
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class ReasoningSingleAgentConfig(GenerationTaskConfig):
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    inference_reasoner: InferenceConfig = field(default_factory=InferenceConfig)
    server: dict = field(default_factory=dict)
    max_steps: int = 15
    max_time: str | None = None  # Format: "hh:mm:ss"
    explicit_feedback: bool = False
    max_limit_in_test_output: int = 1000
    avg_score: bool = True
    agent_prompt_config: str = "eval/ioi/agent/orchestrator"
    summary_prompt_config: str = "eval/ioi/agent/summary"
    reasoner_prompt_config: str = "eval/ioi/agent/agent_tools_solver"
    reasoner_improve_prompt_config: str = "eval/ioi/agent/self_improve_feedback"


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_reasoning_single_agent_config", node=ReasoningSingleAgentConfig)


class ReasoningSingleAgentTask(GenerationTask):
    def __init__(self, cfg: ReasoningSingleAgentConfig):
        super().__init__(cfg)
        self.agent_prompt = get_prompt(cfg.agent_prompt_config)
        self.summary_prompt = get_prompt(cfg.summary_prompt_config)
        self.reasoner_prompt = get_prompt(cfg.reasoner_prompt_config)
        self.reasoner_improve_prompt = get_prompt(cfg.reasoner_improve_prompt_config)

    def setup_prompt(self):
        return None

    def log_example_prompt(self, data):
        return

    def dp_print(self, data_point, *args):
        dp_id = data_point.get("id", "?") if isinstance(data_point, dict) else "?"
        print(f"[{dp_id}]", *args)

    def _extract_cpp(self, text: str | None) -> str | None:
        if not text:
            return None
        matches = re.findall(r"```(?:cpp|c\+\+)\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        return matches[-1].strip() if matches else None

    def _normalize_scores(self, test_case_results: dict) -> dict:
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
        all_outputs = []
        for subtask_data in normalized_results.values():
            all_outputs.extend(subtask_data.get("outputs", []))
        if not all_outputs:
            return 0.0
        total = len(all_outputs)
        passed = sum(1.0 if float(o.get("score", 0.0)) == 1.0 else 0.0 for o in all_outputs)
        return float(passed / total)

    def _filter_test_outputs(self, test_case_results: dict) -> dict:
        max_len = self.cfg.max_limit_in_test_output

        def truncate(val):
            if isinstance(val, str) and len(val) > max_len:
                return val[:max_len] + "...<truncated>"
            return val

        def filter_outputs(outputs):
            return [
                {
                    k: truncate(v) if k in ("run_stdout", "run_stderr", "compile_stdout", "compile_stderr") else v
                    for k, v in o.items()
                }
                for o in outputs
                if float(o.get("score", 0.0)) != 1.0
            ]

        # ICPC-style
        if (
            isinstance(test_case_results, dict)
            and "outputs" in test_case_results
            and "score" in test_case_results
            and isinstance(test_case_results.get("outputs"), list)
        ):
            return {**test_case_results, "outputs": filter_outputs(test_case_results["outputs"])}
        # IOI-style
        return {k: {**v, "outputs": filter_outputs(v.get("outputs", []))} for k, v in test_case_results.items()}

    def _parse_max_time(self, max_time_str: str | None) -> float | None:
        if not max_time_str:
            return None
        parts = max_time_str.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid max_time format: {max_time_str}. Expected hh:mm:ss")
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds

    def _build_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "generate_solution",
                    "description": "Ask the reasoner to generate or improve a C++17 solution. Use feedback to describe what went wrong. Use instructions to enforce a specific algorithm or approach.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "feedback": {
                                "type": "string",
                                "description": "Brief feedback about what went wrong with previous solution (optional, only for improvements)",
                            },
                            "instructions": {
                                "type": "string",
                                "description": "Strict instructions for the reasoner about what approach/algorithm to use.",
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

    def _get_agent_inference_params(self):
        """Get inference params for the orchestrator (thinking disabled)."""
        params = asdict(self.cfg.inference)
        extra_body = dict(params.get("extra_body", {}) or {})
        extra_body["chat_template_kwargs"] = {"thinking": False}
        params["extra_body"] = extra_body
        return params

    def _get_reasoner_inference_params(self):
        """Get inference params for the reasoner (thinking enabled)."""
        params = asdict(self.cfg.inference_reasoner)
        extra_body = dict(params.get("extra_body", {}) or {})
        extra_body["chat_template_kwargs"] = {"thinking": True}
        params["extra_body"] = extra_body
        return params

    async def _agent_turn(self, messages: list[dict]) -> dict:
        tools = self._build_tools()
        try:
            out = await self.generate_with_semaphore(
                prompt=messages, tools=tools, include_response=True, **self._get_agent_inference_params()
            )
        except Exception as e:
            if is_context_window_exceeded_error(e):
                return {"message": None}
            raise

        response = out.get("response")
        if response is None:
            return {"message": None}

        message = response.choices[0].message
        tool_calls = []
        tool_call_ids = []
        if message.tool_calls:
            tool_calls = [{tc.function.name: tc.function.arguments} for tc in message.tool_calls]
            tool_call_ids = [tc.id for tc in message.tool_calls]

        return {
            "message": message,
            "tool_calls": tool_calls,
            "tool_call_ids": tool_call_ids,
            "num_generated_tokens": out.get("num_generated_tokens", 0),
        }

    async def _call_reasoner(
        self,
        problem: str,
        previous_solution: str | None,
        feedback: str | None,
        instructions: str | None,
        data_point: dict,
    ) -> dict:
        if previous_solution:
            messages = self.reasoner_improve_prompt.fill(
                {
                    "subtask_score": data_point.get("subtask_score", "1"),
                    "question": problem,
                    "solution": previous_solution,
                    "feedback": feedback or "",
                }
            )
        else:
            messages = self.reasoner_prompt.fill(
                {"subtask_score": data_point.get("subtask_score", "1"), "question": problem}
            )

        # Inject agent instructions into the system message
        if instructions and messages and messages[0].get("role") == "system":
            messages[0]["content"] += (
                "\n\n### STRICT INSTRUCTIONS ###\n"
                "You MUST follow these instructions when creating your solution. "
                "They take priority over your default approach.\n\n"
                f"{instructions}"
            )
            self.dp_print(data_point, f"Injected instructions into reasoner prompt: {instructions}")

        async with self.semaphore:
            result = await self.llm.generate_async(
                prompt=messages, include_response=False, **self._get_reasoner_inference_params()
            )

        return {
            "generation": result.get("generation", ""),
            "reasoning_content": result.get("reasoning_content", ""),
            "num_generated_tokens": result.get("num_generated_tokens", 0),
        }

    async def _execute_generate_solution(
        self, problem: str, previous_solution: str | None, args: dict, data_point: dict, tool_call_id
    ) -> dict:
        feedback = args.get("feedback")
        instructions = args.get("instructions")

        self.dp_print(
            data_point,
            f"tool: generate_solution(feedback={'yes' if feedback else 'no'}, "
            f"instructions={'yes' if instructions else 'no'})",
        )

        result = await self._call_reasoner(problem, previous_solution, feedback, instructions, data_point)

        code = self._extract_cpp(result.get("generation", ""))
        trace_entries = [
            {
                "source": "reasoner",
                "role": "assistant",
                "content": result.get("generation", ""),
                "reasoning_content": result.get("reasoning_content", ""),
                "feedback": feedback,
                "instructions": instructions,
                "previous_solution": previous_solution,
            }
        ]

        if code:
            tool_out = json.dumps({"solution": code, "status": "success"})
            self.dp_print(data_point, f"reasoner: found valid solution ({len(code)} chars)")
        else:
            tool_out = json.dumps({"error": "No cpp code block found in generation", "status": "error"})
            self.dp_print(data_point, "reasoner: failed to generate code block")

        return {
            "name": "generate_solution",
            "tool_call_id": tool_call_id,
            "tool_out": tool_out,
            "trace_entries": trace_entries,
            "reasoner_tokens": [result.get("num_generated_tokens", 0)],
            "first_valid_code": code,
        }

    async def _execute_submit_solution(self, args: dict, data_point: dict, tool_call_id) -> dict:
        code = args.get("code", "")
        sample = bool(args.get("sample", False))

        if not code:
            tool_out = json.dumps({"error": "No code provided"})
            self.dp_print(data_point, "tool: submit_solution - no code provided")
            return {
                "name": "submit_solution",
                "tool_call_id": tool_call_id,
                "tool_out": tool_out,
                "trace_entries": [],
                "final_code": None,
            }

        self.dp_print(data_point, f"tool: submit_solution(sample={sample}, code_len={len(code)})")

        eval_payload = {
            **data_point,
            "generation": f"```cpp\n{code}\n```",
            "only_sample_tests": sample,
        }
        eval_result = await self.evaluator.eval_single(eval_payload)
        test_case_results = eval_result.get("test_case_results", {})

        is_ioi = "ioi_id" in data_point
        if is_ioi and "subtask" in data_point:
            subtask_name = data_point["subtask"]
            if subtask_name in test_case_results:
                test_case_results = {subtask_name: test_case_results[subtask_name]}

        normalized = self._normalize_scores(test_case_results)

        if self.cfg.explicit_feedback:
            tool_out_dict = {**eval_result, "test_case_results": self._filter_test_outputs(test_case_results)}
        else:
            if is_ioi and "subtask_score" in data_point:
                max_score = data_point["subtask_score"]
                subtask_scores = {k: f"{v['score']}/{max_score}" for k, v in normalized.items()}
            else:
                subtask_scores = {k: v["score"] for k, v in normalized.items()}
            tool_out_dict = {"subtask_scores": subtask_scores}

        if self.cfg.avg_score:
            tool_out_dict["avg_score"] = self._calculate_avg_score(normalized)

        if is_ioi and "subtask_score" in data_point:
            max_score = float(data_point["subtask_score"])
            success = bool(normalized) and all(float(v["score"]) == max_score for v in normalized.values())
        else:
            success = bool(normalized) and all(float(v["score"]) == 1.0 for v in normalized.values())
        tool_out_dict["success"] = success

        tool_out = json.dumps(tool_out_dict)
        self.dp_print(data_point, f"result: success={success}")

        final_code = None
        if success and not sample:
            final_code = code
            self.dp_print(data_point, "solution accepted")

        return {
            "name": "submit_solution",
            "tool_call_id": tool_call_id,
            "tool_out": tool_out,
            "trace_entries": [{"source": "evaluator", "eval_result": eval_result, "code": code, "sample": sample}],
            "final_code": final_code,
        }

    async def _summarize_progress(self, agent_messages: list[dict]) -> str:
        summary_messages_template = self.summary_prompt.fill({})
        summary_messages = [summary_messages_template[0]] + agent_messages + [summary_messages_template[1]]

        try:
            result = await self.llm.generate_async(
                prompt=summary_messages, include_response=False, **self._get_agent_inference_params()
            )
            return result.get("generation", "").strip()
        except Exception:
            return "Previous attempts exhausted context window."

    async def process_single_datapoint(self, data_point, all_data):
        if self.evaluator is None:
            raise ValueError(
                "ReasoningSingleAgent requires an evaluator supporting eval_single "
                "(set ++eval_type and ++eval_config)."
            )

        if data_point.get("subtask_score") is None:
            data_point["subtask_score"] = "1"

        problem = data_point["question"]
        max_time_seconds = self._parse_max_time(self.cfg.max_time)
        start_time = time.time()

        agent_messages = [
            {"role": "system", "content": self.agent_prompt.config.system},
            {"role": "user", "content": f"Problem:\n{problem}"},
        ]
        trace = []
        num_agent_tokens, num_reasoner_tokens = [], []
        final_code = ""
        previous_solution = None

        for step in range(self.cfg.max_steps):
            if max_time_seconds is not None and (time.time() - start_time) >= max_time_seconds:
                self.dp_print(data_point, f"max_time reached at step {step}")
                break

            self.dp_print(data_point, f"step {step + 1}/{self.cfg.max_steps}")

            result = await self._agent_turn(agent_messages)
            if result["message"] is None:
                self.dp_print(data_point, "context window exceeded, generating summary and restarting")

                summary = await self._summarize_progress(agent_messages)
                self.dp_print(data_point, f"summary: {summary[:100]}...")

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
                        "iteration": step,
                    }
                )
                continue

            num_agent_tokens.append(result.get("num_generated_tokens", 0))

            msg = result["message"]
            if hasattr(msg, "model_dump"):
                msg = msg.model_dump()
            agent_messages.append(msg)
            trace.append({"source": "agent", **msg})

            tool_calls = result.get("tool_calls", [])
            tool_call_ids = result.get("tool_call_ids", [])

            if not tool_calls:
                self.dp_print(data_point, "no tool calls, ending")
                break

            # Build coroutines for parallel execution
            coros = []
            for tc, tc_id in zip(tool_calls, tool_call_ids):
                name, raw_args = next(iter(tc.items()))
                args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args) if raw_args else {}

                if name == "generate_solution":
                    coros.append(self._execute_generate_solution(problem, previous_solution, args, data_point, tc_id))
                elif name == "submit_solution":
                    coros.append(self._execute_submit_solution(args, data_point, tc_id))
                else:

                    async def _unknown_tool(n=name, tid=tc_id):
                        tool_out = json.dumps({"error": f"Unknown tool: {n}"})
                        return {"name": n, "tool_call_id": tid, "tool_out": tool_out, "trace_entries": []}

                    coros.append(_unknown_tool())

            tool_results = await asyncio.gather(*coros)

            # Apply results in order
            for tr in tool_results:
                trace.extend(tr.get("trace_entries", []))

                tool_msg = {"role": "tool", "content": tr["tool_out"], "tool_call_id": tr["tool_call_id"]}
                agent_messages.append(tool_msg)
                trace.append({"source": "tool", **tool_msg})

                if tr["name"] == "generate_solution":
                    num_reasoner_tokens.extend(tr.get("reasoner_tokens", []))
                    if tr.get("first_valid_code"):
                        previous_solution = tr["first_valid_code"]
                elif tr["name"] == "submit_solution" and tr.get("final_code"):
                    final_code = tr["final_code"]

            if final_code:
                break

        if not final_code and previous_solution:
            final_code = previous_solution
            self.dp_print(data_point, "using last generated solution")

        return {
            "id": data_point["id"],
            "generation": f"```cpp\n{final_code}\n```",
            "messages": trace,
            "num_generated_tokens": sum(num_agent_tokens) + sum(num_reasoner_tokens),
            "num_generated_tokens_list": {"agent": num_agent_tokens, "reasoner": num_reasoner_tokens},
        }


GENERATION_TASK_CLASS = ReasoningSingleAgentTask


@hydra.main(version_base=None, config_name="base_reasoning_single_agent_config")
def reasoning_single_agent_generation(cfg: ReasoningSingleAgentConfig):
    cfg = ReasoningSingleAgentConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)
    task = ReasoningSingleAgentTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(ReasoningSingleAgentConfig, server_params=server_params())

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        reasoning_single_agent_generation()
