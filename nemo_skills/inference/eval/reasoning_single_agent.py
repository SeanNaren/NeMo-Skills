import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict, field

import hydra

from nemo_skills.inference.eval.agent_utils import (
    extract_cpp,
    parse_max_time,
    process_submission_result,
    sanitize_message,
)
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
        """Get inference params for the reasoner (thinking enabled).

        Sets tokens_to_generate=None so the server auto-caps to the remaining
        context window instead of requesting a fixed budget that may overflow.
        """
        params = asdict(self.cfg.inference_reasoner)
        extra_body = dict(params.get("extra_body", {}) or {})
        extra_body["chat_template_kwargs"] = {"thinking": True}
        params["extra_body"] = extra_body
        params["tokens_to_generate"] = None
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

        code = extract_cpp(result.get("generation", ""))
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
                "target_score": 0.0,
            }

        self.dp_print(data_point, f"tool: submit_solution(sample={sample}, code_len={len(code)})")

        eval_payload = {
            **data_point,
            "generation": f"```cpp\n{code}\n```",
            "only_sample_tests": sample,
        }
        eval_result = await self.evaluator.eval_single(eval_payload)

        result = process_submission_result(
            eval_result,
            data_point,
            explicit_feedback=self.cfg.explicit_feedback,
            avg_score=self.cfg.avg_score,
            max_limit_in_test_output=self.cfg.max_limit_in_test_output,
        )

        success = result["success"]
        tool_out = result["tool_output"]
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
            "target_score": result["target_score"] if not sample else 0.0,
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

    def _init_state(self, data_point):
        """Initialize fresh loop state for a data point."""
        problem = data_point["question"]
        return {
            "agent_messages": [
                {"role": "system", "content": self.agent_prompt.config.system},
                {"role": "user", "content": f"Problem:\n{problem}"},
            ],
            "trace": [],
            "num_agent_tokens": [],
            "num_reasoner_tokens": [],
            "final_code": "",
            "previous_solution": None,
            "best_code": None,
            "best_score": 0.0,
            "step": 0,
        }

    async def process_single_datapoint(self, data_point, all_data):
        if self.evaluator is None:
            raise ValueError(
                "ReasoningSingleAgent requires an evaluator supporting eval_single "
                "(set ++eval_type and ++eval_config)."
            )

        if data_point.get("subtask_score") is None:
            data_point["subtask_score"] = "1"

        problem = data_point["question"]
        max_time_seconds = parse_max_time(self.cfg.max_time)
        start_time = time.time()

        # Load intermediate state if available
        async_position = data_point.get(self.cfg.async_position_key)
        saved_state = self.load_intermediate_state(async_position) if async_position is not None else None

        if saved_state:
            self.dp_print(data_point, f"resuming from step {saved_state['step']}")
            s = saved_state
        else:
            self.dp_print(data_point, "start")
            s = self._init_state(data_point)

        agent_messages = s["agent_messages"]
        trace = s["trace"]
        num_agent_tokens = s["num_agent_tokens"]
        num_reasoner_tokens = s["num_reasoner_tokens"]
        final_code = s["final_code"]
        previous_solution = s["previous_solution"]
        best_code = s.get("best_code")
        best_score = s.get("best_score", 0.0)

        for step in range(s["step"], self.cfg.max_steps):
            if max_time_seconds is not None and (time.time() - start_time) >= max_time_seconds:
                self.dp_print(data_point, f"max_time reached at step {step}")
                if async_position is not None:
                    self._save_state(async_position, step, locals())
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
            msg = sanitize_message(msg)
            agent_messages.append(msg)
            trace.append({"source": "agent", **msg})

            tool_calls = result.get("tool_calls", [])
            tool_call_ids = result.get("tool_call_ids", [])

            if not tool_calls:
                if not final_code:
                    nudge = (
                        "We still do not have a successful submitted solution. "
                        "Please continue till we reach a successfully submitted solution that passes all tests."
                    )
                    self.dp_print(data_point, f"no tool calls, nudging agent (step {step + 1})")
                    nudge_msg = {"role": "user", "content": nudge}
                    agent_messages.append(nudge_msg)
                    trace.append({"source": "system", **nudge_msg})
                    continue
                break

            # Build coroutines for parallel execution
            coros = []
            for tc, tc_id in zip(tool_calls, tool_call_ids):
                name, raw_args = next(iter(tc.items()))
                if isinstance(raw_args, dict):
                    args = raw_args
                elif raw_args:
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        self.dp_print(data_point, f"malformed tool args for {name}, skipping")
                        args = None
                else:
                    args = {}

                if args is None:
                    tool_out = json.dumps({"error": "Malformed tool call arguments"})

                    async def _bad_args(tout=tool_out, tid=tc_id):
                        return {
                            "name": "error",
                            "tool_call_id": tid,
                            "tool_out": tout,
                            "trace_entries": [],
                            "target_score": 0.0,
                        }

                    coros.append(_bad_args())
                    continue

                if name == "generate_solution":
                    coros.append(self._execute_generate_solution(problem, previous_solution, args, data_point, tc_id))
                elif name == "submit_solution":
                    coros.append(self._execute_submit_solution(args, data_point, tc_id))
                else:

                    async def _unknown_tool(n=name, tid=tc_id):
                        tool_out = json.dumps({"error": f"Unknown tool: {n}"})
                        return {
                            "name": n,
                            "tool_call_id": tid,
                            "tool_out": tool_out,
                            "trace_entries": [],
                            "target_score": 0.0,
                        }

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
                elif tr["name"] == "submit_solution":
                    target_score = tr.get("target_score", 0.0)
                    if target_score > best_score:
                        best_score = target_score
                        best_code = args.get("code", "")
                    if tr.get("final_code"):
                        final_code = tr["final_code"]

            # Save intermediate state after each step
            if async_position is not None:
                self._save_state(async_position, step + 1, locals())

            if final_code:
                break

        if not final_code:
            if best_code:
                final_code = best_code
                self.dp_print(data_point, "using best scoring solution")
            elif previous_solution:
                final_code = previous_solution
                self.dp_print(data_point, "using last generated solution")

        # Clear intermediate state when done
        if async_position is not None:
            self.clear_intermediate_state(async_position)

        return {
            "id": data_point["id"],
            "generation": f"```cpp\n{final_code}\n```",
            "messages": trace,
            "num_generated_tokens": sum(num_agent_tokens) + sum(num_reasoner_tokens),
            "num_generated_tokens_list": {"agent": num_agent_tokens, "reasoner": num_reasoner_tokens},
        }

    def _save_state(self, async_position, step, local_vars):
        """Save intermediate state for resume."""
        state = {
            "agent_messages": local_vars["agent_messages"],
            "trace": local_vars["trace"],
            "num_agent_tokens": local_vars["num_agent_tokens"],
            "num_reasoner_tokens": local_vars["num_reasoner_tokens"],
            "final_code": local_vars["final_code"],
            "previous_solution": local_vars["previous_solution"],
            "best_code": local_vars["best_code"],
            "best_score": local_vars["best_score"],
            "step": step,
        }
        self.save_intermediate_state(async_position, state)


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
