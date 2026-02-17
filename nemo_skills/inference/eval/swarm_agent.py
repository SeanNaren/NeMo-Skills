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

SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_solution",
        "description": "Compile and run C++17 code against test cases. Use sample=true to run only sample tests first.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "C++17 source code"},
                "sample": {"type": "boolean", "description": "Run only sample tests", "default": False},
            },
            "required": ["code"],
        },
    },
}

CREATE_SUBAGENT_TOOL = {
    "type": "function",
    "function": {
        "name": "create_subagent",
        "description": "Create a custom subagent with specific system prompt and name for reuse.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Unique name for this agent configuration"},
                "system_prompt": {
                    "type": "string",
                    "description": "System prompt defining the agent's role, capabilities, and boundaries",
                },
            },
            "required": ["name", "system_prompt"],
        },
    },
}

ASSIGN_TASK_TOOL = {
    "type": "function",
    "function": {
        "name": "assign_task",
        "description": (
            "Launch a new agent.\nUsage notes:\n"
            "1. You can launch multiple agents concurrently whenever possible, to maximize performance;\n"
            "2. When the agent is done, it will return a single message back to you."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "agent": {"type": "string", "description": "Specify which created agent to use."},
                "prompt": {"type": "string", "description": "The task for the agent to perform"},
            },
            "required": ["agent", "prompt"],
        },
    },
}

ORCHESTRATOR_TOOLS = [CREATE_SUBAGENT_TOOL, ASSIGN_TASK_TOOL, SUBMIT_TOOL]
SUBAGENT_TOOLS = [SUBMIT_TOOL]


@nested_dataclass(kw_only=True)
class SwarmAgentConfig(GenerationTaskConfig):
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    inference_subagent: InferenceConfig = field(default_factory=InferenceConfig)
    server: dict = field(default_factory=dict)
    max_steps: int = 100
    max_subagent_steps: int = 10
    max_time: str | None = None  # Format: "hh:mm:ss"
    explicit_feedback: bool = False
    max_limit_in_test_output: int = 1000
    avg_score: bool = True
    agent_prompt_config: str = "eval/ioi/agent/swarm_orchestrator"
    summary_prompt_config: str = "eval/ioi/agent/summary"


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_swarm_agent_config", node=SwarmAgentConfig)


class SwarmAgentTask(GenerationTask):
    def __init__(self, cfg: SwarmAgentConfig):
        super().__init__(cfg)
        self.agent_prompt = get_prompt(cfg.agent_prompt_config)
        self.summary_prompt = get_prompt(cfg.summary_prompt_config)

    def setup_prompt(self):
        return None

    def log_example_prompt(self, data):
        return

    def dp_print(self, data_point, *args):
        dp_id = data_point.get("id", "?") if isinstance(data_point, dict) else "?"
        print(f"[{dp_id}]", *args)

    @staticmethod
    def _sanitize_message(msg: dict) -> dict:
        """Ensure tool_call arguments in assistant messages are valid JSON.

        Models sometimes generate invalid JSON escape sequences in tool call
        arguments (e.g. \\q, \\0 from C++ code). When these messages are sent
        back to the API, the server fails to parse the nested JSON.
        """
        tool_calls = msg.get("tool_calls")
        if msg.get("role") != "assistant" or not tool_calls:
            return msg
        for tc in tool_calls:
            func = tc.get("function") or {}
            args_str = func.get("arguments")
            if not isinstance(args_str, str) or not args_str:
                continue
            try:
                json.loads(args_str)
            except json.JSONDecodeError:
                fixed = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", args_str)
                try:
                    parsed = json.loads(fixed)
                    func["arguments"] = json.dumps(parsed)
                except json.JSONDecodeError:
                    func["arguments"] = "{}"
        return msg

    def _extract_cpp(self, text: str | None) -> str | None:
        if not text:
            return None
        matches = re.findall(r"```(?:cpp|c\+\+)\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        return matches[-1].strip() if matches else None

    def _normalize_scores(self, test_case_results: dict) -> dict:
        if (
            isinstance(test_case_results, dict)
            and "outputs" in test_case_results
            and "score" in test_case_results
            and isinstance(test_case_results.get("outputs"), list)
        ):
            return {"overall": {"score": float(test_case_results["score"]), "outputs": test_case_results["outputs"]}}
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

        if (
            isinstance(test_case_results, dict)
            and "outputs" in test_case_results
            and "score" in test_case_results
            and isinstance(test_case_results.get("outputs"), list)
        ):
            return {**test_case_results, "outputs": filter_outputs(test_case_results["outputs"])}
        return {k: {**v, "outputs": filter_outputs(v.get("outputs", []))} for k, v in test_case_results.items()}

    def _parse_max_time(self, max_time_str: str | None) -> float | None:
        if not max_time_str:
            return None
        parts = max_time_str.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid max_time format: {max_time_str}. Expected hh:mm:ss")
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds

    def _get_orchestrator_inference_params(self):
        """Get inference params for the orchestrator.

        Does not force thinking on or off - lets the server/config handle it.
        Forcing thinking=False caused some models (e.g. Kimi K2.5) to emit
        raw tool-call tokens as text instead of using the tool-calling API.
        """
        params = asdict(self.cfg.inference)
        extra_body = dict(params.get("extra_body", {}) or {})
        params["extra_body"] = extra_body
        return params

    def _get_subagent_inference_params(self):
        """Get inference params for sub-agents (thinking enabled).

        Sets tokens_to_generate=None so the server auto-caps to the remaining
        context window instead of requesting a fixed budget that may overflow.
        """
        params = asdict(self.cfg.inference_subagent)
        extra_body = dict(params.get("extra_body", {}) or {})
        extra_body["chat_template_kwargs"] = {"thinking": True}
        params["extra_body"] = extra_body
        params["tokens_to_generate"] = None
        return params

    async def _orchestrator_turn(self, messages: list[dict]) -> dict:
        """Call the orchestrator model with all orchestrator tools."""
        try:
            out = await self.generate_with_semaphore(
                prompt=messages,
                tools=ORCHESTRATOR_TOOLS,
                include_response=True,
                **self._get_orchestrator_inference_params(),
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

    async def _subagent_turn(self, messages: list[dict]) -> dict:
        """Call a sub-agent model with submit_solution tool only."""
        try:
            out = await self.generate_with_semaphore(
                prompt=messages, tools=SUBAGENT_TOOLS, include_response=True, **self._get_subagent_inference_params()
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

    async def _execute_submit(
        self, code: str, sample: bool, data_point: dict, submission_counts: dict
    ) -> tuple[str, bool]:
        """Run submission evaluation and return (tool_output_json, accepted).

        Also increments submission_counts['sample'] or submission_counts['full'].
        """
        if sample:
            submission_counts["sample"] = submission_counts.get("sample", 0) + 1
        else:
            submission_counts["full"] = submission_counts.get("full", 0) + 1
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

        return json.dumps(tool_out_dict), success and not sample

    def _build_subagent_system_prompt(self, user_system_prompt: str) -> str:
        """Augment the user-provided sub-agent system prompt with tool instructions."""
        tool_instructions = (
            "\n\n# Tools\n"
            "You have access to a `submit_solution` tool that compiles and runs C++17 code against test cases.\n"
            "- Call submit_solution(code, sample=true) to quickly test on sample inputs.\n"
            "- If samples pass, call submit_solution(code, sample=false) to run full tests.\n"
            "- If tests fail, analyze the feedback, fix your code, and resubmit.\n"
            "- You MUST use submit_solution to test your code. Do NOT just output code in text.\n"
            "- Iterate until you get a fully passing solution or run out of steps."
        )
        return user_system_prompt + tool_instructions

    async def _run_subagent(
        self,
        agent_name: str,
        system_prompt: str,
        task_prompt: str,
        data_point: dict,
        start_time: float,
        max_time_seconds: float | None,
        submission_counts: dict,
    ) -> dict:
        """Run a sub-agent loop: the sub-agent generates code and iterates using submit_solution."""
        full_system_prompt = self._build_subagent_system_prompt(system_prompt)
        messages = [
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": task_prompt},
        ]
        trace = []
        num_tokens = []
        final_code = None
        last_code = None

        for step in range(self.cfg.max_subagent_steps):
            if max_time_seconds is not None and (time.time() - start_time) >= max_time_seconds:
                self.dp_print(data_point, f"  subagent '{agent_name}' step {step + 1}: max_time reached")
                break

            self.dp_print(data_point, f"  subagent '{agent_name}' step {step + 1}/{self.cfg.max_subagent_steps}")

            result = await self._subagent_turn(messages)
            if result["message"] is None:
                self.dp_print(data_point, f"  subagent '{agent_name}' step {step + 1}: context exceeded")
                break

            num_tokens.append(result.get("num_generated_tokens", 0))

            msg = result["message"]
            if hasattr(msg, "model_dump"):
                msg = msg.model_dump()
            msg = self._sanitize_message(msg)
            messages.append(msg)
            trace.append({"source": "subagent", **msg})

            tool_calls = result.get("tool_calls", [])
            tool_call_ids = result.get("tool_call_ids", [])

            if not tool_calls:
                # Sub-agent may have produced code in text without a tool call
                code = self._extract_cpp(msg.get("content", ""))
                if code:
                    last_code = code
                    self.dp_print(
                        data_point, f"  subagent '{agent_name}' step {step + 1}: produced code in text (no tool call)"
                    )
                else:
                    self.dp_print(data_point, f"  subagent '{agent_name}' step {step + 1}: no tool calls and no code")
                break

            for tc, tc_id in zip(tool_calls, tool_call_ids):
                name, raw_args = next(iter(tc.items()))
                if isinstance(raw_args, dict):
                    args = raw_args
                elif raw_args:
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        self.dp_print(data_point, f"  subagent '{agent_name}' step {step + 1}: malformed tool args")
                        tool_out = json.dumps({"error": "Malformed tool call arguments"})
                        tool_msg = {"role": "tool", "content": tool_out, "tool_call_id": tc_id}
                        messages.append(tool_msg)
                        trace.append({"source": "tool", **tool_msg})
                        continue
                else:
                    args = {}

                if name == "submit_solution":
                    code = args.get("code", "")
                    sample = bool(args.get("sample", False))
                    if code:
                        last_code = code
                    self.dp_print(
                        data_point,
                        f"  subagent '{agent_name}' step {step + 1}: submit_solution(sample={sample})",
                    )
                    tool_out, accepted = await self._execute_submit(code, sample, data_point, submission_counts)
                    if accepted:
                        final_code = code
                        self.dp_print(data_point, f"  subagent '{agent_name}' step {step + 1}: ACCEPTED")
                else:
                    tool_out = json.dumps({"error": f"Unknown tool: {name}"})

                tool_msg = {"role": "tool", "content": tool_out, "tool_call_id": tc_id}
                messages.append(tool_msg)
                trace.append({"source": "tool", **tool_msg})

            if final_code:
                break

        return {
            "accepted_code": final_code,
            "last_code": last_code,
            "accepted": final_code is not None,
            "trace": trace,
            "num_tokens": num_tokens,
        }

    async def _summarize_progress(self, agent_messages: list[dict]) -> str:
        summary_messages_template = self.summary_prompt.fill({})
        summary_messages = [summary_messages_template[0]] + agent_messages + [summary_messages_template[1]]

        try:
            result = await self.llm.generate_async(
                prompt=summary_messages, include_response=False, **self._get_orchestrator_inference_params()
            )
            return result.get("generation", "").strip()
        except Exception:
            return "Previous attempts exhausted context window."

    async def process_single_datapoint(self, data_point, all_data):
        if self.evaluator is None:
            raise ValueError(
                "SwarmAgent requires an evaluator supporting eval_single (set ++eval_type and ++eval_config)."
            )

        if data_point.get("subtask_score") is None:
            data_point["subtask_score"] = "1"

        problem = data_point["question"]
        max_time_seconds = self._parse_max_time(self.cfg.max_time)
        start_time = time.time()

        # Load intermediate state if available
        async_position = data_point.get(self.cfg.async_position_key)
        saved_state = self.load_intermediate_state(async_position) if async_position is not None else None

        if saved_state:
            self.dp_print(data_point, f"resuming from step {saved_state['step']}")
            agent_messages = saved_state["agent_messages"]
            trace = saved_state["trace"]
            num_orchestrator_tokens = saved_state["num_orchestrator_tokens"]
            num_subagent_tokens = saved_state["num_subagent_tokens"]
            final_code = saved_state["final_code"]
            last_code = saved_state["last_code"]
            subagents = saved_state["subagents"]
            submission_counts = saved_state["submission_counts"]
            start_step = saved_state["step"]
        else:
            self.dp_print(data_point, "start orchestration")
            agent_messages = [
                {"role": "system", "content": self.agent_prompt.config.system},
                {"role": "user", "content": f"Problem:\n{problem}"},
            ]
            trace = []
            num_orchestrator_tokens = []
            num_subagent_tokens = []
            final_code = ""
            last_code = None
            subagents = {}
            submission_counts = {"sample": 0, "full": 0}
            start_step = 0

        for step in range(start_step, self.cfg.max_steps):
            if max_time_seconds is not None and (time.time() - start_time) >= max_time_seconds:
                self.dp_print(data_point, f"max_time reached at step {step}")
                if async_position is not None:
                    self._save_state(
                        async_position,
                        step,
                        agent_messages,
                        trace,
                        num_orchestrator_tokens,
                        num_subagent_tokens,
                        final_code,
                        last_code,
                        subagents,
                        submission_counts,
                    )
                break

            self.dp_print(data_point, f"step {step + 1}/{self.cfg.max_steps}")

            result = await self._orchestrator_turn(agent_messages)
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

            num_orchestrator_tokens.append(result.get("num_generated_tokens", 0))

            msg = result["message"]
            if hasattr(msg, "model_dump"):
                msg = msg.model_dump()
            msg = self._sanitize_message(msg)
            agent_messages.append(msg)
            trace.append({"source": "orchestrator", **msg})

            tool_calls = result.get("tool_calls", [])
            tool_call_ids = result.get("tool_call_ids", [])

            if not tool_calls:
                # Check if orchestrator produced code directly
                code = self._extract_cpp(msg.get("content", ""))
                if code:
                    last_code = code

                if not final_code:
                    # No successful submission yet - nudge the orchestrator to keep going
                    nudge = (
                        "We still do not have a successful submitted solution. "
                        "Please continue till we reach a successfully submitted solution that passes all tests."
                    )
                    self.dp_print(data_point, f"no tool calls, nudging orchestrator (step {step + 1})")
                    nudge_msg = {"role": "user", "content": nudge}
                    agent_messages.append(nudge_msg)
                    trace.append({"source": "system", **nudge_msg})
                    continue
                break

            # Separate tool calls into sync (create_subagent) and async (assign_task, submit_solution)
            sync_calls = []
            async_coros = []

            for tc, tc_id in zip(tool_calls, tool_call_ids):
                name, raw_args = next(iter(tc.items()))
                if isinstance(raw_args, dict):
                    args = raw_args
                elif raw_args:
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        self.dp_print(data_point, f"malformed tool args for {name}, skipping")
                        tool_out = json.dumps({"error": "Malformed tool call arguments"})
                        tool_msg = {"role": "tool", "content": tool_out, "tool_call_id": tc_id}
                        agent_messages.append(tool_msg)
                        trace.append({"source": "tool", **tool_msg})
                        continue
                else:
                    args = {}

                if name == "create_subagent":
                    sync_calls.append((tc_id, name, args))
                elif name == "assign_task":
                    async_coros.append((tc_id, name, args))
                elif name == "submit_solution":
                    async_coros.append((tc_id, name, args))
                else:
                    sync_calls.append((tc_id, name, args))

            # Process create_subagent calls first (they're instant)
            for tc_id, name, args in sync_calls:
                if name == "create_subagent":
                    agent_name = args.get("name", "")
                    system_prompt = args.get("system_prompt", "")
                    subagents[agent_name] = system_prompt
                    tool_out = json.dumps({"status": "success", "agent": agent_name})
                    self.dp_print(data_point, f"created subagent: {agent_name}")
                else:
                    tool_out = json.dumps({"error": f"Unknown tool: {name}"})

                tool_msg = {"role": "tool", "content": tool_out, "tool_call_id": tc_id}
                agent_messages.append(tool_msg)
                trace.append({"source": "tool", **tool_msg})

            # Process assign_task and submit_solution calls in parallel
            if async_coros:
                coros = []
                coro_ids = []
                coro_names = []

                for tc_id, name, args in async_coros:
                    if name == "assign_task":
                        agent_name = args.get("agent", "")
                        task_prompt = args.get("prompt", "")
                        system_prompt = subagents.get(agent_name, "")

                        if not system_prompt:
                            tool_out = json.dumps({"error": f"Agent '{agent_name}' not found. Create it first."})
                            tool_msg = {"role": "tool", "content": tool_out, "tool_call_id": tc_id}
                            agent_messages.append(tool_msg)
                            trace.append({"source": "tool", **tool_msg})
                            continue

                        self.dp_print(data_point, f"assign_task to '{agent_name}': {task_prompt[:80]}...")
                        coros.append(
                            self._run_subagent(
                                agent_name,
                                system_prompt,
                                task_prompt,
                                data_point,
                                start_time,
                                max_time_seconds,
                                submission_counts,
                            )
                        )
                        coro_ids.append(tc_id)
                        coro_names.append(("assign_task", agent_name))

                    elif name == "submit_solution":
                        code = args.get("code", "")
                        sample = bool(args.get("sample", False))

                        if not code:
                            tool_out = json.dumps({"error": "No code provided"})
                            tool_msg = {"role": "tool", "content": tool_out, "tool_call_id": tc_id}
                            agent_messages.append(tool_msg)
                            trace.append({"source": "tool", **tool_msg})
                            continue

                        self.dp_print(data_point, f"submit_solution(sample={sample}, code_len={len(code)})")

                        async def _do_submit(c=code, s=sample, dp=data_point, sc=submission_counts):
                            tool_out_str, accepted = await self._execute_submit(c, s, dp, sc)
                            return {"tool_out": tool_out_str, "accepted": accepted, "code": c}

                        coros.append(_do_submit())
                        coro_ids.append(tc_id)
                        coro_names.append(("submit_solution", None))

                if coros:
                    results = await asyncio.gather(*coros)

                    for (coro_name, agent_name), tc_id, res in zip(coro_names, coro_ids, results):
                        if coro_name == "assign_task":
                            subagent_result = res
                            trace.extend(subagent_result.get("trace", []))
                            num_subagent_tokens.extend(subagent_result.get("num_tokens", []))

                            sa_accepted = subagent_result.get("accepted", False)
                            sa_accepted_code = subagent_result.get("accepted_code")
                            sa_last = subagent_result.get("last_code")

                            if sa_accepted and sa_accepted_code:
                                tool_out = json.dumps(
                                    {"status": "accepted", "solution": sa_accepted_code, "agent": agent_name}
                                )
                                final_code = sa_accepted_code
                                last_code = sa_accepted_code
                            elif sa_last:
                                tool_out = json.dumps(
                                    {"status": "not_accepted", "solution": sa_last, "agent": agent_name}
                                )
                                last_code = sa_last
                            else:
                                tool_out = json.dumps(
                                    {
                                        "status": "failed",
                                        "error": "Sub-agent did not produce a solution",
                                        "agent": agent_name,
                                    }
                                )

                            self.dp_print(
                                data_point,
                                f"subagent '{agent_name}' returned: "
                                f"accepted={'yes' if sa_accepted else 'no'}, code={'yes' if sa_last else 'no'}",
                            )

                        elif coro_name == "submit_solution":
                            tool_out = res["tool_out"]
                            if res["accepted"]:
                                final_code = res["code"]
                                self.dp_print(data_point, "solution accepted")
                            if res["code"]:
                                last_code = res["code"]

                        tool_msg = {"role": "tool", "content": tool_out, "tool_call_id": tc_id}
                        agent_messages.append(tool_msg)
                        trace.append({"source": "tool", **tool_msg})

            # Save intermediate state after each orchestrator step
            if async_position is not None:
                self._save_state(
                    async_position,
                    step + 1,
                    agent_messages,
                    trace,
                    num_orchestrator_tokens,
                    num_subagent_tokens,
                    final_code,
                    last_code,
                    subagents,
                    submission_counts,
                )

            if final_code:
                break

        if not final_code and last_code:
            final_code = last_code
            self.dp_print(data_point, "using last available solution")

        self.dp_print(
            data_point,
            f"done: submissions(sample={submission_counts['sample']}, full={submission_counts['full']})",
        )

        # Clear intermediate state when done
        if async_position is not None:
            self.clear_intermediate_state(async_position)

        return {
            "id": data_point["id"],
            "generation": f"```cpp\n{final_code}\n```",
            "messages": trace,
            "num_generated_tokens": sum(num_orchestrator_tokens) + sum(num_subagent_tokens),
            "num_generated_tokens_list": {"orchestrator": num_orchestrator_tokens, "subagent": num_subagent_tokens},
            "num_submissions": submission_counts,
        }

    def _save_state(
        self,
        async_position,
        step,
        agent_messages,
        trace,
        num_orchestrator_tokens,
        num_subagent_tokens,
        final_code,
        last_code,
        subagents,
        submission_counts,
    ):
        """Save intermediate state for resume."""
        state = {
            "agent_messages": agent_messages,
            "trace": trace,
            "num_orchestrator_tokens": num_orchestrator_tokens,
            "num_subagent_tokens": num_subagent_tokens,
            "final_code": final_code,
            "last_code": last_code,
            "subagents": subagents,
            "submission_counts": submission_counts,
            "step": step,
        }
        self.save_intermediate_state(async_position, state)


GENERATION_TASK_CLASS = SwarmAgentTask


@hydra.main(version_base=None, config_name="base_swarm_agent_config")
def swarm_agent_generation(cfg: SwarmAgentConfig):
    cfg = SwarmAgentConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)
    task = SwarmAgentTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(SwarmAgentConfig, server_params=server_params())

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        swarm_agent_generation()
