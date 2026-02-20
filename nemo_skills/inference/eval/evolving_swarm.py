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

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

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

TEST_SOLUTION_TOOL = {
    "type": "function",
    "function": {
        "name": "test_solution",
        "description": "Compile and run C++17 code against SAMPLE test cases only. Use this to iterate on your solution.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "C++17 source code"},
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
SUBAGENT_TOOLS = [TEST_SOLUTION_TOOL]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@nested_dataclass(kw_only=True)
class EvolvingSwarmConfig(GenerationTaskConfig):
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    inference_subagent: InferenceConfig = field(default_factory=InferenceConfig)
    server: dict = field(default_factory=dict)
    max_steps: int = 100
    max_subagent_steps: int = 10
    max_time: str | None = None  # Format: "hh:mm:ss"
    explicit_feedback: bool = False
    max_limit_in_test_output: int = 1000
    avg_score: bool = True
    agent_prompt_config: str = "eval/ioi/agent/evolving_orchestrator"
    summary_prompt_config: str = "eval/ioi/agent/summary"


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_evolving_swarm_config", node=EvolvingSwarmConfig)

# ---------------------------------------------------------------------------
# Approach ledger helpers
# ---------------------------------------------------------------------------


def _format_ledger(approach_ledger: list[dict], best_score: float) -> str:
    """Format the approach ledger as a compact markdown table for injection."""
    lines = [
        f"## Approach Ledger (best score so far: {best_score:.3f})",
        "| # | Agent | Strategy | Sample Score | Full Score | Status |",
        "|---|-------|----------|-------------|------------|--------|",
    ]
    for i, entry in enumerate(approach_ledger, 1):
        sample = f"{entry['best_sample_score']:.2f}" if entry["best_sample_score"] is not None else "-"
        full = f"{entry['best_full_score']:.2f}" if entry["best_full_score"] is not None else "-"
        strategy = entry["strategy"][:60] + ("..." if len(entry["strategy"]) > 60 else "")
        lines.append(f"| {i} | {entry['agent']} | {strategy} | {sample} | {full} | {entry['status']} |")
    lines.append("")
    lines.append(
        "Strategies marked 'exhausted' did not improve after full evaluation. Try DIFFERENT algorithm families."
    )
    return "\n".join(lines)


def _update_ledger_on_submit(approach_ledger: list[dict], agent_name: str | None, score: float):
    """Update the ledger entry for agent_name with a full-submission score."""
    if not agent_name:
        return
    for entry in approach_ledger:
        if entry["agent"] == agent_name:
            if entry["best_full_score"] is None or score > entry["best_full_score"]:
                entry["best_full_score"] = score
                if score > 0:
                    entry["status"] = "promising"
            else:
                # Score didn't improve — mark exhausted
                entry["status"] = "exhausted"
            return


# ---------------------------------------------------------------------------
# Task class
# ---------------------------------------------------------------------------


class EvolvingSwarmTask(GenerationTask):
    def __init__(self, cfg: EvolvingSwarmConfig):
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

    def _get_orchestrator_inference_params(self):
        params = asdict(self.cfg.inference)
        extra_body = dict(params.get("extra_body", {}) or {})
        params["extra_body"] = extra_body
        return params

    def _get_subagent_inference_params(self):
        params = asdict(self.cfg.inference_subagent)
        extra_body = dict(params.get("extra_body", {}) or {})
        extra_body["chat_template_kwargs"] = {"thinking": True}
        params["extra_body"] = extra_body
        params["tokens_to_generate"] = None
        return params

    async def _orchestrator_turn(self, messages: list[dict]) -> dict:
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
    ) -> tuple[str, bool, float]:
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

        result = process_submission_result(
            eval_result,
            data_point,
            explicit_feedback=self.cfg.explicit_feedback,
            avg_score=self.cfg.avg_score,
            max_limit_in_test_output=self.cfg.max_limit_in_test_output,
        )
        return result["tool_output"], result["success"] and not sample, result["target_score"]

    def _build_subagent_system_prompt(self, user_system_prompt: str) -> str:
        tool_instructions = (
            "\n\n# Tools\n"
            "You have access to a `test_solution` tool that compiles and runs C++17 code against SAMPLE test cases only.\n"
            "- Call test_solution(code) to test on sample inputs.\n"
            "- If samples fail, analyze the feedback, fix your code, and retest.\n"
            "- You MUST use test_solution to test your code. Do NOT just output code in text.\n"
            "- Iterate until samples pass or you run out of steps.\n"
            "- Your code will be returned to the orchestrator for full evaluation."
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
        full_system_prompt = self._build_subagent_system_prompt(system_prompt)
        messages = [
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": task_prompt},
        ]
        trace = []
        num_tokens = []
        last_code = None
        best_code = None
        best_sample_score = 0.0

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
            msg = sanitize_message(msg)
            messages.append(msg)
            trace.append({"source": "subagent", **msg})

            tool_calls = result.get("tool_calls", [])
            tool_call_ids = result.get("tool_call_ids", [])

            if not tool_calls:
                code = extract_cpp(msg.get("content", ""))
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

                if name == "test_solution":
                    code = args.get("code", "")
                    if code:
                        last_code = code
                    self.dp_print(
                        data_point,
                        f"  subagent '{agent_name}' step {step + 1}: test_solution(sample=True)",
                    )
                    tool_out, _accepted, target_score = await self._execute_submit(
                        code, True, data_point, submission_counts
                    )
                    if code and target_score > best_sample_score:
                        best_sample_score = target_score
                        best_code = code
                else:
                    tool_out = json.dumps({"error": f"Unknown tool: {name}"})

                tool_msg = {"role": "tool", "content": tool_out, "tool_call_id": tc_id}
                messages.append(tool_msg)
                trace.append({"source": "tool", **tool_msg})

        return {
            "best_code": best_code,
            "best_sample_score": best_sample_score,
            "last_code": last_code,
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
                "EvolvingSwarm requires an evaluator supporting eval_single (set ++eval_type and ++eval_config)."
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
            agent_messages = saved_state["agent_messages"]
            trace = saved_state["trace"]
            num_orchestrator_tokens = saved_state["num_orchestrator_tokens"]
            num_subagent_tokens = saved_state["num_subagent_tokens"]
            final_code = saved_state["final_code"]
            last_code = saved_state["last_code"]
            best_code = saved_state.get("best_code")
            best_score = saved_state.get("best_score", 0.0)
            subagents = saved_state["subagents"]
            submission_counts = saved_state["submission_counts"]
            approach_ledger = saved_state.get("approach_ledger", [])
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
            best_code = None
            best_score = 0.0
            subagents = {}
            submission_counts = {"sample": 0, "full": 0}
            approach_ledger = []
            start_step = 0

        # Track which agent produced the last subagent-returned code (for ledger updates)
        last_returning_agent = None

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
                        best_code,
                        best_score,
                        subagents,
                        submission_counts,
                        approach_ledger,
                    )
                break

            self.dp_print(data_point, f"step {step + 1}/{self.cfg.max_steps}")

            result = await self._orchestrator_turn(agent_messages)
            if result["message"] is None:
                self.dp_print(data_point, "context window exceeded, generating summary and restarting")

                summary = await self._summarize_progress(agent_messages)
                self.dp_print(data_point, f"summary: {summary[:100]}...")

                # Preserve ledger across context resets
                ledger_text = _format_ledger(approach_ledger, best_score) if approach_ledger else ""
                reset_content = f"Problem:\n{problem}\n\nPrevious attempt summary:\n{summary}"
                if ledger_text:
                    reset_content += f"\n\n{ledger_text}"

                agent_messages = [
                    {"role": "system", "content": self.agent_prompt.config.system},
                    {"role": "user", "content": reset_content},
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
            msg = sanitize_message(msg)
            agent_messages.append(msg)
            trace.append({"source": "orchestrator", **msg})

            tool_calls = result.get("tool_calls", [])
            tool_call_ids = result.get("tool_call_ids", [])

            if not tool_calls:
                code = extract_cpp(msg.get("content", ""))
                if code:
                    last_code = code

                if not final_code:
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

            # Process create_subagent calls first
            for tc_id, name, args in sync_calls:
                if name == "create_subagent":
                    agent_name = args.get("name", "")
                    system_prompt = args.get("system_prompt", "")
                    subagents[agent_name] = system_prompt
                    tool_out = json.dumps({"status": "success", "agent": agent_name})
                    self.dp_print(data_point, f"created subagent: {agent_name}")

                    # Add to approach ledger
                    approach_ledger.append(
                        {
                            "agent": agent_name,
                            "strategy": system_prompt[:200],
                            "best_sample_score": None,
                            "best_full_score": None,
                            "status": "active",
                        }
                    )
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
                            tool_out_str, accepted, target_score = await self._execute_submit(c, s, dp, sc)
                            return {
                                "tool_out": tool_out_str,
                                "accepted": accepted,
                                "code": c,
                                "target_score": target_score,
                                "sample": s,
                            }

                        coros.append(_do_submit())
                        coro_ids.append(tc_id)
                        coro_names.append(("submit_solution", last_returning_agent))

                if coros:
                    results = await asyncio.gather(*coros)

                    for (coro_name, agent_name), tc_id, res in zip(coro_names, coro_ids, results):
                        if coro_name == "assign_task":
                            subagent_result = res
                            trace.extend(subagent_result.get("trace", []))
                            num_subagent_tokens.extend(subagent_result.get("num_tokens", []))

                            sa_best_code = subagent_result.get("best_code")
                            sa_last = subagent_result.get("last_code")
                            sa_best_sample_score = subagent_result.get("best_sample_score", 0.0)

                            # Update ledger with sample score
                            for entry in approach_ledger:
                                if entry["agent"] == agent_name:
                                    entry["best_sample_score"] = sa_best_sample_score
                                    break

                            returned_code = sa_best_code or sa_last
                            if returned_code:
                                tool_out = json.dumps(
                                    {
                                        "status": "code_ready",
                                        "solution": returned_code,
                                        "agent": agent_name,
                                        "sample_score": sa_best_sample_score,
                                        "note": "Use submit_solution(code, sample=false) to run full evaluation.",
                                    }
                                )
                                last_code = returned_code
                                last_returning_agent = agent_name
                            else:
                                tool_out = json.dumps(
                                    {
                                        "status": "failed",
                                        "error": "Sub-agent did not produce a solution",
                                        "agent": agent_name,
                                    }
                                )
                                # Mark as exhausted if no code produced
                                for entry in approach_ledger:
                                    if entry["agent"] == agent_name:
                                        entry["status"] = "exhausted"
                                        break

                            self.dp_print(
                                data_point,
                                f"subagent '{agent_name}' returned: "
                                f"code={'yes' if returned_code else 'no'}, "
                                f"sample_score={sa_best_sample_score:.2f}",
                            )

                        elif coro_name == "submit_solution":
                            tool_out = res["tool_out"]
                            if res["accepted"]:
                                final_code = res["code"]
                                self.dp_print(data_point, "solution accepted")
                            if res["code"]:
                                last_code = res["code"]
                            if not res.get("sample") and res["code"]:
                                target_score = res.get("target_score", 0.0)
                                if target_score > best_score:
                                    best_score = target_score
                                    best_code = res["code"]
                                # Update ledger for the agent that produced this code
                                _update_ledger_on_submit(approach_ledger, agent_name, target_score)

                        tool_msg = {"role": "tool", "content": tool_out, "tool_call_id": tc_id}
                        agent_messages.append(tool_msg)
                        trace.append({"source": "tool", **tool_msg})

            # Inject approach ledger after processing tool calls (if any approaches exist)
            if approach_ledger:
                ledger_text = _format_ledger(approach_ledger, best_score)
                ledger_msg = {"role": "user", "content": ledger_text}
                agent_messages.append(ledger_msg)
                trace.append({"source": "ledger", **ledger_msg, "iteration": step + 1})
                self.dp_print(
                    data_point,
                    f"step {step + 1}: injected ledger ({len(approach_ledger)} approaches, "
                    f"{sum(1 for e in approach_ledger if e['status'] == 'exhausted')} exhausted)",
                )

            # Save intermediate state
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
                    best_code,
                    best_score,
                    subagents,
                    submission_counts,
                    approach_ledger,
                )

            if final_code:
                break

        if not final_code:
            if best_code:
                final_code = best_code
                self.dp_print(data_point, "using best scoring solution")
            elif last_code:
                final_code = last_code
                self.dp_print(data_point, "using last available solution")

        self.dp_print(
            data_point,
            f"done: submissions(sample={submission_counts['sample']}, full={submission_counts['full']})",
        )

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
        best_code,
        best_score,
        subagents,
        submission_counts,
        approach_ledger,
    ):
        state = {
            "agent_messages": agent_messages,
            "trace": trace,
            "num_orchestrator_tokens": num_orchestrator_tokens,
            "num_subagent_tokens": num_subagent_tokens,
            "final_code": final_code,
            "last_code": last_code,
            "best_code": best_code,
            "best_score": best_score,
            "subagents": subagents,
            "submission_counts": submission_counts,
            "approach_ledger": approach_ledger,
            "step": step,
        }
        self.save_intermediate_state(async_position, state)


GENERATION_TASK_CLASS = EvolvingSwarmTask


@hydra.main(version_base=None, config_name="base_evolving_swarm_config")
def evolving_swarm_generation(cfg: EvolvingSwarmConfig):
    cfg = EvolvingSwarmConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)
    task = EvolvingSwarmTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(EvolvingSwarmConfig, server_params=server_params())

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        evolving_swarm_generation()
