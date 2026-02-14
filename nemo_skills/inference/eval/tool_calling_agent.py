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


@nested_dataclass(kw_only=True)
class ToolCallingAgentConfig(GenerationTaskConfig):
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: dict = field(default_factory=dict)
    prompt_config: str = "eval/ioi/agent/tool_calling"
    max_steps: int = 15
    max_time: str | None = None  # Format: "hh:mm:ss"
    explicit_feedback: bool = False
    max_limit_in_test_output: int = 1000


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_tool_calling_agent_config", node=ToolCallingAgentConfig)


class ToolCallingAgentTask(GenerationTask):
    def __init__(self, cfg: ToolCallingAgentConfig):
        super().__init__(cfg)
        self.agent_prompt = get_prompt(cfg.prompt_config)

    def setup_prompt(self):
        return None

    def log_example_prompt(self, data):
        return

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

    async def _agent_turn(self, messages: list[dict]) -> dict:
        try:
            out = await self.generate_with_semaphore(
                prompt=messages, tools=[SUBMIT_TOOL], include_response=True, **asdict(self.cfg.inference)
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

        reasoning_content = getattr(message, "reasoning_content", None) or ""

        return {
            "message": message,
            "tool_calls": tool_calls,
            "tool_call_ids": tool_call_ids,
            "reasoning_content": reasoning_content,
            "num_generated_tokens": out.get("num_generated_tokens", 0),
        }

    async def _execute_submit(self, code: str, sample: bool, data_point: dict) -> tuple[str, bool]:
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
            result_dict = {**eval_result, "test_case_results": self._filter_test_outputs(test_case_results)}
        else:
            if is_ioi and "subtask_score" in data_point:
                max_score = data_point["subtask_score"]
                subtask_scores = {k: f"{v['score']}/{max_score}" for k, v in normalized.items()}
            else:
                subtask_scores = {k: v["score"] for k, v in normalized.items()}
            result_dict = {"subtask_scores": subtask_scores}

        if is_ioi and "subtask_score" in data_point:
            max_score = float(data_point["subtask_score"])
            success = bool(normalized) and all(float(v["score"]) == max_score for v in normalized.values())
        else:
            success = bool(normalized) and all(float(v["score"]) == 1.0 for v in normalized.values())

        result_dict["success"] = success
        return json.dumps(result_dict), success and not sample

    def dp_print(self, data_point, *args):
        dp_id = data_point.get("id", "?") if isinstance(data_point, dict) else "?"
        print(f"[{dp_id}]", *args)

    async def process_single_datapoint(self, data_point, all_data):
        if self.evaluator is None:
            raise ValueError(
                "ToolCallingAgent requires an evaluator supporting eval_single (set ++eval_type and ++eval_config)."
            )

        if data_point.get("subtask_score") is None:
            data_point["subtask_score"] = "1"

        max_time_seconds = self._parse_max_time(self.cfg.max_time)
        start_time = time.time()

        messages = self.agent_prompt.fill(
            {"subtask_score": data_point.get("subtask_score", "1"), "question": data_point["question"]}
        )
        trace = []
        num_tokens = []
        final_code = ""
        last_code = None

        for step in range(self.cfg.max_steps):
            if max_time_seconds is not None and (time.time() - start_time) >= max_time_seconds:
                self.dp_print(data_point, f"max_time reached at step {step}")
                break

            self.dp_print(data_point, f"step {step + 1}/{self.cfg.max_steps}")

            result = await self._agent_turn(messages)
            if result["message"] is None:
                self.dp_print(data_point, "context window exceeded")
                break

            num_tokens.append(result.get("num_generated_tokens", 0))

            reasoning = result.get("reasoning_content", "")
            if reasoning:
                preview = reasoning[:200] + "..." if len(reasoning) > 200 else reasoning
                self.dp_print(data_point, f"reasoning ({len(reasoning)} chars): {preview}")

            msg = result["message"]
            if hasattr(msg, "model_dump"):
                msg = msg.model_dump()
            messages.append(msg)
            trace.append({"source": "agent", **msg})

            tool_calls = result.get("tool_calls", [])
            tool_call_ids = result.get("tool_call_ids", [])

            if not tool_calls:
                self.dp_print(data_point, "no tool calls, ending")
                break

            should_stop = False
            for tc, tc_id in zip(tool_calls, tool_call_ids):
                name, raw_args = next(iter(tc.items()))
                args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args) if raw_args else {}

                if name == "submit_solution":
                    code = args.get("code", "")
                    sample = bool(args.get("sample", False))
                    self.dp_print(data_point, f"submit_solution(sample={sample}, len={len(code)})")

                    if code:
                        last_code = code
                    tool_out, accepted = await self._execute_submit(code, sample, data_point)
                    if accepted:
                        final_code = code
                        should_stop = True
                        self.dp_print(data_point, "solution accepted")
                else:
                    tool_out = json.dumps({"error": f"Unknown tool: {name}"})

                tool_msg = {"role": "tool", "content": tool_out, "tool_call_id": tc_id}
                messages.append(tool_msg)
                trace.append({"source": "tool", **tool_msg})

            if should_stop:
                break

        if not final_code and last_code:
            final_code = last_code
            self.dp_print(data_point, "using last submitted solution")

        return {
            "id": data_point["id"],
            "generation": f"```cpp\n{final_code}\n```",
            "messages": trace,
            "num_generated_tokens": sum(num_tokens),
            "num_generated_tokens_list": num_tokens,
        }


GENERATION_TASK_CLASS = ToolCallingAgentTask


@hydra.main(version_base=None, config_name="base_tool_calling_agent_config")
def tool_calling_agent_generation(cfg: ToolCallingAgentConfig):
    cfg = ToolCallingAgentConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)
    task = ToolCallingAgentTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(ToolCallingAgentConfig, server_params=server_params())

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        tool_calling_agent_generation()
