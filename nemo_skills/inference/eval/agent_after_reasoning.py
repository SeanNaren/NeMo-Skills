import asyncio
import json
import logging
import re
import sys
from dataclasses import asdict, field
from pathlib import Path

import hydra
from omegaconf import ListConfig

from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.inference.eval.bfcl import ClientMessageParser, ServerMessageParser
from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.model import get_model, server_params
from nemo_skills.inference.model.utils import is_context_window_exceeded_error
from nemo_skills.prompt.utils import get_token_count
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class AgentAfterReasoningConfig(GenerateSolutionsConfig):
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # agent
    inference_reasoner: InferenceConfig = field(default_factory=InferenceConfig)
    server: dict = field(default_factory=dict)
    use_client_parsing: bool = False
    model_name: str | None = None
    max_steps: int = 10
    agent_system_message: str = (
        "You are a tool-using coding agent. You will be given a candidate C++17 solution. "
        "Your goal is to iteratively improve the solution by submitting it for testing. "
        "Call submit_solution with the code (and sample=true if you want to run only samples first). "
        "Analyze the results and provide improved versions of the code in subsequent submissions."
    )
    reasoner_system_message: str = (
        "You are a reasoning-focused competitive programming solver. "
        "Return exactly one C++17 solution inside a single ```cpp``` code block."
    )

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
                f"AgentAfterReasoning requires exactly 2 models (agent, reasoner) via server.base_url, got {len(base_url)}. "
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
cs.store(name="base_agent_after_reasoning_config", node=AgentAfterReasoningConfig)


class AgentAfterReasoningGenerationTask(GenerationTask):
    def __init__(self, cfg: AgentAfterReasoningConfig):
        self.server_addresses = cfg.server["base_url"]
        self.model_names = cfg.server["model"]
        self.server_types = cfg.server["server_type"]
        self.reasoner_llm = None
        self.reasoner_semaphore = None
        super().__init__(cfg)
        self.message_parser = ClientMessageParser(cfg) if cfg.use_client_parsing else ServerMessageParser(cfg)

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

    def _build_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "submit_solution",
                    "description": "Compile and run the given C++17 code. Set sample=true to run only sample tests.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "C++17 source code to submit"},
                            "sample": {"type": "boolean", "description": "Run only sample tests", "default": False},
                        },
                        "required": ["code"],
                    },
                },
            }
        ]

    def _extract_cpp(self, text: str | None) -> str | None:
        if not text:
            return None
        text = text.split("<|end|><|start|>assistant<|channel|>final<|message|>")[-1]
        m = re.findall(r"```(?:cpp|c\+\+)\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        return m[-1].strip() if m else None

    def _normalize_scores(self, test_case_results: dict) -> dict:
        if (
            isinstance(test_case_results, dict)
            and "outputs" in test_case_results
            and "score" in test_case_results
            and isinstance(test_case_results.get("outputs"), list)
        ):
            return {"overall": {"score": float(test_case_results["score"])}}
        return {k: {"score": float(v.get("score", 0.0))} for k, v in test_case_results.items()}

    async def _agent_turn(self, messages: list[dict], tools: list[dict]) -> dict:
        if self.cfg.system_message:
            messages = [{"role": "system", "content": self.cfg.system_message}] + messages
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

    async def _reasoner_turn(self, messages: list[dict]) -> dict:
        async with self.reasoner_semaphore:
            return await self.reasoner_llm.generate_async(
                prompt=messages, include_response=False, **asdict(self.cfg.inference_reasoner)
            )

    async def process_single_datapoint(self, data_point, all_data):
        if self.evaluator is None:
            raise ValueError(
                "AgentAfterReasoning requires an evaluator supporting eval_single (set ++eval_type and ++eval_config)."
            )

        if data_point.get("subtask_score") is None:
            data_point["subtask_score"] = "1"

        problem = data_point.get("question") or data_point.get("problem") or ""
        self.dp_print(data_point, "start")

        # Step 1: Run reasoner once to get initial solution
        reasoner_messages = [
            {"role": "system", "content": self.cfg.reasoner_system_message},
            {"role": "user", "content": problem},
        ]
        trace = [{"source": "reasoner", **reasoner_messages[0]}, {"source": "reasoner", **reasoner_messages[1]}]

        self.dp_print(data_point, "running reasoner for initial solution")
        r = await self._reasoner_turn(reasoner_messages)
        r_msg = {
            "role": "assistant",
            "content": r.get("generation", ""),
            "reasoning_content": r.get("reasoning_content", ""),
        }
        self.dp_print(
            data_point,
            f"reasoner_tokens={r.get('num_generated_tokens', 0)} content_len={len(r_msg['content'])}",
        )
        trace.append({"source": "reasoner", **r_msg})

        # Extract initial code (without reasoning trace)
        initial_code = self._extract_cpp(r_msg["content"])
        if not initial_code:
            self.dp_print(data_point, "reasoner: no cpp block found")
            return {
                "id": data_point["id"],
                "generation": "",
                "messages": trace,
                "num_generated_tokens": r.get("num_generated_tokens", 0),
                "error": "no_code_generated",
            }

        # Step 2: Agent iterates on the solution
        tools = self._build_tools()
        agent_messages = [
            {"role": "system", "content": self.cfg.agent_system_message},
            {"role": "user", "content": f"Here is a candidate solution to improve:\n```cpp\n{initial_code}\n```"},
        ]
        trace.append({"source": "agent", **agent_messages[0]})
        trace.append({"source": "agent", **agent_messages[1]})

        num_agent_tokens = [r.get("num_generated_tokens", 0)]
        final_code = initial_code
        out_of_context = False

        for step in range(max(1, int(self.cfg.max_steps))):
            self.dp_print(data_point, f"step {step + 1}/{max(1, int(self.cfg.max_steps))}: agent")

            # Agent makes a turn
            a = await self._agent_turn(agent_messages, tools)
            if a.get("message") is None:
                self.dp_print(data_point, "agent: out_of_context")
                out_of_context = True
                break

            num_agent_tokens.append(a.get("num_generated_tokens", 0))
            msg = a["message"]
            if hasattr(msg, "model_dump"):
                msg = msg.model_dump()

            # Add assistant message to conversation
            agent_messages.append(msg)
            trace.append({"source": "agent", **msg})

            tool_calls = a.get("generation", [])
            tool_call_ids = a.get("tool_call_ids", [])

            if not isinstance(tool_calls, list) or len(tool_calls) == 0:
                self.dp_print(data_point, "agent: no tool call, ending iteration")
                break

            self.dp_print(data_point, f"agent_tool_calls={len(tool_calls)}")

            should_stop = False
            for gen, tool_call_id in zip(tool_calls, tool_call_ids or [None] * len(tool_calls)):
                (name, raw_args) = next(iter(gen.items()))
                if name != "submit_solution":
                    tool_out = json.dumps({"error": f"unknown tool {name}"})
                    tool_msg = {"role": "tool", "content": tool_out, "tool_call_id": tool_call_id}
                    agent_messages.append(tool_msg)
                    trace.append({"source": "tool", **tool_msg})
                    self.dp_print(data_point, f"tool: {tool_out}")
                    continue

                args = raw_args
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except Exception:
                        args = {"code": raw_args}

                submitted = args.get("code") or final_code
                sample = bool(args.get("sample", False))
                self.dp_print(data_point, f"submit_solution(sample={sample}) code_len={len(submitted)}")

                eval_payload = {**data_point, "generation": f"```cpp\n{submitted}\n```", "only_sample_tests": sample}
                eval_result = await self.evaluator.eval_single(eval_payload)
                test_case_results = eval_result.get("test_case_results", {})
                normalized = self._normalize_scores(test_case_results)
                subtask_scores = {k: v["score"] for k, v in normalized.items()}
                success = bool(normalized) and all(float(s) == 1.0 for s in subtask_scores.values())
                tool_out = json.dumps({"subtask_scores": subtask_scores, "success": success})

                tool_msg = {"role": "tool", "content": tool_out, "tool_call_id": tool_call_id}
                agent_messages.append(tool_msg)
                trace.append({"source": "tool", **tool_msg})
                self.dp_print(data_point, f"result: {tool_out}")

                if success:
                    final_code = submitted
                    should_stop = True

            if should_stop:
                self.dp_print(data_point, "success")
                break

        out = {
            "id": data_point["id"],
            "generation": final_code,
            "messages": trace,
            "num_generated_tokens": sum(num_agent_tokens),
            "num_generated_tokens_list": {"agent": num_agent_tokens},
        }
        if out_of_context:
            out["error"] = "_ran_out_of_context_"
            self.dp_print(data_point, "stopped: out_of_context")
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


GENERATION_TASK_CLASS = AgentAfterReasoningGenerationTask


@hydra.main(version_base=None, config_name="base_agent_after_reasoning_config")
def agent_after_reasoning_generation(cfg: AgentAfterReasoningConfig):
    cfg = AgentAfterReasoningConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)
    task = AgentAfterReasoningGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(AgentAfterReasoningConfig, server_params=server_params())

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        agent_after_reasoning_generation()
