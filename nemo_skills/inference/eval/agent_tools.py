import asyncio
import json
import logging
import re
import sys
from dataclasses import field

import hydra
from omegaconf import ListConfig

from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.inference.eval.bfcl import ClientMessageParser, ServerMessageParser
from nemo_skills.inference.eval.bfcl_utils import MAXIMUM_STEP_LIMIT
from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.model import get_model, server_params
from nemo_skills.inference.model.utils import is_context_window_exceeded_error
from nemo_skills.prompt.utils import get_prompt, get_token_count
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class AgentToolsConfig(GenerateSolutionsConfig):
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: dict = field(default_factory=dict)
    inference_solution: InferenceConfig = field(default_factory=InferenceConfig)
    generate_prompt_config: str = "eval/ioi/agent/solver"
    improve_prompt_config: str = "eval/ioi/agent/self_improve_feedback"
    prompt_config: str = "eval/ioi/agent/agent_tools_solver"
    use_client_parsing: bool = True
    model_name: str | None = None
    max_steps: int = 30

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

        if len(base_url) < 2:
            raise ValueError(
                f"AgentTools requires exactly 2 models (via server.base_url list), got {len(base_url)}. "
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

        # Store normalized lists
        original_server = self.server.copy()
        self.server["base_url"] = base_url
        self.server["model"] = model
        self.server["server_type"] = server_type

        # Temporarily set to first server for parent validation
        self.server = {
            "base_url": base_url[0],
            "model": model[0],
            "server_type": server_type[0],
        }
        try:
            super().__post_init__()
        finally:
            # Restore multi-server config
            self.server = original_server
            self.server["base_url"] = base_url
            self.server["model"] = model
            self.server["server_type"] = server_type


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_agent_tools_generation_config", node=AgentToolsConfig)


class AgentToolsGenerationTask(GenerationTask):
    def __init__(self, cfg: AgentToolsConfig):
        # Extract server configuration (normalized lists) before parent init
        self.server_addresses = cfg.server["base_url"]
        self.model_names = cfg.server["model"]
        self.server_types = cfg.server["server_type"]

        # Will be initialized in setup_llm
        self.solution_llm = None
        self.solution_semaphore = None

        super().__init__(cfg)
        self.prompt = get_prompt(cfg.prompt_config, examples_type=cfg.examples_type)
        self.generate_prompt = get_prompt(cfg.generate_prompt_config, examples_type=cfg.examples_type)
        self.improve_prompt = get_prompt(cfg.improve_prompt_config, examples_type=cfg.examples_type)
        self.message_parser = ClientMessageParser(cfg) if cfg.use_client_parsing else ServerMessageParser(cfg)

    def setup_llm(self):
        # Create sandbox like base class
        self.sandbox = get_sandbox(**self.cfg.sandbox) if self.cfg.sandbox else None

        # Validate exactly two servers provided
        zipped = list(zip(self.server_addresses, self.model_names, self.server_types))
        assert len(zipped) == 2, f"Expected exactly 2 servers (agent, solution), got {len(zipped)}"

        # Build agent and solution clients
        clients = []
        for idx, (address, model_name, server_type) in enumerate(zipped):
            if not isinstance(address, str) or not address:
                raise ValueError(f"Invalid base_url for server {idx}: {address}")
            if not address.startswith(("http://", "https://")):
                address = f"http://{address}"
            if not address.endswith("/v1"):
                address = f"{address}/v1"
            server_config = {"server_type": server_type, "model": model_name, "base_url": address}
            model_client = get_model(**server_config)
            clients.append(model_client)

        # Assign primary agent llm and secondary solution llm
        self.llm = clients[0]
        self.solution_llm = clients[1]

        # Initialize separate semaphore for the solution llm
        if (
            getattr(self.cfg, "parallel_thinking", None) is not None
            and getattr(self.cfg.parallel_thinking, "mode", None) is not None
        ):
            divisor = getattr(getattr(self.solution_llm, "cfg", None), "max_concurrent_requests", 1)
            self.solution_semaphore = asyncio.Semaphore(self.cfg.max_concurrent_requests // max(divisor, 1))
        else:
            self.solution_semaphore = asyncio.Semaphore(self.cfg.max_concurrent_requests)

        return self.llm

    def log_example_prompt(self, data):
        return

    async def _generate_single_assistant_turn(self, inference_state_dict):
        messages = inference_state_dict["messages"]
        tools = inference_state_dict["tools"]
        if self.cfg.system_message:
            messages = [{"role": "system", "content": self.cfg.system_message}] + messages
        input_dict = self.message_parser.construct_input_dict(messages, tools)
        return_dict = {}
        if self.cfg.count_prompt_tokens:
            num_input_tokens = get_token_count(
                self.hf_tokenizer, messages=input_dict["prompt"], tools=input_dict.get("tools", None)
            )
            return_dict["num_input_tokens"] = num_input_tokens
        try:
            output = await self.generate_with_semaphore(**input_dict)
        except Exception as error:
            if is_context_window_exceeded_error(error):
                LOG.warning(f"AgentTools generation failed due to running out of context. {error}")
                return_dict.update({"message": None, "generation": ""})
                return return_dict
            else:
                raise error
        print("raw agent output--------------------------------\n", output, "--------------------------------")
        parsed_response = self.message_parser.parse_output_dict(output)
        return_dict.update(parsed_response)
        return return_dict

    def _parse_reasoning_from_message_content(self, model_response_text: str | None):
        if model_response_text is None:
            return None
        if self.cfg.end_reasoning_string in model_response_text:
            return model_response_text.split(self.cfg.end_reasoning_string)[-1].lstrip("\n")
        return ""

    # copied
    def extract_code_block(self, text: str):
        text = text.split("<|end|><|start|>assistant<|channel|>final<|message|>")[-1]
        matches = re.findall(r"```cpp(.*?)```", text, re.DOTALL)
        return matches[-1].strip() if matches else None

    def _normalize_test_case_results(self, test_case_results: dict) -> dict:
        # ICPC-style: flat dict with outputs list
        if (
            isinstance(test_case_results, dict)
            and "outputs" in test_case_results
            and "score" in test_case_results
            and isinstance(test_case_results.get("outputs"), list)
        ):
            return {"overall": {"score": float(test_case_results["score"])}}
        # IOI-style: dict of subtasks
        return {k: {"score": float(v.get("score", 0.0))} for k, v in test_case_results.items()}

    async def _call_solution_llm(self, messages):
        if not self.solution_llm:
            raise RuntimeError("solution LLM not configured")
        async with self.solution_semaphore:
            return await self.solution_llm.generate_async(prompt=messages, **self.cfg.inference_solution.__dict__)

    def _build_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "submit_solution",
                    "description": "Compile and run the given C++ code. Set sample=true to run only sample tests.",
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
            {
                "type": "function",
                "function": {
                    "name": "generate_solution",
                    "description": "Use a separate model to draft a new C++17 solution as a code block. The question is provided automatically to the function.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "note": {
                                "type": "string",
                                "description": "High-level additional notes to provide to the generation model when creating the solution.",
                            }
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "improve_solution",
                    "description": "Improve an existing C++17 solution using feedback. The question is provided automatically to the generation model when improving the solution.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prev_code": {"type": "string", "description": "Previous c++ solution attempt to improve"},
                            "feedback": {
                                "type": "string",
                                "description": "points of improvements to take into consideration when making a new solution",
                            },
                        },
                        "required": ["prev_code"],
                    },
                },
            },
        ]

    async def process_single_datapoint(self, data_point, all_data):
        # ICPC does not have a subtask score, we add it manually (max score is 1)
        if data_point.get("subtask_score") is None:
            data_point["subtask_score"] = "1"
        messages = self.fill_prompt(data_point, all_data)
        tools = self._build_tools()
        state_dict = {"messages": messages, "tools": tools}

        num_generated_tokens_list = []
        num_input_tokens_list = []
        out_of_context = False
        final_solution = ""
        step_count = 0

        while True:
            model_response = await self._generate_single_assistant_turn(state_dict)
            print(f"model_response: {model_response}")
            if model_response["message"] is None:
                out_of_context = True
                print("Quitting generation due to running out of context.")
                break

            num_generated_tokens_list.append(model_response.get("num_generated_tokens", 0))
            if self.cfg.count_prompt_tokens:
                num_input_tokens_list.append(model_response.get("num_input_tokens", 0))

            if self.cfg.parse_reasoning:
                trimmed = self._parse_reasoning_from_message_content(
                    self.message_parser.get_response_text(model_response["message"])
                )
                self.message_parser.set_response_text(model_response["message"], trimmed)

            state_dict["messages"].append(model_response["message"])
            final_solution = self.message_parser.get_response_text(model_response["message"]) or final_solution

            tool_calls = model_response.get("generation", [])
            tool_call_ids = model_response.get("tool_call_ids", [])
            if not isinstance(tool_calls, list) or len(tool_calls) == 0:
                break

            print(f"tool_calls: {tool_calls}")

            execution_results = []
            should_terminate = False
            for gen in tool_calls:
                try:
                    (name, raw_args) = next(iter(gen.items()))
                    args = raw_args
                    if isinstance(raw_args, str):
                        try:
                            args = json.loads(raw_args)
                        except Exception:
                            print(f"invalid arguments {raw_args}")
                            execution_results.append(json.dumps({"error": "invalid arguments"}))
                            continue
                    if name == "submit_solution":
                        code = args["code"]
                        sample = bool(args["sample"])
                        eval_payload = {**data_point, "generation": code, "only_sample_tests": sample}
                        eval_result = await self.evaluator.eval_single(eval_payload)
                        test_case_results = eval_result.get("test_case_results", {})
                        normalized = self._normalize_test_case_results(test_case_results)
                        subtask_scores = {k: v["score"] for k, v in normalized.items()}
                        success = bool(normalized) and all(float(s) == 1.0 for s in subtask_scores.values())
                        execution_results.append(json.dumps({"subtask_scores": subtask_scores, "success": success}))
                        if success:
                            final_solution = code
                            should_terminate = True
                    elif name in ("generate_solution", "improve_solution"):
                        if name == "generate_solution":
                            # Fill solver prompt
                            msgs = self.generate_prompt.fill(
                                {"subtask_score": data_point["subtask_score"], "question": data_point["question"]}
                            )
                            note = args["note"]
                            msgs.append({"role": "user", "content": note})
                        else:
                            # Fill self-improve prompt
                            prev_code = args["prev_code"]
                            msgs = self.improve_prompt.fill(
                                {
                                    "subtask_score": data_point["subtask_score"],
                                    "question": data_point["question"],
                                    "solution": prev_code,
                                    "feedback": args["feedback"],
                                }
                            )

                        # todo: currently we do not keep previous messages.
                        sol_out = await self._call_solution_llm(msgs)
                        raw = sol_out.get("generation", "")
                        code = self.extract_code_block(raw)
                        if not code:
                            code = "failed to create new solution, suggest retrying again"
                        execution_results.append(json.dumps({"code": code}))
                    else:
                        print(f"unknown tool {name}")
                        execution_results.append(json.dumps({"error": f"unknown tool {name}"}))
                except Exception as e:
                    print(f"error {e}")
                    execution_results.append(json.dumps({"error": str(e)}))

            for execution_result, tool_call_id in zip(execution_results, tool_call_ids):
                state_dict["messages"].append(
                    {"role": "tool", "content": execution_result, "tool_call_id": tool_call_id}
                )

            if should_terminate:
                print(f"[Success] Problem {data_point['id']}: All test cases passed. Stopping early.")
                break

            step_count += 1
            if step_count >= min(int(self.cfg.max_steps), MAXIMUM_STEP_LIMIT):
                print(f"Forced stop after {min(int(self.cfg.max_steps), MAXIMUM_STEP_LIMIT)} steps.")
                break
            print(f"messages: {state_dict['messages']}")

        out = {
            "id": data_point["id"],
            "generation": final_solution,
            "messages": state_dict["messages"],
            "num_generated_tokens": sum(num_generated_tokens_list),
            "num_generated_tokens_list": num_generated_tokens_list,
        }
        print("Exited loop, output\n", out)

        if self.cfg.count_prompt_tokens:
            out["num_input_tokens"] = sum(num_input_tokens_list)
            out["num_input_tokens_list"] = num_input_tokens_list
        if out_of_context:
            out["error"] = "_ran_out_of_context_"
        return out

    def wait_for_server(self):
        """Wait for all servers to be ready by calling parent's wait method for each.

        Delegates to base class's wait_for_server() method for each server address.
        This reuses all base class validation and subprocess logic.
        """
        LOG.info(f"Waiting for {len(self.server_addresses)} server(s) to be ready...")

        # Store original multi-server config
        original_server_config = self.cfg.server.copy()

        try:
            for idx, (address, model_name) in enumerate(zip(self.server_addresses, self.model_names)):
                # Ensure base_url has http:// prefix if not already present
                if not address.startswith(("http://", "https://")):
                    address = f"http://{address}"

                LOG.info(f"Waiting for Server {idx} ({model_name}) @ {address}...")
                # Temporarily set server config to single server for parent method
                self.cfg.server = {"base_url": address}
                super().wait_for_server()
                LOG.info(f"✓ Server {idx} ({model_name}) is ready!")
        finally:
            # Always restore original config
            self.cfg.server = original_server_config

    def wait_for_sandbox(self):
        if self.cfg.wait_for_sandbox and self.sandbox:
            self.sandbox.wait_for_sandbox()


GENERATION_TASK_CLASS = AgentToolsGenerationTask


@hydra.main(version_base=None, config_name="base_agent_tools_generation_config")
def agent_tools_generation(cfg: AgentToolsConfig):
    cfg = AgentToolsConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)
    task = AgentToolsGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(AgentToolsConfig, server_params=server_params())

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        agent_tools_generation()
