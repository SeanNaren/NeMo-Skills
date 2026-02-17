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
from nemo_skills.inference.eval.agent_utils import parse_max_time, process_submission_result
from nemo_skills.inference.eval.bfcl import ClientMessageParser, ServerMessageParser
from nemo_skills.inference.generate import GenerationTask, GenerationTaskConfig, InferenceConfig
from nemo_skills.inference.model import get_model, server_params
from nemo_skills.inference.model.utils import is_context_window_exceeded_error
from nemo_skills.prompt.utils import get_prompt, get_token_count
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class ReasoningAgentConfig(GenerationTaskConfig):
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # agent (orchestrator)
    inference_reasoner: InferenceConfig = field(default_factory=InferenceConfig)  # reasoner
    server: dict = field(default_factory=dict)
    use_client_parsing: bool = False
    model_name: str | None = None
    max_steps: int = 10
    max_time: str | None = None  # Format: "hh:mm:ss" (e.g., "03:45:00")
    explicit_feedback: bool = False
    max_limit_in_test_output: int = 1000
    avg_score: bool = True
    agent_prompt_config: str = "eval/ioi/agent/orchestrator"
    summary_prompt_config: str = "eval/ioi/agent/summary"
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
        self.summary_prompt = get_prompt(cfg.summary_prompt_config)
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
        """Extract C++ code from content, including from final output channel."""
        if not text:
            return None

        # Extract from final output channel
        text_final = text.split("<|end|><|start|>assistant<|channel|>final<|message|>")[-1]
        m = re.findall(r"```(?:cpp|c\+\+)\s*(.*?)```", text_final, re.DOTALL | re.IGNORECASE)
        if m:
            return m[-1].strip()

        return None

    def _build_tools(self):
        """Build tools available to the orchestrator agent."""
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
                                "description": "Strict instructions for the reasoner about what approach/algorithm to use. These override the reasoner's default behavior.",
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
        self,
        problem: str,
        previous_solution: str | None,
        feedback: str | None,
        instructions: str | None,
        data_point: dict,
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

            # Inject agent instructions into the system message so the reasoner
            # treats them as high-priority directives (not optional feedback).
            if instructions and messages and messages[0].get("role") == "system":
                messages[0]["content"] += (
                    "\n\n### STRICT INSTRUCTIONS ###\n"
                    "You MUST follow these instructions when creating your solution. They take priority over your default approach.\n\n"
                    f"{instructions}"
                )
                self.dp_print(data_point, f"Injected instructions into reasoner prompt: {instructions}")

            reasoner_params = asdict(self.cfg.inference_reasoner)
            if reasoner_params.get("tokens_to_generate") is None:
                reasoner_params["tokens_to_generate"] = self.cfg.inference.tokens_to_generate
            result = await self.reasoner_llm.generate_async(prompt=messages, include_response=False, **reasoner_params)

            return {
                "generation": result.get("generation", ""),
                "reasoning_content": result.get("reasoning_content", ""),
                "num_generated_tokens": result.get("num_generated_tokens", 0),
            }

    async def _execute_generate_solution(
        self, problem: str, previous_solution: str | None, args: dict, data_point: dict, tool_call_id
    ) -> dict:
        """Execute a generate_solution tool call. Returns results without mutating shared state."""
        feedback = args.get("feedback")
        instructions = args.get("instructions")

        self.dp_print(
            data_point,
            f"tool: generate_solution(feedback={'yes' if feedback else 'no'}, "
            f"instructions={'yes' if instructions else 'no'})",
        )

        result = await self._call_reasoner(problem, previous_solution, feedback, instructions, data_point)

        code = self._extract_cpp(result.get("generation", ""))
        reasoner_tokens = [result.get("num_generated_tokens", 0)]
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
            "reasoner_tokens": reasoner_tokens,
            "first_valid_code": code,
        }

    async def _execute_submit_solution(self, args: dict, data_point: dict, tool_call_id) -> dict:
        """Execute a submit_solution tool call. Returns results without mutating shared state."""
        code = args.get("code", "")
        sample = bool(args.get("sample", False))

        if not code:
            tool_out = json.dumps({"error": "No code provided"})
            self.dp_print(data_point, "tool: submit_solution — no code provided")
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
        """Generate a concise summary of attempts when context is exceeded."""
        # Get summary prompt messages
        summary_messages_template = self.summary_prompt.fill({})

        # Replace system message and append user message for summary request
        summary_messages = [summary_messages_template[0]] + agent_messages + [summary_messages_template[1]]

        try:
            result = await self.llm.generate_async(
                prompt=summary_messages, include_response=False, **asdict(self.cfg.inference)
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
        max_time_seconds = parse_max_time(self.cfg.max_time)
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
            best_code = saved_state.get("best_code")
            best_score = saved_state.get("best_score", 0.0)
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
            best_code = None
            best_score = 0.0
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
                            "best_code": best_code,
                            "best_score": best_score,
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

            # Build coroutines for all tool calls to execute in parallel
            coros = []
            for gen, tool_call_id in zip(tool_calls, tool_call_ids or [None] * len(tool_calls)):
                (name, raw_args) = next(iter(gen.items()))

                args = raw_args
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except Exception:
                        args = {}

                if name == "generate_solution":
                    coros.append(
                        self._execute_generate_solution(problem, previous_solution, args, data_point, tool_call_id)
                    )
                elif name == "submit_solution":
                    coros.append(self._execute_submit_solution(args, data_point, tool_call_id))
                else:

                    async def _unknown_tool(n=name, tid=tool_call_id):
                        tool_out = json.dumps({"error": f"Unknown tool: {n}"})
                        return {
                            "name": n,
                            "tool_call_id": tid,
                            "tool_out": tool_out,
                            "trace_entries": [],
                            "target_score": 0.0,
                        }

                    coros.append(_unknown_tool())

            # Execute all tool calls in parallel
            tool_results = await asyncio.gather(*coros)

            # Apply results in original order
            for result in tool_results:
                # Append reasoner trace entries (before the tool response)
                trace.extend(result.get("trace_entries", []))

                # Append tool response to conversation and trace
                tool_msg = {"role": "tool", "content": result["tool_out"], "tool_call_id": result["tool_call_id"]}
                agent_messages.append(tool_msg)
                trace.append({"source": "tool", **tool_msg})

                # Apply state from generate_solution
                if result["name"] == "generate_solution":
                    num_reasoner_tokens.extend(result.get("reasoner_tokens", []))
                    if result.get("first_valid_code"):
                        previous_solution = result["first_valid_code"]

                # Apply state from submit_solution
                elif result["name"] == "submit_solution":
                    target_score = result.get("target_score", 0.0)
                    if target_score > best_score:
                        best_score = target_score
                        best_code = (
                            result.get("trace_entries", [{}])[0].get("code") if result.get("trace_entries") else None
                        )
                    if result.get("final_code"):
                        final_code = result["final_code"]

            # Save intermediate state after each iteration
            if async_position is not None:
                state = {
                    "agent_messages": agent_messages,
                    "trace": trace,
                    "num_agent_tokens": num_agent_tokens,
                    "num_reasoner_tokens": num_reasoner_tokens,
                    "previous_solution": previous_solution,
                    "final_code": final_code,
                    "best_code": best_code,
                    "best_score": best_score,
                    "out_of_context": out_of_context,
                    "iteration": iteration + 1,  # Save next iteration to start from
                }
                self.save_intermediate_state(async_position, state)

            # Check if we have a final solution
            if final_code:
                break

        # Use best or last generated solution if available
        if not final_code:
            if best_code:
                final_code = best_code
                self.dp_print(data_point, "using best scoring solution")
            elif previous_solution:
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
