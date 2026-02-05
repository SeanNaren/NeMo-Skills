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
class OrchestratorAgentConfig(GenerateSolutionsConfig):
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # orchestrator
    inference_reasoner: InferenceConfig = field(default_factory=InferenceConfig)  # reasoning agents
    server: dict = field(default_factory=dict)
    use_client_parsing: bool = False
    model_name: str | None = None
    max_steps: int = 5
    max_agents: int = 5
    orchestrator_system_message: str = (
        "You are an orchestrator that coordinates multiple reasoning agents to solve competitive programming problems.\n\n"
        "Your task is to:\n"
        "1. Analyze the problem and decompose it into different solution approaches\n"
        "2. Spawn specialized agents (up to {max_agents}) with specific roles and instructions\n"
        "3. Review solutions from agents and decide which to submit for evaluation\n"
        "4. Use the submit_solution tool to test promising solutions (PENALTY APPLIES - use wisely!)\n\n"
        "Agent Spawning Format:\n"
        "Output a JSON object with an 'agents' list:\n"
        "```json\n"
        "{{\n"
        '  "agents": [\n'
        '    {{"role": "greedy_solver", "instruction": "Solve using greedy approach..."}},\n'
        '    {{"role": "dp_solver", "instruction": "Solve using dynamic programming..."}}\n'
        "  ]\n"
        "}}\n"
        "```\n\n"
        "Submission Tool:\n"
        "Use submit_solution(code, sample=true/false) to test solutions.\n"
        "- Set sample=true for sample tests (safer, no full penalty)\n"
        "- Set sample=false for full evaluation (PENALTY APPLIES)\n\n"
        "You have {max_steps} steps total to produce a working solution."
    )
    reasoner_system_message: str = (
        "You are a specialized competitive programming solver.\n"
        "Follow the instructions provided and return exactly one C++17 solution inside a single ```cpp``` code block."
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
                f"OrchestratorAgent requires exactly 2 models (orchestrator, reasoner) via server.base_url, got {len(base_url)}. "
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

        # Format system message with config values
        self.orchestrator_system_message = self.orchestrator_system_message.format(
            max_agents=self.max_agents, max_steps=self.max_steps
        )


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_orchestrator_agent_config", node=OrchestratorAgentConfig)


class OrchestratorAgentGenerationTask(GenerationTask):
    def __init__(self, cfg: OrchestratorAgentConfig):
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

    def _extract_agents_json(self, text: str) -> list[dict] | None:
        """Extract agents JSON from orchestrator output.

        Expected format:
        ```json
        {
          "agents": [
            {"role": "...", "instruction": "..."},
            ...
          ]
        }
        ```
        """
        if not text:
            return None

        # Extract from final output channel if present
        text_final = text.split("<|end|><|start|>assistant<|channel|>final<|message|>")[-1]

        # Try to find JSON block
        json_match = re.search(r"```json\s*(.*?)```", text_final, re.DOTALL | re.IGNORECASE)
        if json_match:
            try:
                data = json.loads(json_match.group(1).strip())
                agents = data.get("agents", [])
                if isinstance(agents, list) and len(agents) > 0:
                    # Validate each agent has required fields
                    for agent in agents:
                        if not isinstance(agent, dict) or "role" not in agent or "instruction" not in agent:
                            return None
                    return agents
            except json.JSONDecodeError:
                pass

        return None

    def _extract_cpp(self, text: str | None) -> str | None:
        """Extract C++ code from orchestrator final output."""
        if not text:
            return None

        # Extract from final output channel
        text_final = text.split("<|end|><|start|>assistant<|channel|>final<|message|>")[-1]
        m = re.findall(r"```(?:cpp|c\+\+)\s*(.*?)```", text_final, re.DOTALL | re.IGNORECASE)
        if m:
            return m[-1].strip()

        return None

    def _build_tools(self):
        """Build tools available to the orchestrator."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "submit_solution",
                    "description": (
                        "Submit a solution for official evaluation. PENALTY APPLIES - use sparingly. "
                        "Set sample=true to run only sample tests first."
                    ),
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

    async def _orchestrator_turn(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Call orchestrator model."""
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

    async def _spawn_agent(self, agent_spec: dict, problem: str, agent_id: int) -> dict:
        """Spawn a single reasoning agent with specific role and instruction."""
        agent_messages = [
            {"role": "system", "content": self.cfg.reasoner_system_message},
            {
                "role": "user",
                "content": f"Problem:\n{problem}\n\nYour Role: {agent_spec['role']}\n\nInstructions:\n{agent_spec['instruction']}",
            },
        ]

        async with self.reasoner_semaphore:
            out = await self.reasoner_llm.generate_async(
                prompt=agent_messages, include_response=False, **asdict(self.cfg.inference_reasoner)
            )

        return {
            "agent_id": agent_id,
            "role": agent_spec["role"],
            "generation": out.get("generation", ""),
            "reasoning_content": out.get("reasoning_content", ""),
            "num_generated_tokens": out.get("num_generated_tokens", 0),
        }

    async def _spawn_agents(self, agents: list[dict], problem: str) -> list[dict]:
        """Spawn multiple agents in parallel and collect their solutions."""
        # Limit to max_agents
        agents = agents[: self.cfg.max_agents]

        # Execute all agents in parallel
        tasks = [self._spawn_agent(agent, problem, idx) for idx, agent in enumerate(agents)]
        results = await asyncio.gather(*tasks)

        return results

    def _format_agent_solutions(self, agent_results: list[dict]) -> str:
        """Format agent solutions for orchestrator review."""
        lines = ["Agent Solutions:"]
        for result in agent_results:
            lines.append(f"\n## Agent {result['agent_id']} ({result['role']}) ##")
            code = self._extract_cpp(result["generation"])
            if code:
                lines.append(f"```cpp\n{code}\n```")
            else:
                lines.append("(No valid C++ solution produced)")

        return "\n".join(lines)

    async def process_single_datapoint(self, data_point, all_data):
        if self.evaluator is None:
            raise ValueError(
                "OrchestratorAgent requires an evaluator supporting eval_single (set ++eval_type and ++eval_config)."
            )

        if data_point.get("subtask_score") is None:
            data_point["subtask_score"] = "1"

        problem = data_point.get("question") or data_point.get("problem") or ""
        self.dp_print(data_point, "start")

        # Initialize orchestrator conversation
        orchestrator_system = {"role": "system", "content": self.cfg.orchestrator_system_message}
        orchestrator_messages = [
            orchestrator_system,
            {"role": "user", "content": f"Problem:\n{problem}"},
        ]

        trace = [
            {"source": "orchestrator", **orchestrator_system},
            {"source": "orchestrator", "role": "user", "content": problem},
        ]

        num_orchestrator_tokens, num_agent_tokens = [], []
        final_code, out_of_context = "", False
        total_agents_spawned = 0
        agent_results = []
        tools = self._build_tools()

        for step in range(1, self.cfg.max_steps + 1):
            remaining_steps = self.cfg.max_steps - step + 1
            self.dp_print(
                data_point, f"step {step}/{self.cfg.max_steps}: orchestrator (remaining_steps={remaining_steps})"
            )

            # Remind orchestrator of remaining steps
            if step > 1:
                orchestrator_messages.append(
                    {
                        "role": "user",
                        "content": f"You have {remaining_steps} steps remaining. Either spawn new agents or provide the final solution in ```cpp``` tags.",
                    }
                )

            # Orchestrator turn
            o = await self._orchestrator_turn(orchestrator_messages, tools)
            if o.get("message") is None:
                self.dp_print(data_point, "orchestrator: out_of_context")
                out_of_context = True
                break

            num_orchestrator_tokens.append(o.get("num_generated_tokens", 0))

            msg = o["message"]
            if hasattr(msg, "model_dump"):
                msg = msg.model_dump()

            self.dp_print(
                data_point,
                f"orchestrator_tokens={o.get('num_generated_tokens', 0)} content={msg.get('content', '')[:100]}",
            )

            orchestrator_messages.append(msg)
            trace.append({"source": "orchestrator", **msg})

            # Handle tool calls (submission)
            tool_calls = o.get("generation", [])
            tool_call_ids = o.get("tool_call_ids", [])
            if isinstance(tool_calls, list) and len(tool_calls) > 0:
                self.dp_print(data_point, f"orchestrator: {len(tool_calls)} tool call(s)")

                for gen, tool_call_id in zip(tool_calls, tool_call_ids or [None] * len(tool_calls)):
                    (name, raw_args) = next(iter(gen.items()))
                    if name != "submit_solution":
                        tool_out = json.dumps({"error": f"unknown tool {name}"})
                        trace.append(
                            {"source": "tool", "role": "tool", "content": tool_out, "tool_call_id": tool_call_id}
                        )
                        orchestrator_messages.append(
                            {"role": "tool", "content": tool_out, "tool_call_id": tool_call_id}
                        )
                        self.dp_print(data_point, f"tool: {tool_out}")
                        continue

                    # Parse arguments
                    args = raw_args
                    if isinstance(raw_args, str):
                        try:
                            args = json.loads(raw_args)
                        except Exception:
                            args = {"code": raw_args}

                    submitted_code = args.get("code", "")
                    sample = bool(args.get("sample", False))

                    if not submitted_code:
                        tool_out = json.dumps({"error": "No code provided to submit_solution"})
                        trace.append(
                            {"source": "tool", "role": "tool", "content": tool_out, "tool_call_id": tool_call_id}
                        )
                        orchestrator_messages.append(
                            {"role": "tool", "content": tool_out, "tool_call_id": tool_call_id}
                        )
                        self.dp_print(data_point, f"tool: {tool_out}")
                        continue

                    # Execute submission
                    self.dp_print(data_point, f"submit_solution(sample={sample}) code_len={len(submitted_code)}")
                    eval_payload = {
                        **data_point,
                        "generation": f"```cpp\n{submitted_code}\n```",
                        "only_sample_tests": sample,
                    }
                    eval_result = await self.evaluator.eval_single(eval_payload)
                    test_case_results = eval_result.get("test_case_results", {})

                    # Build tool response
                    tool_out = json.dumps({"test_case_results": test_case_results})
                    trace.append({"source": "tool", "role": "tool", "content": tool_out, "tool_call_id": tool_call_id})
                    orchestrator_messages.append({"role": "tool", "content": tool_out, "tool_call_id": tool_call_id})
                    self.dp_print(data_point, f"result: success={eval_result.get('is_correct', False)}")

                    # If successful, save as final code
                    if eval_result.get("is_correct", False):
                        final_code = submitted_code
                        self.dp_print(data_point, "submission successful!")
                        break

                # If we got a successful submission, we can break the outer loop
                if final_code:
                    break

                # Continue to next orchestrator turn with tool results
                continue

            # Check if orchestrator provided final solution (no tool calls)
            final_code = self._extract_cpp(msg.get("content", ""))
            if final_code:
                self.dp_print(data_point, f"orchestrator: final solution provided (code_len={len(final_code)})")
                break

            # Try to extract agent spawn request
            agents = self._extract_agents_json(msg.get("content", ""))
            if not agents:
                self.dp_print(data_point, "orchestrator: no agents JSON or final cpp found")
                feedback = (
                    f"No valid agents JSON or final ```cpp``` solution found. "
                    f"You have {remaining_steps - 1} steps remaining."
                )
                orchestrator_messages.append({"role": "user", "content": feedback})
                trace.append({"source": "orchestrator", "role": "user", "content": feedback})
                continue

            # Spawn agents
            num_agents = len(agents)
            total_agents_spawned += num_agents
            self.dp_print(data_point, f"spawning {num_agents} agents (total_spawned={total_agents_spawned})")

            agent_results = await self._spawn_agents(agents, problem)

            # Collect token counts
            for result in agent_results:
                num_agent_tokens.append(result.get("num_generated_tokens", 0))

            # Log agent results to trace
            for result in agent_results:
                trace.append(
                    {
                        "source": "agent",
                        "agent_id": result["agent_id"],
                        "role": result["role"],
                        "content": result["generation"],
                        "reasoning_content": result.get("reasoning_content", ""),
                    }
                )

            self.dp_print(
                data_point,
                f"agents completed: agent_tokens={sum(result.get('num_generated_tokens', 0) for result in agent_results)}",
            )

            # Format solutions for orchestrator
            solutions_text = self._format_agent_solutions(agent_results)
            orchestrator_messages.append({"role": "user", "content": solutions_text})
            trace.append({"source": "orchestrator", "role": "user", "content": solutions_text})

        # If no final solution after max_steps, try to extract from last agent output
        if not final_code and agent_results:
            self.dp_print(data_point, "no final solution, trying to extract from last agent")
            for result in reversed(agent_results):
                code = self._extract_cpp(result["generation"])
                if code:
                    final_code = code
                    break

        out = {
            "id": data_point["id"],
            "generation": final_code,
            "messages": trace,
            "num_generated_tokens": sum(num_orchestrator_tokens) + sum(num_agent_tokens),
            "num_generated_tokens_list": {"orchestrator": num_orchestrator_tokens, "agents": num_agent_tokens},
            "total_agents_spawned": total_agents_spawned,
        }

        if out_of_context:
            out["error"] = "_ran_out_of_context_"
            self.dp_print(data_point, "stopped: out_of_context")
        elif final_code:
            self.dp_print(data_point, f"completed with solution (code_len={len(final_code)})")
        else:
            self.dp_print(data_point, "completed without solution")

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


GENERATION_TASK_CLASS = OrchestratorAgentGenerationTask


@hydra.main(version_base=None, config_name="base_orchestrator_agent_config")
def orchestrator_agent_generation(cfg: OrchestratorAgentConfig):
    cfg = OrchestratorAgentConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)
    task = OrchestratorAgentGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(OrchestratorAgentConfig, server_params=server_params())

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        orchestrator_agent_generation()
