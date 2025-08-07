import asyncio
import logging
import os
import re
import sys
import tempfile
from dataclasses import field

import hydra

from nemo_skills.code_execution.sandbox import LocalSandbox
from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.model import server_params
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


async def compile_and_run_cpp(code_string: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        executable_path = os.path.join(temp_dir, "a.out")
        compile_command = ["g++", "-x", "c++", "-o", executable_path, "-"]
        compiler_process = await asyncio.create_subprocess_exec(
            *compile_command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        _, compile_stderr = await compiler_process.communicate(input=code_string.encode())

        if compiler_process.returncode != 0:
            raise RuntimeError(f"C++ compilation failed:\n{compile_stderr.decode()}")

        run_process = await asyncio.create_subprocess_exec(
            executable_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        run_stdout, run_stderr = await run_process.communicate()
        return run_stdout.decode(), run_stderr.decode()


def extract_code_block(text: str):
    matches = re.findall(r"```cpp(.*?)```", text, re.DOTALL)
    return matches[-1].strip() if matches else None


def extract_test_input(text: str):
    matches = re.findall(r"```script(.*?)```", text, re.DOTALL)
    return matches[-1].strip() if matches else None


@nested_dataclass(kw_only=True)
class IOIExecutionConfig(GenerateSolutionsConfig):
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: dict = field(default_factory=dict)
    prompt_config: str = "eval/ioi/codegen"
    improve_prompt_config: str = "eval/ioi/codegen_improve"
    test_prompt_config: str = "eval/ioi/codegen_tests"
    improve_test_prompt_config: str = "eval/ioi/codegen_improve_test"
    language: str = "cpp"
    total_steps: int = 5
    num_test_generations: int = 5


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_ioi_generation_config", node=IOIExecutionConfig)


class IOIExecutionGenerationTask(GenerationTask):
    def __init__(self, cfg: IOIExecutionConfig):
        super().__init__(cfg)
        prompt_kwargs = {
            "prompt_template": cfg.prompt_template,
            "code_tags": cfg.code_tags,
            "examples_type": cfg.examples_type,
        }
        self.prompts = {
            "initial": get_prompt(cfg.prompt_config, **prompt_kwargs),
            "improve_solution": get_prompt(cfg.improve_prompt_config, **prompt_kwargs),
            "test_generation": get_prompt(cfg.test_prompt_config, **prompt_kwargs),
            "improve_test": get_prompt(cfg.improve_test_prompt_config, **prompt_kwargs),
        }
        self.sandbox = LocalSandbox()

    def log_example_prompt(self, data):
        pass

    async def _call_llm(self, data_point, all_data, prompt_key, **extra_data):
        return await super().process_single_datapoint(
            {**data_point, **extra_data}, all_data, prompt=self.prompts[prompt_key]
        )

    async def process_single_datapoint(self, data_point, all_data, prompt=None):
        chat_history = []

        solution_response = await self._call_llm(data_point, all_data, "initial")
        solution = extract_code_block(solution_response['generation'])
        if not solution:
            raise ValueError(f"Failed to generate an initial solution: {solution_response}")
        chat_history.append(solution_response)

        test_gen_tasks = [
            self._call_llm(data_point, all_data, "test_generation", solution=solution)
            for _ in range(self.cfg.num_test_generations)
        ]
        test_responses = await asyncio.gather(*test_gen_tasks)
        test_script = next((s for r in test_responses if (s := extract_test_input(r['generation']))), None)
        if not test_script:
            raise ValueError(f"Failed to extract a valid test script from {len(test_responses)} attempts.")

        for _ in range(self.cfg.total_steps):
            stdout, stderr = await compile_and_run_cpp(test_script)
            output = stdout + stderr

            common_args = {"solution": solution, "script": test_script, "output": output}

            # Keep a single task for solution improvement
            improve_sol_task = self._call_llm(data_point, all_data, "improve_solution", **common_args)

            # Create multiple tasks for script improvement
            improve_script_tasks = [
                self._call_llm(data_point, all_data, "improve_test", **common_args)
                for _ in range(self.cfg.num_test_generations)
            ]

            # Await the solution task and all script tasks in parallel
            sol_resp, *script_resps = await asyncio.gather(improve_sol_task, *improve_script_tasks)

            new_solution = extract_code_block(sol_resp['generation'])
            if not new_solution:
                raise ValueError(f"Failed to extract improved solution. Response: {sol_resp}")

            # Find the first valid script and its corresponding response from all attempts
            successful_extraction = next(
                ((resp, script) for resp in script_resps if (script := extract_test_input(resp['generation']))),
                (None, None)
            )
            script_resp, new_script = successful_extraction

            if not new_script:
                raise ValueError(f"Failed to extract a valid improved test script from {len(script_resps)} attempts.")

            solution, test_script = new_solution, new_script
            chat_history.append([sol_resp, script_resp, output])

        return {'generation': solution, 'steps': chat_history}

GENERATION_TASK_CLASS = IOIExecutionGenerationTask


@hydra.main(version_base=None, config_name='base_ioi_generation_config')
def ioi_generation(cfg: IOIExecutionConfig):
    cfg = IOIExecutionConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)
    task = IOIExecutionGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(IOIExecutionConfig, server_params=server_params())

if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        ioi_generation()