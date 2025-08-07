# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import logging
import os
import re
import sys
import tempfile
import time
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
            *compile_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, compile_stderr_bytes = await compiler_process.communicate(input=code_string.encode())

        if compiler_process.returncode != 0:
            raise RuntimeError(f"C++ compilation failed:\n{compile_stderr_bytes.decode()}")

        try:
            run_process = await asyncio.create_subprocess_exec(
                executable_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            run_stdout_bytes, run_stderr_bytes = await run_process.communicate()
            return run_stdout_bytes.decode(), run_stderr_bytes.decode()
        except FileNotFoundError:
            raise RuntimeError("Execution failed: Compiled executable not found.")


def extract_code_block(text: str):
    matches = re.findall(r"```cpp(.*?)```", text, re.DOTALL)
    return matches[-1].strip() if matches else None


def extract_test_input(text: str):
    matches = re.findall(r"```script(.*?)```", text, re.DOTALL)
    return matches[-1].strip() if matches else None


@nested_dataclass(kw_only=True)
class IOIExecutionConfig(GenerateSolutionsConfig):
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters
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
        self.test_prompt = get_prompt(
            self.cfg.test_prompt_config,
            self.cfg.prompt_template,
            self.cfg.code_tags,
            examples_type=self.cfg.examples_type
        )
        self.improve_prompt = get_prompt(
            self.cfg.improve_prompt_config,
            self.cfg.prompt_template,
            self.cfg.code_tags,
            examples_type=self.cfg.examples_type
        )
        self.improve_test_prompt = get_prompt(
            self.cfg.improve_test_prompt_config,
            self.cfg.prompt_template,
            self.cfg.code_tags,
            examples_type=self.cfg.examples_type
        )
        self.sandbox = LocalSandbox()
        self.use_repair_prompt = False

    def log_example_prompt(self, data):
        return

    async def process_single_datapoint(self, data_point, all_data, prompt=None):
        """Will do all necessary generations to get a single answer for the data point."""
        chat_history = []

        # generate an initial solution
        llm_output = await super().process_single_datapoint(data_point, all_data, prompt=self.prompt)
        cur_solution = extract_code_block(llm_output['generation'])

        if not cur_solution:
            raise ValueError(
                f"Failed to generate a solution, received {llm_output}"
            )

        chat_history.append(llm_output)

        # generate test inputs
        data_point['solution'] = cur_solution

        start_tests = time.time()
        tasks = [
            GenerationTask.process_single_datapoint(self, data_point, all_data, prompt=self.test_prompt)
            for _ in range(self.cfg.num_test_generations)
        ]
        all_results = await asyncio.gather(*tasks)
        print(f'time taken for test generation {time.time() - start_tests}s')

        print('\n'.join([x['generation'] for x in all_results]))

        test_script = next(
            (
                extracted for result in all_results
                if (extracted := extract_test_input(result["generation"]))
            ),
            None
        )

        if test_script is None:
            raise ValueError(
                f"Failed to extract a valid test input from {len(all_results)} attempts. Received results: {all_results}"
            )

        std_output, std_err = await compile_and_run_cpp(test_script)

        output = std_output + std_err

        for x in range(self.cfg.total_steps):

            data_point['output'] = output
            data_point['script'] = test_script
            data_point['solution'] = cur_solution

            improve_sol_output = super().process_single_datapoint(data_point, all_data, prompt=self.improve_prompt)
            improve_script_output = super().process_single_datapoint(data_point, all_data, prompt=self.improve_test_prompt)

            improve_sol_output, improve_script_output = await asyncio.gather(improve_sol_output, improve_script_output)

            cur_solution = extract_code_block(improve_sol_output['generation'])

            if not cur_solution:
                raise ValueError(
                    f"Failed to generate a solution, received {llm_output}"
                )

            test_script = extract_test_input(improve_script_output['generation'])

            if test_script is None:
                raise ValueError(
                    f"Failed to extract a valid test input from {len(all_results)} attempts. Received results: {all_results}"
                )

            std_output, std_err = await compile_and_run_cpp(test_script)

            output = std_output + std_err

            chat_history.append([improve_sol_output, improve_script_output, output])

        return {'generation': cur_solution, 'steps': chat_history}


GENERATION_TASK_CLASS = IOIExecutionGenerationTask


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name='base_ioi_generation_config')
def ioi_generation(cfg: IOIExecutionConfig):
    cfg = IOIExecutionConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = IOIExecutionGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    IOIExecutionConfig,
    server_params=server_params(),
)

if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        ioi_generation()
