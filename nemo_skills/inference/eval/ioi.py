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

import logging
import re
import sys
from dataclasses import field

import hydra
import openai

from nemo_skills.code_execution import format_code_output
from nemo_skills.code_execution.sandbox import LocalSandbox
from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.model import server_params
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


def extract_code_block_and_test_input(text: str, language: str):
    py_matches = re.findall(rf"```{language}(.*?)```", text, re.DOTALL)
    in_matches = re.findall(r"```inputs(.*?)```", text, re.DOTALL)
    return py_matches[-1].strip() if py_matches else None, in_matches[-1].strip() if in_matches else None


@nested_dataclass(kw_only=True)
class IOIExecutionConfig(GenerateSolutionsConfig):
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters
    server: dict = field(default_factory=dict)
    prompt_config: str = "eval/ioi/codegen_tests"
    fix_prompt_config: str = "eval/ioi/codegen_improve"
    language: str = "python"
    total_steps: int = 5


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_ioi_generation_config", node=IOIExecutionConfig)


class IOIExecutionGenerationTask(GenerationTask):
    def __init__(self, cfg: IOIExecutionConfig):
        super().__init__(cfg)
        self.repair_prompt = get_prompt(
            self.cfg.fix_prompt_config,
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
        total_steps = self.cfg.total_steps
        chat_history = []
        print("Processing single element")
        llm_output = await super().process_single_datapoint(data_point, all_data)
        print("Processed")
        chat_history.append(llm_output)

        # run execution/improve steps
        for cur_step in range(total_steps):
            code_block, test_inputs = extract_code_block_and_test_input(llm_output["generation"], self.cfg.language)
            print("extracted code block and inputs", code_block, test_inputs)
            if not code_block or not test_inputs:
                raise ValueError(
                    f"Failed to generate a code block and test input, received: {llm_output}"
                )

            output, _ = self.sandbox.execute_code(
                generated_code=code_block,
                std_input=test_inputs,
                language=self.cfg.language
            )
            print("sandbox output", output)
            data_point['solution'] = f"```{self.cfg.language}\n{code_block}\n```"
            data_point['inputs'] = f"```inputs\n{test_inputs}\n```"
            data_point['output'] = format_code_output(
                output,
                code_output_begin=self.prompt.config.code_tags.code_output_begin,
                code_output_end=self.prompt.config.code_tags.code_output_end,
                code_output_format=self.prompt.config.code_tags.code_output_format
            )
            try:
                llm_output = await super().process_single_datapoint(data_point, all_data, prompt=self.repair_prompt)
            # TODO: this is a hack (as not all servers return that),
            # but eventually we should support handling errors like this globally for all generations
            except openai.BadRequestError as e:
                if 'Please reduce the length of the messages or completion' in str(e):
                    LOG.warning(
                        "LocAgent generation failed due to running out of context. " "Failing for subsequent subtasks automatically.",
                    )
                raise e
            chat_history.append(llm_output)

        code_block, test_inputs = extract_code_block_and_test_input(llm_output["generation"], self.cfg.language)
        return {'generation': code_block, 'steps': chat_history}


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
