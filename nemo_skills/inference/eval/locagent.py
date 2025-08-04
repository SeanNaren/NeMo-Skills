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
import sys
from dataclasses import field

import hydra
import openai
import re

from nemo_skills.code_execution.sandbox import LocalSandbox
from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, remove_thinking, setup_logging
from nemo_skills.inference.eval.locagent_utils.dialog_processor import DialogProcessor
from nemo_skills.inference.eval.locagent_utils.tool_executor import ToolExecutor

LOG = logging.getLogger(get_logger_name(__file__))


# def extract_code_block(text: str):
#     match = re.search(r"```(?:[^\n]*\n)?(.*?)```", text, re.DOTALL)
#     return match.group(1) if match else None


def extract_response(text: str):
    if self.cfg.remove_thinking and "</think>" in text:
        dialog_text = text.split("</think>")[1].lstrip().rstrip()
    else:
        dialog_text = text

    if "###Tool" in dialog_text:
        LOG.info("Found ###Tool block in output")
        return DialogProcessor._extract_tool_calls(dialog_text)
    elif "###Locations" in dialog_text:
        LOG.info("Found ###Locations block in output")
        return DialogProcessor._extract_locations(dialog_text)
    else:
        LOG.warning("No ###Tool or ###Locations found, checking for implicit tool calls")
        return DialogProcessor._extract_implicit_tool_calls(dialog_text)


def is_location_block(block_content: dict):
    return block_content["type"] == "locations"
    # return isinstance(block_content, str) and block_content.lstrip().startswith('###Locations')


def is_tool_call(extracted_block: dict):
    return block_content["type"] == "tool_calls"


@nested_dataclass(kw_only=True)
class LocalAgentGenerationConfig(GenerateSolutionsConfig):
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters
    server: dict = field(default_factory=dict)
    prompt_config: str = "eval/locagent/system"
    mount_directory: str = "/repos/"
    remove_thinking: bool = True
    total_steps: bool = 5
    file_extensions: set = {".py"}
    exclude_dirs: set = {
        "test",
        "tests",
        "testing",
        "test_",
        "_test",
        "__pycache__",
        ".git",
        ".github",
        "docs",
        "examples",
        "scripts",
        "tools",
        "utils",
        "migrations",
        "venv",
        "env",
        "node_modules",
        "dist",
        "build",
        "target",
        "bin",
        "obj",
        "coverage",
        ".pytest_cache",
        ".tox",
        ".mypy_cache",
        "locale",
        "translations",
        "i18n",
        "l10n",
        "static",
        "assets",
        "media",
        "uploads",
        "logs",
        "tmp",
        "temp",
        "cache",
        "vendor",
        "lib",
        "libs",
        "dependencies",
        "config",
        "conf",
        "settings",
        "local_settings",
        "fixtures",
        "data",
        "datasets",
        "notebooks",
        "jupyter",
        "ipynb_checkpoints",
        "deploy",
        "deployment",
        "docker",
        "kubernetes",
        "ci",
        "cd",
        "github",
        "gitlab",
        "bitbucket",
        "readme",
        "license",
        "changelog",
        "contributing",
    }


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_locagent_generation_config", node=LocalAgentGenerationConfig)


class LocAgentGenerationTask(GenerationTask):
    def __init__(self, cfg: LocalAgentGenerationConfig):
        super().__init__(cfg)
        self.sandbox = LocalSandbox()
        self.tool_executor = ToolExecutor()

    async def process_single_datapoint(self, data_point, all_data):
        """Will do all necessary generations to get a single answer for the data point."""
        total_steps = self.cfg.total_steps
        previous_llm_code = [None] * total_steps
        task_solutions = {}
        total_generated_tokens = 0

        # todo: execute tree structure so that we can append to prompt
        self.sandbox.execute_code("tree ")

        for cur_step in range(total_steps):
            try:
                llm_output = await super().process_single_datapoint(data_point, all_data)
            # TODO: this is a hack (as not all servers return that),
            # but eventually we should support handling errors like this globally for all generations
            except openai.BadRequestError as e:
                if 'Please reduce the length of the messages or completion' in str(e):
                    LOG.warning(
                        "LocAgent generation failed due to running out of context. " "Failing for subsequent subtasks automatically.",
                    )
                raise e

            total_generated_tokens += llm_output.get('num_generated_tokens', 0)

            previous_llm_code[cur_step] = llm_output

            if self.cfg.remove_thinking:
                remove_thinking(llm_output, 'generation', self.cfg.thinking_begin, self.cfg.thinking_end)
            extracted_block = extract_response(llm_output['generation'])

            if not extracted_block:
                LOG.warning("Model failed to generate a tool use or location. Ending generation.")
                # todo (hov): add resampling with different temperature if necessary.
                break

            if is_location_block(extracted_block):
                break

            if is_tool_call(extracted_block):
                tool_call_result = self.tool_executor.execute_tool(extracted_block, repo_dir)

        # generation is a dict["problem_id.subtask_step": full_solution] here
        return {'generation': task_solutions, 'num_generated_tokens': total_generated_tokens}


GENERATION_TASK_CLASS = LocAgentGenerationTask


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name='base_locagent_generation_config')
def locagent_generation(cfg: LocalAgentGenerationConfig):
    cfg = LocalAgentGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = LocAgentGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    LocalAgentGenerationConfig,
    server_params=server_params(),
)

if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        locagent_generation()
