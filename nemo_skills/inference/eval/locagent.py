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
import pickle
import sys
from dataclasses import field

import hydra
import openai

from nemo_skills.inference.eval.locagent_utils.dialog_processor import DialogProcessor
from nemo_skills.inference.eval.locagent_utils.tool_executor import ToolExecutor
from nemo_skills.inference.eval.locagent_utils.utils import tree_structure_from_pickle
from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, remove_thinking, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


tool_call_template = """

"""


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
    show_line_counts: bool = True


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_locagent_generation_config", node=LocalAgentGenerationConfig)


class LocAgentGenerationTask(GenerationTask):
    def __init__(self, cfg: LocalAgentGenerationConfig):
        super().__init__(cfg)
        self.tool_executor = ToolExecutor(cfg)

    async def process_single_datapoint(self, data_point, all_data):
        """Will do all necessary generations to get a single answer for the data point."""
        total_steps = self.cfg.total_steps
        previous_llm_code = [None] * total_steps
        chat_history = [data_point]
        task_solutions = {}
        total_generated_tokens = 0
        instance_filepath = f"{self.cfg.mount_directory}/{data_point['instance_id']}.pickle"

        with open(instance_filepath, 'rb') as f:
            data = pickle.load(f)
        tree_structure = tree_structure_from_pickle(data, self.cfg.exclude_dirs)

        data_point['repo_tree'] = tree_structure

        for cur_step in range(total_steps):
            try:
                llm_output = await super().process_single_datapoint(chat_history, all_data)
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
            extracted_block = DialogProcessor.extract_response(text=llm_output['generation'], remove_thinking=self.cfg.remove_thinking)

            if not extracted_block:
                LOG.warning("Model failed to generate a tool use or location. Ending generation.")
                # todo (hov): add resampling with different temperature if necessary.
                break

            if extracted_block["type"] == "locations":
                break

            if extracted_block["type"] == "tool_calls":
                tool_call_result = self.tool_executor.execute_tool(extracted_block, data)

            LOG.info(tool_call_result)
            break

        # generation is a dict["problem_id.subtask_step": full_solution] here
        LOG.info("Succesfully executed!")
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
