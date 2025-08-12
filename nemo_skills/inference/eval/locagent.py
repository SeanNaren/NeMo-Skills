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
from pathlib import Path
from copy import deepcopy
from dataclasses import field

import hydra
import openai

from nemo_skills.inference.eval.locagent_utils.dialog_processor import DialogProcessor
from nemo_skills.inference.eval.locagent_utils.tool_executor import ToolExecutor
from nemo_skills.inference.eval.locagent_utils.utils import tree_repo_dict, filter_repo_dict
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
    mount_directory: str = "/repos/"
    remove_thinking: bool = True
    total_steps: int = 5

    # CHANGED: Switched from set to list for OmegaConf compatibility
    file_extensions: list = field(default_factory=lambda: ["py"])
    
    # Whether to enable implicit tool call detection (fallback when no explicit tool calls found)
    enable_implicit_tool_detection: bool = True
    
    # Common words to filter out when detecting implicit search queries
    common_words_filter: list = field(
        default_factory=lambda: [
            "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "is", "are", "was", "were", "be", "been", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "may", "might", "can",
            "this", "that", "these", "those", "a", "an", "as", "if", "then", "else",
            "when", "where", "why", "how", "what", "which", "who", "whom", "whose",
            "need", "find", "search", "look", "function", "class", "method", "variable", "query"
        ]
    )

    # CHANGED: Switched from set to list for OmegaConf compatibility
    exclude_dirs: list = field(
        default_factory=lambda: [
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
        ]
    )
    show_line_counts: bool = True


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_locagent_generation_config", node=LocalAgentGenerationConfig)


class LocAgentGenerationTask(GenerationTask):
    def __init__(self, cfg: LocalAgentGenerationConfig):
        super().__init__(cfg)
        self.tool_executor = ToolExecutor()

    def log_example_prompt(self, data):
        return

    async def process_single_datapoint(self, data_point, all_data):
        """Will do all necessary generations to get a single answer for the data point."""
        total_steps = self.cfg.total_steps
        chat_history = []

        total_generated_tokens = 0
        instance_filepath = Path(self.cfg.mount_directory).joinpath(f"{data_point['instance_id']}.pkl")

        # repo_dict is dict with 'structure' containing the actual repo tree dict_keys(['repo', 'base_commit',
        # 'structure', 'instance_id'])
        with open(instance_filepath, 'rb') as f:
            repo_dict = pickle.load(f)
        repo_dict = filter_repo_dict(repo_dict, self.cfg.exclude_dirs, self.cfg.file_extensions)
        tree_structure = tree_repo_dict(repo_dict)

        inputs = f"""
### Problem Description
{data_point["problem_statement"]}

### Repository Structure
{tree_structure}
"""

        data_point['turns'] = [{"inputs": inputs}]

        reason = None
        status = None
        try:
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
                        status = "failed"
                        reason = "context_length_exceeded"
                        break
                    # For any other BadRequestError, also fail gracefully and store the error
                    LOG.warning(f"LocAgent generation failed with BadRequestError: {e}")
                    status = "failed"
                    reason = f"bad_request_error: {str(e)}"
                    break

                total_generated_tokens += llm_output.get('num_generated_tokens', 0)

                chat_history.append(llm_output)

                if self.cfg.remove_thinking:
                    remove_thinking(llm_output, 'generation', self.cfg.thinking_begin, self.cfg.thinking_end)
                extracted_block = DialogProcessor.extract_response(llm_output['generation'], self.cfg)

                if not extracted_block:
                    LOG.warning("Model failed to generate a tool use or location. Ending generation.")
                    # todo (hov): add resampling with different temperature if necessary.
                    status = "failed"
                    reason = "no_tool_or_location_generated"
                    break

                data_point['turns'][-1]['assistant'] = extracted_block
                data_point['turns'][-1]['assistant_raw'] = llm_output['generation']
                data_point['turns'][-1]['assistant_raw_w_think'] = llm_output.get('_full_generation', llm_output['generation'])
                if extracted_block["type"] == "tool_calls":
                    tool_call_result = self.tool_executor.execute_tool(extracted_block["tool_call"], repo_dict)
                    data_point['turns'].append({"inputs": tool_call_result})
                elif extracted_block["type"] == "locations":
                    data_point["locations"] = extracted_block["locations"]
                    status = "success"
                    reason = None
                    break

                print("Current messages", data_point['turns'])

            if status is None:
                # If we exit the loop without setting status, treat as failed
                status = "failed"
                if reason is None:
                    reason = "unknown_failure"
        except Exception as e:
            LOG.error(f"Unexpected error in process_single_datapoint: {e}")
            status = "failed"
            reason = f"exception: {str(e)}"

        # generation is a dict["problem_id.subtask_step": full_solution] here
        # """
        # [
        #     User: Problem statement,
        #     Assistant: {“generation”: generation with reasoning trace, “tool_call”: … },
        #     User: {“tool_output”: “”},
        #     Assistant: {“generation”: … “location”: …}
        # ]
        # """
        return {
            'generation': chat_history,
            'total_generated_tokens': total_generated_tokens,
            'num_turns': len(chat_history),
            'status': status,
            'reason': reason,
        }


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
