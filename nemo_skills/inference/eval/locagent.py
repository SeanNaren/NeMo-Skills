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
from dataclasses import field

import hydra
import openai

from nemo_skills.inference.eval.locagent_utils.utils import tree_repo_dict, filter_repo_dict
from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, remove_thinking, setup_logging

PROMPT_TEMPLATE_VERSION: str = "v5"  # Template version for prompts

import importlib

module_base = f"nemo_skills.inference.eval.locagent_utils.{PROMPT_TEMPLATE_VERSION}"

dialog_processor = importlib.import_module(f"{module_base}.dialog_processor")
tool_executor = importlib.import_module(f"{module_base}.tool_executor")

DialogProcessor = dialog_processor.DialogProcessor
ToolExecutor = tool_executor.ToolExecutor

LOG = logging.getLogger(get_logger_name(__file__))


tool_call_template = """

"""


@nested_dataclass(kw_only=True)
class LocalAgentGenerationConfig(GenerateSolutionsConfig):
    # Core inference settings
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters
    server: dict = field(default_factory=dict)  # Server configuration for model hosting
    
    # Agent behavior settings
    mount_directory: str = "/repos/"  # Directory where repositories are mounted
    remove_thinking: bool = True  # Whether to strip thinking tags from output
    total_steps: int = 7  # Maximum number of agent steps per problem
    
    # Repository filtering settings
    file_extensions: list = field(default_factory=lambda: ["py"])  # File types to include in repo
    exclude_dirs: list = field(  # Directory names to exclude from repository analysis
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
    
    # Tool detection and search settings
    enable_implicit_tool_detection: bool = True  # Enable fallback tool detection when no explicit calls found
    common_words_filter: list = field(  # Words to filter out when detecting implicit search queries
        default_factory=lambda: [
            "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "is", "are", "was", "were", "be", "been", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "may", "might", "can",
            "this", "that", "these", "those", "a", "an", "as", "if", "then", "else",
            "when", "where", "why", "how", "what", "which", "who", "whom", "whose",
            "need", "find", "search", "look", "function", "class", "method", "variable", "query"
        ]
    )
    
    # Display settings
    show_line_counts: bool = False  # Show file line counts in repository tree output
    max_view_lines: int = 1000  # Maximum lines to show in view tool (0 = no limit)


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_locagent_generation_config", node=LocalAgentGenerationConfig)


class LocAgentGenerationTask(GenerationTask):
    def __init__(self, cfg: LocalAgentGenerationConfig):
        super().__init__(cfg)
        self.tool_executor = ToolExecutor(cfg)

    def log_example_prompt(self, data):
        return

    async def process_single_datapoint(self, data_point, all_data):
        """Will do all necessary generations to get a single answer for the data point."""
        # Filter out samples with empty problem statements
        if not data_point.get('problem_statement', '').strip():
            LOG.warning(f"Skipping data point {data_point.get('instance_id', 'unknown')} due to empty problem statement")
            return {
                'generation': [],
                'total_generated_tokens': 0,
                'num_turns': 0,
                'status': 'skipped',
                'reason': 'empty_problem_statement',
            }
        
        total_steps = self.cfg.total_steps
        chat_history = []

        total_generated_tokens = 0
        instance_filepath = Path(self.cfg.mount_directory).joinpath(f"{data_point['instance_id']}.pkl")

        # repo_dict is dict with 'structure' containing the actual repo tree dict_keys(['repo', 'base_commit',
        # 'structure', 'instance_id'])
        with open(instance_filepath, 'rb') as f:
            repo_dict = pickle.load(f)
        repo_dict = filter_repo_dict(repo_dict, self.cfg.exclude_dirs, self.cfg.file_extensions)
        tree_structure = tree_repo_dict(repo_dict, self.cfg.show_line_counts)

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
                
                # Check if we've reached the maximum steps without success
                if cur_step == total_steps - 1 and status is None:
                    status = "failed"
                    reason = "max_steps_exceeded"
                    break

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


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        HELP_MESSAGE = get_help_message(
            LocalAgentGenerationConfig,
            server_params=server_params(),
        )
        print(HELP_MESSAGE)
    else:
        setup_logging()
        locagent_generation()
