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

"""
VARIANT 6: Minimal Classic (Pre-All-Improvements)
HYPOTHESIS: ALL the advanced features added after Sept 9 are hurting performance
Strip everything back to the bare minimum like the successful run
"""

import copy
import importlib
import logging
import pickle
import sys
from dataclasses import field
from pathlib import Path

import hydra

import openai
from nemo_skills.inference.eval.artsiv_utils.utils import (calculate_ground_truth_percentage,
                                                             extract_locations_from_patch, filter_repo_dict,
                                                             tree_repo_dict)
from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, remove_thinking, setup_logging

PROMPT_TEMPLATE_VERSION: str = "v4"

module_base = f"nemo_skills.inference.eval.artsiv_utils.{PROMPT_TEMPLATE_VERSION}"

dialog_processor = importlib.import_module(f"{module_base}.dialog_processor")
tool_executor = importlib.import_module(f"{module_base}.tool_executor")

DialogProcessor = dialog_processor.DialogProcessor
ToolExecutor = tool_executor.ToolExecutor
truncate_dialogue_history = dialog_processor.truncate_dialogue_history

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class ArtsivGenerationConfig(GenerateSolutionsConfig):
    # MINIMAL: Only core inference settings from successful run
    inference: InferenceConfig = field(default_factory=lambda: InferenceConfig(
        temperature=0.7,
        top_k=0,
        top_p=0.95,
        min_p=0.0,
        random_seed=0,
        tokens_to_generate=81920,  # Full allocation - no reductions
        repetition_penalty=1.0,
        top_logprobs=None,
        extra_body={}
    ))
    server: dict = field(default_factory=dict)

    # Essential settings only
    mount_directory: str = "/repos/"
    remove_thinking: bool = True
    total_steps: int = 20

    # Essential repository filtering
    file_extensions: list = field(default_factory=lambda: ["py", "cfg"])
    exclude_dirs: list = field(
        default_factory=lambda: [
            "test", "tests", "testing", "test_", "_test", "__pycache__", ".git", ".github",
            "docs", "examples", "scripts", "tools", "venv", "env", "node_modules", "dist",
            "build", "target", "bin", "obj", "coverage", ".pytest_cache", ".tox", ".mypy_cache",
            "locale", "translations", "i18n", "l10n", "static", "assets", "media", "uploads",
            "logs", "tmp", "temp", "vendor", "libs", "dependencies", "settings", "local_settings",
            "fixtures", "data", "datasets", "notebooks", "jupyter", "ipynb_checkpoints", "deploy",
            "deployment", "docker", "kubernetes", "ci", "cd", "github", "gitlab", "bitbucket",
            "readme", "license", "changelog", "contributing",
        ]
    )

    # Essential tool detection
    enable_implicit_tool_detection: bool = True
    common_words_filter: list = field(
        default_factory=lambda: [
            "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "can", "this", "that", "these", "those",
            "a", "an", "as", "if", "then", "else", "when", "where", "why", "how", "what", "which",
            "who", "whom", "whose", "need", "find", "search", "look", "function", "class", "method",
            "variable", "query",
        ]
    )

    # Essential context
    max_seq_length: int = 262144
    show_line_counts: bool = False
    max_view_lines: int = 1000

    # MINIMAL: Only basic truncation
    truncation_strategy: str = "bookend"
    
    # DISABLE ALL ADVANCED FEATURES THAT WERE ADDED POST-SEPT-9
    # (Keep fields for hydra compatibility but disable functionality)
    enable_loop_detection: bool = False  # Disable - might interfere
    loop_detection_threshold: int = 3  # Keep field for hydra
    
    enable_enhanced_context: bool = False  # Disable - uses complex logic
    context_safety_margin: float = 0.9  # Keep field for hydra  
    use_tiktoken: bool = False  # Disable - part of complex context management
    
    enable_final_turn_prompt: bool = False  # Disable - was added later
    final_turn_instruction_type: str = "aligned"  # Keep field for hydra
    final_turn_threshold: float = 1.0  # Keep field for hydra
    
    enable_response_length_management: bool = False  # DEFINITELY disable - added Sept 13


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_artsiv_generation_config", node=ArtsivGenerationConfig)


class ArtsivGenerationTask(GenerationTask):
    def __init__(self, cfg: ArtsivGenerationConfig):
        # MINIMAL: No token buffering, no adjustments - pure vintage approach
        super().__init__(cfg)
        self.tool_executor = ToolExecutor(cfg)
        LOG.info(f"Using simple truncation strategy: {cfg.truncation_strategy}")

    def log_example_prompt(self, data):
        return

    async def process_single_datapoint(self, data_point, all_data):
        """MINIMAL: Stripped down to essential functionality only."""

        if not data_point.get('problem_statement', '').strip():
            LOG.warning(
                f"Skipping data point {data_point.get('instance_id', 'unknown')} due to empty problem statement"
            )
            return {
                'generation': [],
                'total_generated_tokens': 0,
                'num_turns': 0,
                'status': 'skipped',
                'reason': 'empty_problem_statement',
                'turns': [],
            }

        total_steps = self.cfg.total_steps
        chat_history = []
        total_generated_tokens = 0

        # Simple turn initialization
        if 'turns' in data_point and isinstance(data_point['turns'], list) and len(data_point['turns']) > 0:
            for i, turn in enumerate(data_point['turns']):
                if isinstance(turn, dict):
                    turn.setdefault('inputs', '')
                    turn.setdefault('assistant', '')
                    turn.setdefault('tool_call', None)
                    turn.setdefault('tool_output', '')
                else:
                    data_point['turns'][i] = {"inputs": "", "assistant": "", "tool_call": None, "tool_output": ""}
        else:
            data_point['turns'] = [{"inputs": "", "assistant": "", "tool_call": None, "tool_output": ""}]

        try:
            instance_filepath = Path(self.cfg.mount_directory).joinpath(f"{data_point['instance_id']}.pkl")

            with open(instance_filepath, 'rb') as f:
                repo_dict = pickle.load(f)
            repo_dict = filter_repo_dict(repo_dict, self.cfg.exclude_dirs, self.cfg.file_extensions)
            tree_structure = tree_repo_dict(repo_dict, self.cfg.show_line_counts)

            ground_truth_in_repo_percentage = 0.0
            if 'patch' in data_point and data_point['patch']:
                try:
                    locations = extract_locations_from_patch(data_point['patch'])
                    ground_truth_in_repo_percentage, debug_info = calculate_ground_truth_percentage(
                        repo_dict, locations, self.cfg.exclude_dirs, self.cfg.file_extensions
                    )
                    data_point['_missing_ground_truth_files'] = debug_info.get('missing_files_details', [])
                except Exception as e:
                    LOG.warning(f"Error checking ground truth files: {e}")

            data_point['_ground_truth_in_repo_percentage'] = ground_truth_in_repo_percentage

            inputs = f"""
### Problem Description
{data_point["problem_statement"]}

### Repository Structure
{tree_structure}
"""

            data_point['turns'][0]['inputs'] = inputs

        except Exception as e:
            LOG.error(f"Error loading repository for instance {data_point.get('instance_id', 'unknown')}: {e}")
            return {
                'generation': [],
                'total_generated_tokens': 0,
                'num_turns': 0,
                'status': 'failed',
                'reason': f'repository_loading_error: {str(e)}',
                'turns': data_point['turns'],
            }

        reason = None
        status = None
        try:
            for cur_step in range(total_steps):
                if (
                    'turns' not in data_point
                    or not isinstance(data_point['turns'], list)
                    or len(data_point['turns']) == 0
                ):
                    LOG.error(f"Invalid turns structure at step {cur_step}")
                    status = "failed"
                    reason = "invalid_turns_structure"
                    break

                # MINIMAL: Only basic truncation if needed
                if hasattr(self.cfg, 'max_seq_length') and self.cfg.max_seq_length is not None and self.cfg.max_seq_length > 0:
                    original_turns_count = len(data_point['turns'])
                    
                    # Use simple sequential truncation
                    data_point['turns'] = truncate_dialogue_history(
                        data_point['turns'], self.cfg.max_seq_length, self.cfg.inference.tokens_to_generate
                    )
                    
                    if len(data_point['turns']) < original_turns_count:
                        LOG.info(f"Truncated dialogue from {original_turns_count} to {len(data_point['turns'])} turns")

                prepared_data_point = copy.deepcopy(data_point)

                # MINIMAL: Direct LLM call with no interference
                try:
                    LOG.info(f"Sending {len(prepared_data_point['turns'])} turns to LLM")
                    llm_output = await super().process_single_datapoint(prepared_data_point, all_data)
                    
                except openai.BadRequestError as e:
                    if 'Please reduce the length of the messages or completion' in str(e) or 'is longer than the model\'s context length' in str(e):
                        LOG.warning("Generation failed due to context length. Failing gracefully.")
                        status = "failed"
                        reason = "context_length_exceeded"
                        break
                    LOG.warning(f"Generation failed with BadRequestError: {e}")
                    status = "failed"
                    reason = f"bad_request_error: {str(e)}"
                    break

                generated_tokens = llm_output.get('num_generated_tokens', 0)
                total_generated_tokens += generated_tokens
                
                # Simple truncation check
                if generated_tokens == self.cfg.inference.tokens_to_generate:
                    LOG.warning(f"Model hit token limit ({generated_tokens} tokens). Response may be incomplete.")
                    llm_output['_likely_truncated'] = True

                chat_history.append(llm_output)

                # Simple thinking removal
                if self.cfg.remove_thinking:
                    remove_thinking(llm_output, 'generation', self.cfg.thinking_begin, self.cfg.thinking_end)

                # Simple response extraction
                try:
                    extracted_block = DialogProcessor.extract_response(llm_output['generation'], self.cfg)
                except Exception as e:
                    LOG.error(f"Error extracting response: {e}")
                    status = "failed"
                    reason = f"response_extraction_error: {str(e)}"
                    break

                if not extracted_block:
                    LOG.warning("No tool use or location found. Ending generation.")
                    if llm_output.get('_likely_truncated', False):
                        status = "failed"
                        reason = "response_truncated_at_token_limit"
                    else:
                        status = "failed"
                        reason = "no_tool_or_location_generated"
                    break

                # Simple turn update
                if data_point['turns'] and len(data_point['turns']) > 0:
                    current_turn = data_point['turns'][-1]
                    if isinstance(current_turn, dict):
                        current_turn['assistant'] = llm_output['generation']
                        current_turn['assistant_raw'] = llm_output.get('raw_generation', llm_output['generation'])
                        current_turn['assistant_raw_w_think'] = llm_output.get('_full_generation', llm_output['generation'])

                        if extracted_block:
                            if extracted_block.get("type") == "tool_calls":
                                current_turn['tool_call'] = extracted_block.get("tool_call", None)
                            elif extracted_block.get("type") == "locations":
                                current_turn['locations'] = extracted_block.get("locations", [])

                if extracted_block.get("type") == "tool_calls":
                    if "tool_call" not in extracted_block:
                        LOG.error(f"Missing 'tool_call' in extracted block")
                        status = "failed"
                        reason = "missing_tool_call_in_extracted_block"
                        break
                    tool_call_result = self.tool_executor.execute_tool(extracted_block["tool_call"], repo_dict)

                    if data_point['turns'] and len(data_point['turns']) > 0:
                        current_turn = data_point['turns'][-1]
                        if isinstance(current_turn, dict):
                            current_turn['tool_output'] = tool_call_result
                            
                            new_turn = {
                                "inputs": tool_call_result,
                                "assistant": "",
                                "tool_call": None,
                                "tool_output": "",
                            }
                            data_point['turns'].append(new_turn)

                elif extracted_block.get("type") == "locations":
                    if "locations" not in extracted_block:
                        LOG.error(f"Missing 'locations' in extracted block")
                        status = "failed"
                        reason = "missing_locations_in_extracted_block"
                        break
                    data_point["locations"] = extracted_block["locations"]
                    status = "success"
                    reason = None
                    break

                if cur_step == total_steps - 1 and status is None:
                    status = "failed"
                    reason = "max_steps_exceeded"
                    break

            if status is None:
                status = "failed"
                reason = "unknown_failure"
                
        except Exception as e:
            LOG.error(f"Unexpected error: {e}")
            status = "failed"
            reason = f"exception: {str(e)}"

        # Simple cleanup
        if 'turns' not in data_point:
            data_point['turns'] = []

        for i, turn in enumerate(data_point.get('turns', [])):
            if not isinstance(turn, dict):
                data_point['turns'][i] = {
                    "inputs": str(turn) if turn else "",
                    "assistant": "",
                    "tool_call": None,
                    "tool_output": "",
                }
            else:
                turn.setdefault('inputs', '')
                turn.setdefault('assistant', '')
                turn.setdefault('tool_call', None)
                turn.setdefault('tool_output', '')

        ground_truth_in_repo_percentage = data_point.get('_ground_truth_in_repo_percentage', 0.0)
        missing_ground_truth_files = data_point.get('_missing_ground_truth_files', [])

        data_point.pop('_ground_truth_in_repo_percentage', None)
        data_point.pop('_missing_ground_truth_files', None)

        return {
            'generation': chat_history,
            'total_generated_tokens': total_generated_tokens,
            'num_turns': len(chat_history),
            'status': status,
            'reason': reason,
            'turns': data_point.get('turns', []),
            'ground_truth_in_repo_percentage': ground_truth_in_repo_percentage,
            'missing_ground_truth_files': missing_ground_truth_files,
        }


GENERATION_TASK_CLASS = ArtsivGenerationTask

@hydra.main(version_base=None, config_name='base_artsiv_generation_config')
def artsiv_generation(cfg: ArtsivGenerationConfig):
    cfg = ArtsivGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = ArtsivGenerationTask(cfg)
    task.generate()


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        HELP_MESSAGE = get_help_message(
            ArtsivGenerationConfig,
            server_params=server_params(),
        )
        print(HELP_MESSAGE)
    else:
        setup_logging()
        artsiv_generation()
