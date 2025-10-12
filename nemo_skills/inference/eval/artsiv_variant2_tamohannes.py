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

import copy
import importlib
import logging
import pickle
import sys
from dataclasses import field
from pathlib import Path

import hydra
import openai

from nemo_skills.inference.eval.artsiv_utils.context_manager import ContextManager
from nemo_skills.inference.eval.artsiv_utils.patch_processor import PatchProcessor
from nemo_skills.inference.eval.artsiv_utils.repo_manager import RepoManager
from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, remove_thinking, setup_logging

module_base = "nemo_skills.inference.eval.artsiv_utils"

dialog_processor = importlib.import_module(f"{module_base}.dialog_processor")
tool_executor = importlib.import_module(f"{module_base}.tool_executor")

DialogProcessor = dialog_processor.DialogProcessor
ToolExecutor = tool_executor.ToolExecutor

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class ArtsivGenerationConfig(GenerateSolutionsConfig):
    inference: InferenceConfig = field(
        default_factory=lambda: InferenceConfig(
            temperature=0.99,
            top_k=0,
            top_p=0.95,
            min_p=0.0,
            random_seed=0,
            tokens_to_generate=81920,
            repetition_penalty=1.0,
            top_logprobs=None,
            extra_body={},
        )
    )
    server: dict = field(default_factory=dict)

    # Agent behavior settings
    mount_directory: str = "/repos/"
    remove_thinking: bool = True  # Keep thinking removal
    total_steps: int = 20

    # Repository filtering settings
    file_extensions: list = field(default_factory=lambda: ["py", "cfg"])
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
            "vendor",
            "libs",
            "dependencies",
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

    # Tool detection settings
    enable_implicit_tool_detection: bool = True
    common_words_filter: list = field(
        default_factory=lambda: [
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "this",
            "that",
            "these",
            "those",
            "a",
            "an",
            "as",
            "if",
            "then",
            "else",
            "when",
            "where",
            "why",
            "how",
            "what",
            "which",
            "who",
            "whom",
            "whose",
            "need",
            "find",
            "search",
            "look",
            "function",
            "class",
            "method",
            "variable",
            "query",
        ]
    )

    max_seq_length: int = 262144
    show_line_counts: bool = False
    max_view_lines: int = 1000

    # Loop detection settings
    enable_loop_detection: bool = True
    loop_detection_threshold: int = 3

    # Enhanced context management settings
    enable_enhanced_context: bool = True
    context_safety_margin: float = 0.9
    use_tiktoken: bool = True

    # Final turn prompt settings
    enable_final_turn_prompt: bool = True
    final_turn_threshold: float = 1.0

    # Response length management
    enable_response_length_management: bool = True
    max_retries: int = 2
    inject_length_warnings: bool = True
    response_warning_threshold: float = 0.75
    response_critical_threshold: float = 0.9


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_artsiv_generation_config", node=ArtsivGenerationConfig)


class ArtsivGenerationTask(GenerationTask):
    def __init__(self, cfg: ArtsivGenerationConfig):
        super().__init__(cfg)
        self.tool_executor = ToolExecutor(cfg)

    def log_example_prompt(self, data):
        return

    async def process_single_datapoint(self, data_point, all_data):
        """Will do all necessary generations to get a single answer for the data point."""

        # Visual separator for better log readability
        instance_id = data_point.get('instance_id', 'unknown')
        
        # Create a logging prefix for this instance
        log_prefix = f"[{instance_id}] "
        
        # Helper function to log with instance prefix
        def log_info(msg, indent=0):
            LOG.info(f"{log_prefix}{' ' * indent}{msg}")
            
        def log_debug(msg, indent=0):
            LOG.debug(f"{log_prefix}{' ' * indent}{msg}")
            
        def log_error(msg, indent=0):
            LOG.error(f"{log_prefix}{' ' * indent}{msg}")
            
        def log_warning(msg, indent=0):
            LOG.warning(f"{log_prefix}{' ' * indent}{msg}")
        
        log_info("\n" + "=" * 100)
        log_info(f"{'='*15} PROCESSING SAMPLE: {instance_id} {'='*15}")
        log_info("=" * 100 + "\n")

        log_debug(
            f"Initial data_point keys: {list(data_point.keys()) if isinstance(data_point, dict) else 'not a dict'}"
        )
        if 'turns' in data_point:
            log_debug(f"Initial turns structure: {data_point['turns']}")

        if not data_point.get('problem_statement', '').strip():
            log_warning(
                f"Skipping data point due to empty problem statement"
            )

            # Visual separator for skipped sample
            log_info("\n" + "=" * 100)
            log_info(f"{'='*15} SKIPPED SAMPLE: {instance_id} (empty problem statement) {'='*15}")
            log_info("=" * 100 + "\n")

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

        if 'turns' in data_point and isinstance(data_point['turns'], list) and len(data_point['turns']) > 0:
            for i, turn in enumerate(data_point['turns']):
                if isinstance(turn, dict):
                    turn.setdefault('turn_id', i)  # Add turn_id
                    turn.setdefault('inputs', '')
                    turn.setdefault('assistant', '')
                    turn.setdefault('tool_call', None)
                    turn.setdefault('tool_output', '')
                    log_debug(
                        f"Turn {i} after setdefault - keys: {list(turn.keys())}, has assistant: {'assistant' in turn}",
                        indent=4
                    )
                else:
                    log_warning(f"Found non-dict turn at index {i}: {type(turn)}, replacing with empty structure", indent=4)
                    data_point['turns'][i] = {
                        "turn_id": i,
                        "inputs": "", 
                        "assistant": "", 
                        "tool_call": None, 
                        "tool_output": "",
                        "_retry_count": 0,
                    }
        else:
            data_point['turns'] = [{
                "turn_id": 0,
                "inputs": "", 
                "assistant": "", 
                "tool_call": None, 
                "tool_output": "",
                "_retry_count": 0,
            }]

        try:
            instance_filepath = Path(self.cfg.mount_directory).joinpath(f"{data_point['instance_id']}.pkl")

            with open(instance_filepath, 'rb') as f:
                repo_dict = pickle.load(f)
            repo_dict = RepoManager.filter_repo_dict(repo_dict, self.cfg.exclude_dirs, self.cfg.file_extensions)
            tree_structure = RepoManager.tree_repo_dict(repo_dict, self.cfg.show_line_counts)

            ground_truth_in_repo_percentage = 0.0
            if 'patch' in data_point and data_point['patch']:
                try:
                    locations = PatchProcessor.extract_locations_from_patch(data_point['patch'])
                    ground_truth_in_repo_percentage, debug_info = RepoManager.calculate_ground_truth_percentage(
                        repo_dict, locations, self.cfg.exclude_dirs, self.cfg.file_extensions
                    )
                    log_debug(f"Ground truth check debug info: {debug_info}", indent=4)
                    data_point['_missing_ground_truth_files'] = debug_info.get('missing_files_details', [])
                except Exception as e:
                    log_warning(f"Error checking ground truth files: {e}", indent=4)

            data_point['_ground_truth_in_repo_percentage'] = ground_truth_in_repo_percentage

            inputs = f"""
### Problem Description
{data_point["problem_statement"]}

### Repository Structure
{tree_structure}
"""

            data_point['turns'][0]['inputs'] = inputs
            data_point['turns'][0]['turn_id'] = 0  # Ensure turn_id is set
            # Estimate tokens for initial problem statement (rough estimate)
            data_point['turns'][0]['_input_tokens'] = len(inputs) // 4
            data_point['turns'][0]['_retry_count'] = 0  # Initial turn has no retries
            log_debug(f"Initialized turns with problem statement, turn count: {len(data_point['turns'])}", indent=4)

        except Exception as e:
            log_error(f"Error loading repository: {e}")

            # Visual separator for repository loading error
            log_info("\n" + "=" * 100)
            log_info(f"{'='*15} FAILED SAMPLE: {instance_id} (repository loading error) {'='*15}")
            log_info("=" * 100 + "\n")

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
                # Visual separator for each step
                log_info(f"\n{'─'*80}", indent=4)
                log_info(f"{'─'*20} Step {cur_step + 1}/{total_steps} {'─'*20}", indent=4)
                log_info(f"{'─'*80}\n", indent=4)

                if (
                    'turns' not in data_point
                    or not isinstance(data_point['turns'], list)
                    or len(data_point['turns']) == 0
                ):
                    log_error(f"Invalid turns structure at step {cur_step}: {data_point.get('turns', 'missing')}", indent=8)
                    status = "failed"
                    reason = "invalid_turns_structure"
                    break

                if (
                    hasattr(self.cfg, 'max_seq_length')
                    and self.cfg.max_seq_length is not None
                    and self.cfg.max_seq_length > 0
                ):
                    original_turns_count = len(data_point['turns'])

                    # Use the first-and-recent truncation strategy
                    log_debug("Using first-and-recent truncation strategy", indent=8)

                    # Get truncation preview for logging
                    preview = ContextManager.get_truncation_preview(
                        data_point['turns'], self.cfg.max_seq_length, self.cfg.inference.tokens_to_generate
                    )
                    if "No truncation needed" not in preview:
                        log_debug(f"Truncation preview:\n{preview}", indent=12)

                    # Perform truncation
                    data_point['turns'], truncation_stats = ContextManager.first_and_recent_truncate(
                        data_point['turns'], self.cfg.max_seq_length, self.cfg.inference.tokens_to_generate
                    )

                    if truncation_stats['removed_turns'] > 0:
                        log_info(f"Truncated dialogue from {original_turns_count} to {len(data_point['turns'])} turns", indent=12)
                        log_info(f"Truncation stats: {truncation_stats}", indent=12)

                prepared_data_point = copy.deepcopy(data_point)
                
                # Track which turns are included in the context for this generation
                context_turn_ids = [turn.get('turn_id', i) for i, turn in enumerate(prepared_data_point['turns'])]
                log_debug(f"Context includes turn IDs: {context_turn_ids}", indent=8)

                if self.cfg.enable_loop_detection and len(chat_history) >= self.cfg.loop_detection_threshold - 1:
                    is_loop, loop_info = ContextManager.detect_repetitive_tool_calls(
                        chat_history, self.cfg.loop_detection_threshold - 1
                    )

                    if is_loop:
                        log_warning(
                            f"Potential loop detected before generation! Previous {loop_info['total_repetitions']} calls were identical",
                            indent=12
                        )
                        prepared_data_point['turns'] = ContextManager.inject_loop_intervention(
                            prepared_data_point['turns'], loop_info
                        )
                        # Sync back to original data_point to persist the intervention
                        data_point['turns'] = copy.deepcopy(prepared_data_point['turns'])
                        # Update turn_ids for any new turns
                        for i, turn in enumerate(data_point['turns']):
                            turn['turn_id'] = i
                        log_debug(f"Added loop intervention as turn {len(data_point['turns'])-1}", indent=12)

                if getattr(self.cfg, 'enable_enhanced_context', True):
                    will_fit, error_msg, context_stats = ContextManager.check_context_before_generation(prepared_data_point, self.cfg)
                    if not will_fit:
                        log_error(f"Context length check failed: {error_msg}", indent=12)
                        log_error(f"Context stats: {context_stats}", indent=12)
                        status = "failed"
                        reason = "context_length_exceeded_proactive"
                        break

                if ContextManager.should_inject_final_turn(
                    cur_step,
                    total_steps,
                    status,
                    enable_final_turn_prompt=getattr(self.cfg, 'enable_final_turn_prompt', True),
                ):
                    log_info(f"Injecting final turn instruction at step {cur_step + 1}/{total_steps}", indent=8)
                    prepared_data_point['turns'] = ContextManager.inject_final_turn_instruction(
                        prepared_data_point['turns'], is_final_turn=True
                    )
                    # Sync back to original data_point to persist the final turn instruction
                    # Check if the instruction was added to an existing turn or as a new turn
                    if len(prepared_data_point['turns']) > len(data_point['turns']):
                        # New turn was added
                        data_point['turns'] = copy.deepcopy(prepared_data_point['turns'])
                        # Update turn_ids for any new turns
                        for i, turn in enumerate(data_point['turns']):
                            turn['turn_id'] = i
                        log_debug(f"Added final turn instruction as new turn {len(data_point['turns'])-1}", indent=8)
                    else:
                        # Instruction was appended to existing turn
                        data_point['turns'] = copy.deepcopy(prepared_data_point['turns'])
                        log_debug(f"Appended final turn instruction to existing turn", indent=8)
                    # Update context_turn_ids to include any modified turns
                    context_turn_ids = [turn.get('turn_id', i) for i, turn in enumerate(prepared_data_point['turns'])]
                    log_debug(f"Updated context after final turn instruction: {context_turn_ids}", indent=8)

                response_type = 'normal'
                if cur_step == total_steps - 1:
                    response_type = 'final_turn'

                safe_generation_limit = None
                if self.cfg.enable_response_length_management:
                    if self.cfg.max_seq_length:

                        current_tokens = ContextManager.count_dialogue_tokens(prepared_data_point['turns'])
                        safe_generation_limit = ContextManager.calculate_safe_token_limit(
                            current_tokens,
                            self.cfg.max_seq_length,
                            self.cfg.context_safety_margin,
                            max_generation_tokens=self.cfg.inference.tokens_to_generate,
                        )
                        log_debug(f"Safe generation limit: {safe_generation_limit} tokens", indent=12)

                retry_count = 0
                while retry_count <= self.cfg.max_retries:
                    try:
                        log_info(
                            f"Sending {len(prepared_data_point['turns'])} turns to LLM (attempt {retry_count + 1})",
                            indent=12
                        )

                        llm_output = await super().process_single_datapoint(prepared_data_point, all_data)
                        
                        # Store context information in the output
                        llm_output['_context_turn_ids'] = context_turn_ids

                        # Get the actual number of generated tokens
                        actual_generated_tokens = llm_output.get('num_generated_tokens', 0)

                        # Get the full generation for failure analysis
                        full_gen = llm_output.get('_full_generation', llm_output.get('generation', ''))

                        if self.cfg.enable_response_length_management:
                            # Use the single tokens_to_generate config for all response types
                            max_tokens = self.cfg.inference.tokens_to_generate

                            # Check if response is acceptable based on actual token count
                            # If response uses exactly max_tokens, it's likely truncated
                            is_acceptable = actual_generated_tokens < max_tokens
                            warning_msg = ""

                            if actual_generated_tokens > max_tokens:
                                warning_msg = f"Response exceeds token limit: {actual_generated_tokens} > {max_tokens}"
                            elif actual_generated_tokens >= max_tokens:
                                warning_msg = f"Response at token limit: {actual_generated_tokens} = {max_tokens} (likely truncated)"
                            elif actual_generated_tokens > max_tokens * self.cfg.response_critical_threshold:
                                warning_msg = f"Response approaching token limit: {actual_generated_tokens}/{max_tokens} ({(actual_generated_tokens/max_tokens)*100:.1f}%)"
                            elif actual_generated_tokens > max_tokens * self.cfg.response_warning_threshold:
                                warning_msg = f"Response length warning: {actual_generated_tokens}/{max_tokens} ({(actual_generated_tokens/max_tokens)*100:.1f}%)"

                            stats = {
                                'actual_tokens': actual_generated_tokens,
                                'max_tokens': max_tokens,
                                'percentage': (actual_generated_tokens / max_tokens) * 100 if max_tokens > 0 else 0,
                                'response_type': response_type,
                            }

                            if warning_msg:
                                log_warning(f"Response length check: {warning_msg}", indent=16)
                                log_debug(f"Response stats: {stats}", indent=16)

                            if not is_acceptable and retry_count < self.cfg.max_retries:
                                log_error(f"Response too long: {stats['actual_tokens']} tokens", indent=16)

                                failure_analysis = ContextManager.analyze_response_failure(
                                    full_gen,
                                    prepared_data_point['turns'],
                                    self.cfg.max_seq_length or 128000,
                                    actual_total_tokens=actual_generated_tokens,
                                )
                                log_info(f"Failure analysis: {failure_analysis}", indent=16)

                                if self.cfg.inject_length_warnings:
                                    if response_type == 'final_turn':
                                        warning_msg = f"Please provide a shorter, more focused answer with concise findings and locations. Include only the most essential bullet points in <findings> without excessive explanation, and directly state the bug locations in <locations>."

                                    else:
                                        warning_msg = f"Please be more concise: reduce your thinking/reasoning to only the most essential analysis steps. Skip redundant explanations and focus on the critical path to finding the bug."

                                    prepared_data_point['turns'] = ContextManager.inject_length_warning(
                                        prepared_data_point['turns'], warning_msg
                                    )
                                    # Sync back to original data_point to persist the warning
                                    data_point['turns'] = copy.deepcopy(prepared_data_point['turns'])
                                    # Update turn_ids for any new turns
                                    for i, turn in enumerate(data_point['turns']):
                                        turn['turn_id'] = i
                                    # Update context_turn_ids to include the new warning turn
                                    context_turn_ids = [turn.get('turn_id', i) for i, turn in enumerate(prepared_data_point['turns'])]
                                    log_debug(f"Added length warning as turn {len(data_point['turns'])-1}", indent=20)

                                retry_count += 1
                                continue

                        break

                    except openai.BadRequestError as e:
                        if 'Please reduce the length of the messages or completion' in str(
                            e
                        ) or 'is longer than the model\'s context length' in str(e):
                            log_warning(
                                "Artsiv generation failed due to running out of context. "
                                "Failing for subsequent subtasks automatically.",
                                indent=16
                            )
                            status = "failed"
                            reason = "context_length_exceeded"
                            break
                        log_warning(f"Artsiv generation failed with BadRequestError: {e}", indent=12)
                        status = "failed"
                        reason = f"bad_request_error: {str(e)}"
                        break

                # Use the actual_generated_tokens we already retrieved
                total_generated_tokens += actual_generated_tokens

                if actual_generated_tokens >= self.cfg.inference.tokens_to_generate:
                    log_warning(
                        f"Model generated {actual_generated_tokens} tokens (configured limit: {self.cfg.inference.tokens_to_generate}). "
                        f"Response was likely truncated. Consider the response incomplete.",
                        indent=8
                    )
                    llm_output['_likely_truncated'] = True

                chat_history.append(llm_output)

                if self.cfg.enable_loop_detection and len(chat_history) >= self.cfg.loop_detection_threshold:
                    is_loop, loop_info = ContextManager.detect_repetitive_tool_calls(chat_history, self.cfg.loop_detection_threshold)

                    if is_loop:
                        log_warning(
                            f"Loop detected! Agent has repeated the same tool call {loop_info['total_repetitions']} times",
                            indent=12
                        )
                        log_debug(f"Loop details: {loop_info}", indent=12)

                        data_point['turns'] = ContextManager.inject_loop_intervention(data_point['turns'], loop_info)
                        # Update turn_ids for any new turns
                        for i, turn in enumerate(data_point['turns']):
                            turn['turn_id'] = i
                        log_debug(f"Added post-generation loop intervention as turn {len(data_point['turns'])-1}", indent=12)

                        loop_warning = {'_loop_detected': True, '_loop_info': loop_info, '_intervention_added': True}
                        chat_history[-1].update(loop_warning)

                        pattern_analysis = ContextManager.analyze_loop_patterns(chat_history)
                        log_debug(f"Pattern analysis: {pattern_analysis}", indent=12)

                if self.cfg.remove_thinking:
                    remove_thinking(llm_output, 'generation', self.cfg.thinking_begin, self.cfg.thinking_end)

                try:
                    extracted_block = DialogProcessor.extract_response(llm_output['generation'], self.cfg)
                except Exception as e:
                    log_error(f"Error extracting response from LLM output: {e}", indent=8)
                    log_debug(f"LLM output was: {llm_output.get('generation', 'None')[:500]}...", indent=8)
                    status = "failed"
                    reason = f"response_extraction_error: {str(e)}"
                    break

                if not extracted_block:
                    log_warning("Model failed to generate a tool use or location. Ending generation.", indent=8)
                    if llm_output.get('_likely_truncated', False):
                        status = "failed"
                        reason = "response_truncated_at_token_limit"
                        log_error(
                            f"Response was truncated at token limit ({actual_generated_tokens} tokens) "
                            f"and no valid tool/location was extracted. The model needs more tokens "
                            f"to complete its response, but a buffer should have been reserved.",
                            indent=12
                        )
                    else:
                        status = "failed"
                        reason = "no_tool_or_location_generated"
                    break

                try:
                    if data_point['turns'] and len(data_point['turns']) > 0:
                        current_turn = data_point['turns'][-1]
                        if isinstance(current_turn, dict):
                            current_turn['assistant'] = llm_output['generation']
                            current_turn['assistant_raw'] = llm_output.get('raw_generation', llm_output['generation'])
                            current_turn['assistant_raw_w_think'] = llm_output.get(
                                '_full_generation', llm_output['generation']
                            )
                            current_turn['_llm_tokens'] = actual_generated_tokens  # Store actual LLM token count
                            current_turn['_context_turn_ids'] = context_turn_ids  # Store which turns were in context
                            current_turn['_retry_count'] = retry_count  # Track number of retries for this turn

                            if extracted_block:
                                if extracted_block.get("type") == "tool_calls":
                                    current_turn['tool_call'] = extracted_block.get("tool_call", None)
                                elif extracted_block.get("type") == "locations":
                                    current_turn['locations'] = extracted_block.get("locations", [])
                        else:
                            log_error(f"Current turn is not a dict: {type(current_turn)}", indent=16)
                            status = "failed"
                            reason = "invalid_turn_structure"
                            break
                    else:
                        log_error("No turns available to add assistant response", indent=12)
                        status = "failed"
                        reason = "no_turns_available"
                        break
                except Exception as e:
                    log_error(f"Error adding assistant response to turn: {e}", indent=8)
                    status = "failed"
                    reason = f"turn_update_error: {str(e)}"
                    break

                if extracted_block.get("type") == "tool_calls":
                    if "tool_call" not in extracted_block:
                        log_error(f"Missing 'tool_call' in extracted block: {extracted_block}", indent=12)
                        status = "failed"
                        reason = "missing_tool_call_in_extracted_block"
                        break

                    # Visual separator for tool execution
                    tool_name = extracted_block.get("tool_call", {}).get("tool", "unknown")
                    log_info(f"\n{'▸'*60}", indent=8)
                    log_info(f"{'▸'*10} Executing Tool: {tool_name} {'▸'*10}", indent=8)
                    log_info(f"{'▸'*60}", indent=8)

                    tool_output_content, tool_output_tokens = self.tool_executor.execute_tool(
                        extracted_block["tool_call"], repo_dict
                    )

                    tool_output_to_store = tool_output_content

                    # Log tool output size (tool executor already counted tokens using tiktoken)
                    if self.cfg.enable_response_length_management:
                        log_info(f"Tool output size: {tool_output_tokens} tokens, {len(tool_output_content)} chars", indent=12)
                        log_info(f"{'▸'*60}\n", indent=12)

                    if data_point['turns'] and len(data_point['turns']) > 0:
                        current_turn = data_point['turns'][-1]
                        if isinstance(current_turn, dict):
                            current_turn['tool_output'] = tool_output_to_store
                            current_turn['_tool_tokens'] = tool_output_tokens  # Store actual token count
                            log_debug(f"Added tool output to current turn {len(data_point['turns'])-1}", indent=16)

                            new_turn = {
                                "turn_id": len(data_point['turns']),
                                "inputs": tool_output_to_store,
                                "assistant": "",
                                "tool_call": None,
                                "tool_output": "",
                                "_retry_count": 0,  # New turns start with 0 retries
                            }
                            data_point['turns'].append(new_turn)
                            log_debug(f"Added new turn for next iteration, total turns: {len(data_point['turns'])}, turn_id: {new_turn['turn_id']}", indent=16)
                        else:
                            log_error(f"Current turn is not a dict: {type(current_turn)}", indent=16)
                            status = "failed"
                            reason = "invalid_turn_structure_for_tool_output"
                            break
                    else:
                        log_error("No turns available to add tool output", indent=12)
                        status = "failed"
                        reason = "no_turns_for_tool_output"
                        break
                elif extracted_block.get("type") == "locations":
                    if "locations" not in extracted_block:
                        log_error(f"Missing 'locations' in extracted block: {extracted_block}", indent=12)
                        status = "failed"
                        reason = "missing_locations_in_extracted_block"
                        break
                    data_point["locations"] = extracted_block["locations"]
                    status = "success"
                    reason = None
                    break

                if data_point.get('turns') and len(data_point['turns']) > 0:
                    last_turn = data_point['turns'][-1]
                    has_assistant = isinstance(last_turn, dict) and 'assistant' in last_turn
                    log_debug(
                        f"Current turn count: {len(data_point['turns'])}, last turn has assistant: {has_assistant}",
                        indent=8
                    )
                else:
                    log_debug("No turns available to check for assistant field", indent=8)

                if cur_step == total_steps - 1 and status is None:
                    status = "failed"
                    reason = "max_steps_exceeded"
                    break

            if status is None:
                status = "failed"
                if reason is None:
                    reason = "unknown_failure"
        except Exception as e:
            # Visual separator for error
            log_error("\n" + "!" * 100)
            log_error(f"{'!'*15} ERROR IN SAMPLE: {instance_id} {'!'*15}")
            log_error("!" * 100)

            log_error(f"Unexpected error in process_single_datapoint: {e}")
            import traceback

            full_traceback = traceback.format_exc()
            log_error(f"Full traceback:\n{full_traceback}")

            print(f"\n{'!'*100}")
            print(f"{'!'*15} ERROR IN SAMPLE: {instance_id} {'!'*15}")
            print(f"{'!'*100}")
            print(f"ERROR in process_single_datapoint: {e}")
            print(f"Error type: {type(e).__name__}")
            print(f"Full traceback:\n{full_traceback}")

            if isinstance(data_point, dict):
                log_error(f"data_point keys: {list(data_point.keys())}", indent=4)
                print(f"data_point keys: {list(data_point.keys())}")

                if 'turns' in data_point:
                    log_error(f"Number of turns: {len(data_point['turns'])}", indent=8)
                    print(f"Number of turns: {len(data_point['turns'])}")

                    for i, turn in enumerate(data_point['turns'][:5]):
                        if isinstance(turn, dict):
                            log_error(f"Turn {i} keys: {list(turn.keys())}", indent=12)
                            log_error(f"Turn {i} has 'assistant': {'assistant' in turn}", indent=12)
                            print(f"Turn {i} keys: {list(turn.keys())}")
                            print(f"  - has 'assistant': {'assistant' in turn}")
                            print(f"  - has 'inputs': {'inputs' in turn}")
                            print(f"  - has 'tool_call': {'tool_call' in turn}")
                            print(f"  - has 'tool_output': {'tool_output' in turn}")
                        else:
                            log_error(f"Turn {i} is not a dict: {type(turn)}", indent=12)
                            print(f"Turn {i} is not a dict: {type(turn)}, value: {turn}")
                else:
                    log_error("No 'turns' key in data_point", indent=8)
                    print("No 'turns' key in data_point")
            else:
                log_error(f"data_point is not a dict: {type(data_point)}", indent=4)
                print(f"data_point is not a dict: {type(data_point)}")

            print(f"{'='*60}\n")
            log_error("=== END DEBUG STATE ===")

            status = "failed"
            reason = f"exception: {str(e)}"

        if 'turns' not in data_point:
            log_warning("Missing 'turns' in data_point at return time, initializing empty structure")
            data_point['turns'] = []

        for i, turn in enumerate(data_point.get('turns', [])):
            if not isinstance(turn, dict):
                log_error(f"Turn {i} is not a dictionary: {type(turn)}", indent=4)
                data_point['turns'][i] = {
                    "turn_id": i,
                    "inputs": str(turn) if turn else "",
                    "assistant": "",
                    "tool_call": None,
                    "tool_output": "",
                    "_retry_count": 0,
                }
            else:
                if 'inputs' not in turn:
                    turn['inputs'] = ''
                if 'assistant' not in turn:
                    turn['assistant'] = ''
                if 'tool_call' not in turn:
                    turn['tool_call'] = None
                if 'tool_output' not in turn:
                    turn['tool_output'] = ''
                if 'turn_id' not in turn:
                    turn['turn_id'] = i

                log_debug(
                    f"Final turn {i} validation - has assistant: {'assistant' in turn}, keys: {list(turn.keys())}",
                    indent=4
                )

        if 'turns' in data_point:
            log_debug(f"Returning {len(data_point['turns'])} turns")

        ground_truth_in_repo_percentage = data_point.get('_ground_truth_in_repo_percentage', 0.0)
        missing_ground_truth_files = data_point.get('_missing_ground_truth_files', [])

        data_point.pop('_ground_truth_in_repo_percentage', None)
        data_point.pop('_missing_ground_truth_files', None)

        # Visual separator for end of sample processing
        log_info("\n" + "=" * 100)
        log_info(f"{'='*15} COMPLETED SAMPLE: {instance_id} (Status: {status}) {'='*15}")
        log_info("=" * 100 + "\n")

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
