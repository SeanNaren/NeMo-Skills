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

from nemo_skills.inference.eval.locagent_utils.utils import (
    calculate_ground_truth_percentage,
    extract_locations_from_patch,
    filter_repo_dict,
    tree_repo_dict,
)
from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, remove_thinking, setup_logging

# Import bookend truncation strategies (optional, backwards compatible)
try:
    from nemo_skills.inference.eval.locagent_utils.bookend_truncation import (
        bookend_truncate_dialogue_history,
        smart_bookend_truncate,
    )

    BOOKEND_TRUNCATION_AVAILABLE = True
except ImportError:
    BOOKEND_TRUNCATION_AVAILABLE = False

# Import loop detection utilities (optional, backwards compatible)
try:
    from nemo_skills.inference.eval.locagent_utils.loop_detection import (
        analyze_loop_patterns,
        detect_repetitive_tool_calls,
        inject_loop_intervention,
        prevent_loop_generation,
    )

    LOOP_DETECTION_AVAILABLE = True
except ImportError:
    LOOP_DETECTION_AVAILABLE = False

# Import enhanced context management (optional, backwards compatible)
try:
    from nemo_skills.inference.eval.locagent_utils.enhanced_context_management import (
        TokenCounter,
        check_context_before_generation,
        enhanced_truncate_dialogue,
    )

    ENHANCED_CONTEXT_AVAILABLE = True
except ImportError:
    ENHANCED_CONTEXT_AVAILABLE = False

# Import final turn prompt injection (optional, backwards compatible)
try:
    from nemo_skills.inference.eval.locagent_utils.final_turn_prompt import (
        inject_final_turn_instruction,
        should_inject_final_turn,
    )

    FINAL_TURN_PROMPT_AVAILABLE = True
except ImportError:
    FINAL_TURN_PROMPT_AVAILABLE = False

PROMPT_TEMPLATE_VERSION: str = "v4"


module_base = f"nemo_skills.inference.eval.locagent_utils.{PROMPT_TEMPLATE_VERSION}"

dialog_processor = importlib.import_module(f"{module_base}.dialog_processor")
tool_executor = importlib.import_module(f"{module_base}.tool_executor")

DialogProcessor = dialog_processor.DialogProcessor
ToolExecutor = tool_executor.ToolExecutor
# Import only the token management functions we need from dialog_processor
truncate_dialogue_history = dialog_processor.truncate_dialogue_history

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class LocalAgentGenerationConfig(GenerateSolutionsConfig):
    # Core inference settings
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters
    server: dict = field(default_factory=dict)  # Server configuration for model hosting

    # Agent behavior settings
    mount_directory: str = "/repos/"  # Directory where repositories are mounted
    remove_thinking: bool = True  # Whether to strip thinking tags from output
    total_steps: int = 20  # Maximum number of agent steps per problem

    # Repository filtering settings
    file_extensions: list = field(default_factory=lambda: ["py", "cfg"])  # File types to include in repo
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
            # "utils",  # Removed - too many legitimate utility files
            # "migrations",  # Removed - Django migrations contain bug fixes
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
            # "cache", - Removed - django/core/cache/backends/filebased.py from swe-bench lite
            "vendor",
            # "lib",  # Removed - matplotlib's main source directory!
            "libs",
            "dependencies",
            # "config",  # Removed - configuration files often have bugs
            # "conf",    # Removed - configuration files often have bugs
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

    max_seq_length: int | None = 120000  # Maximum context length in tokens (set via CLI)

    # Display settings
    show_line_counts: bool = False  # Show file line counts in repository tree output
    max_view_lines: int = 1000  # Maximum lines to show in view tool (0 = no limit) - reduced from 1000 to 300

    # Truncation strategy settings
    truncation_strategy: str = "bookend"  # Options: "sequential" (default), "bookend", "smart_bookend", "enhanced"

    # Loop detection settings
    enable_loop_detection: bool = True  # Enable detection and prevention of repetitive tool calls
    loop_detection_threshold: int = 3  # Number of identical calls to trigger loop detection

    # Enhanced context management settings
    enable_enhanced_context: bool = True  # Use enhanced context management with better token counting
    context_safety_margin: float = 0.9  # Use only this fraction of max context (0.9 = 90%)
    use_tiktoken: bool = True  # Use tiktoken for accurate token counting if available

    # Final turn prompt settings
    enable_final_turn_prompt: bool = True  # Inject instruction on final turn to force location prediction
    final_turn_instruction_type: str = "aligned"  # Type of instruction: aligned, standard, urgent, gentle, detailed
    final_turn_threshold: float = 1.0  # When to trigger (1.0 = only last turn, 0.8 = last 20% of turns)

    # Summarization settings (currently disabled, preserved for future use)
    enable_turn_summarization: bool = False  # Enable context summarization to reduce token usage
    max_summary_sentences: int = 5  # Maximum sentences in the investigation summary
    min_turns_for_summarization: int = 10  # Minimum turns before summarization kicks in
    summarization_model: bool = False  # Whether to use LLM for summarization


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_locagent_generation_config", node=LocalAgentGenerationConfig)


class LocAgentGenerationTask(GenerationTask):
    def __init__(self, cfg: LocalAgentGenerationConfig):
        super().__init__(cfg)
        self.tool_executor = ToolExecutor(cfg)

        # Log truncation strategy info
        if not BOOKEND_TRUNCATION_AVAILABLE and cfg.truncation_strategy in ['bookend', 'smart_bookend']:
            LOG.warning(f"Bookend truncation module not available. Falling back to sequential truncation.")
        else:
            LOG.info(f"Using truncation strategy: {cfg.truncation_strategy}")

    def log_example_prompt(self, data):
        return

    async def process_single_datapoint(self, data_point, all_data):
        """Will do all necessary generations to get a single answer for the data point."""

        # Log initial state of data_point for debugging
        LOG.debug(
            f"Initial data_point keys: {list(data_point.keys()) if isinstance(data_point, dict) else 'not a dict'}"
        )
        if 'turns' in data_point:
            LOG.debug(f"Initial turns structure: {data_point['turns']}")

        # Filter out samples with empty problem statements
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
                'turns': [],  # Include empty turns array for consistency
            }

        total_steps = self.cfg.total_steps
        chat_history = []
        total_generated_tokens = 0

        # Initialize or fix turns structure early to ensure it always exists with proper fields
        if 'turns' in data_point and isinstance(data_point['turns'], list) and len(data_point['turns']) > 0:
            # Ensure existing turns have all required fields
            for i, turn in enumerate(data_point['turns']):
                if isinstance(turn, dict):
                    # Add missing fields with default values
                    turn.setdefault('inputs', '')
                    turn.setdefault('assistant', '')
                    turn.setdefault('tool_call', None)
                    turn.setdefault('tool_output', '')
                    LOG.debug(
                        f"Turn {i} after setdefault - keys: {list(turn.keys())}, has assistant: {'assistant' in turn}"
                    )
                else:
                    # Replace non-dict turn with proper structure
                    LOG.warning(f"Found non-dict turn at index {i}: {type(turn)}, replacing with empty structure")
                    data_point['turns'][i] = {"inputs": "", "assistant": "", "tool_call": None, "tool_output": ""}
        else:
            # Initialize new turns structure
            data_point['turns'] = [{"inputs": "", "assistant": "", "tool_call": None, "tool_output": ""}]

        try:
            instance_filepath = Path(self.cfg.mount_directory).joinpath(f"{data_point['instance_id']}.pkl")

            # repo_dict is dict with 'structure' containing the actual repo tree dict_keys(['repo', 'base_commit',
            # 'structure', 'instance_id'])
            with open(instance_filepath, 'rb') as f:
                repo_dict = pickle.load(f)
            repo_dict = filter_repo_dict(repo_dict, self.cfg.exclude_dirs, self.cfg.file_extensions)
            tree_structure = tree_repo_dict(repo_dict, self.cfg.show_line_counts)

            # Calculate ground truth files percentage using utility function
            ground_truth_in_repo_percentage = 0.0
            if 'patch' in data_point and data_point['patch']:
                try:
                    # Extract file paths from patch
                    locations = extract_locations_from_patch(data_point['patch'])

                    # Use utility function to calculate percentage
                    ground_truth_in_repo_percentage, debug_info = calculate_ground_truth_percentage(
                        repo_dict, locations, self.cfg.exclude_dirs, self.cfg.file_extensions
                    )

                    # Log debug info if needed
                    LOG.debug(f"Ground truth check debug info: {debug_info}")

                    # Store missing files info for aggregation
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

            # Update the first turn with actual content
            data_point['turns'][0]['inputs'] = inputs
            LOG.debug(f"Initialized turns with problem statement, turn count: {len(data_point['turns'])}")

        except Exception as e:
            LOG.error(f"Error loading repository for instance {data_point.get('instance_id', 'unknown')}: {e}")
            # Return early with error status
            return {
                'generation': [],
                'total_generated_tokens': 0,
                'num_turns': 0,
                'status': 'failed',
                'reason': f'repository_loading_error: {str(e)}',
                'turns': data_point['turns'],  # Will have the empty structure
            }

        reason = None
        status = None
        try:
            for cur_step in range(total_steps):
                # Validate turns structure at the beginning of each iteration
                if (
                    'turns' not in data_point
                    or not isinstance(data_point['turns'], list)
                    or len(data_point['turns']) == 0
                ):
                    LOG.error(f"Invalid turns structure at step {cur_step}: {data_point.get('turns', 'missing')}")
                    status = "failed"
                    reason = "invalid_turns_structure"
                    break

                # Check and truncate dialogue history if needed before making the LLM call
                if (
                    hasattr(self.cfg, 'max_seq_length')
                    and self.cfg.max_seq_length is not None
                    and self.cfg.max_seq_length > 0
                ):
                    original_turns_count = len(data_point['turns'])

                    # Apply selected truncation strategy
                    truncation_strategy = getattr(self.cfg, 'truncation_strategy', 'sequential')

                    # Use enhanced context management if available and enabled
                    if (
                        ENHANCED_CONTEXT_AVAILABLE
                        and getattr(self.cfg, 'enable_enhanced_context', True)
                        and (truncation_strategy == 'enhanced' or getattr(self.cfg, 'use_tiktoken', True))
                    ):
                        LOG.debug(f"Using enhanced context management")
                        # Initialize token counter if not already done
                        if not hasattr(self, '_token_counter'):
                            self._token_counter = TokenCounter(getattr(self.cfg, 'model', 'gpt-4'))

                        # Use enhanced truncation
                        data_point['turns'], truncation_stats = enhanced_truncate_dialogue(
                            data_point['turns'],
                            self.cfg.max_seq_length,
                            self.cfg.inference.tokens_to_generate,
                            safety_margin=getattr(self.cfg, 'context_safety_margin', 0.9),
                            token_counter=self._token_counter,
                        )
                        LOG.info(f"Enhanced truncation stats: {truncation_stats}")

                    elif truncation_strategy == 'bookend' and BOOKEND_TRUNCATION_AVAILABLE:
                        LOG.debug(f"Using bookend truncation strategy")
                        data_point['turns'] = bookend_truncate_dialogue_history(
                            data_point['turns'], self.cfg.max_seq_length, self.cfg.inference.tokens_to_generate
                        )
                    elif truncation_strategy == 'smart_bookend' and BOOKEND_TRUNCATION_AVAILABLE:
                        LOG.debug(f"Using smart bookend truncation strategy")
                        data_point['turns'] = smart_bookend_truncate(
                            data_point['turns'], self.cfg.max_seq_length, self.cfg.inference.tokens_to_generate
                        )
                    else:
                        # Default to sequential truncation (backwards compatible)
                        if truncation_strategy != 'sequential' and not BOOKEND_TRUNCATION_AVAILABLE:
                            LOG.warning(f"Truncation strategy '{truncation_strategy}' not available, using sequential")
                        LOG.debug(f"Using sequential truncation strategy")
                        data_point['turns'] = truncate_dialogue_history(
                            data_point['turns'], self.cfg.max_seq_length, self.cfg.inference.tokens_to_generate
                        )

                    if len(data_point['turns']) < original_turns_count:
                        LOG.info(
                            f"Truncated dialogue from {original_turns_count} to {len(data_point['turns'])} turns using {truncation_strategy} strategy"
                        )

                # Use original data_point for LLM call
                prepared_data_point = copy.deepcopy(data_point)

                # Loop prevention - check if we should modify the prompt to prevent repetition
                if (
                    LOOP_DETECTION_AVAILABLE
                    and self.cfg.enable_loop_detection
                    and len(chat_history) >= self.cfg.loop_detection_threshold - 1
                ):
                    # Check for loops in existing history before generating
                    is_loop, loop_info = detect_repetitive_tool_calls(
                        chat_history, self.cfg.loop_detection_threshold - 1
                    )

                    if is_loop:
                        LOG.warning(
                            f"Potential loop detected before generation! Previous {loop_info['total_repetitions']} calls were identical"
                        )
                        # Inject intervention message to prevent loop continuation
                        prepared_data_point['turns'] = inject_loop_intervention(
                            prepared_data_point['turns'], loop_info
                        )

                # Proactive context length check before making LLM call
                if ENHANCED_CONTEXT_AVAILABLE and getattr(self.cfg, 'enable_enhanced_context', True):
                    will_fit, error_msg, context_stats = check_context_before_generation(
                        prepared_data_point, self.cfg, getattr(self, '_token_counter', None)
                    )
                    if not will_fit:
                        LOG.error(f"Context length check failed: {error_msg}")
                        LOG.error(f"Context stats: {context_stats}")
                        status = "failed"
                        reason = "context_length_exceeded_proactive"
                        break

                # Check if we should inject final turn instruction
                if FINAL_TURN_PROMPT_AVAILABLE and should_inject_final_turn(
                    cur_step,
                    total_steps,
                    status,
                    enable_final_turn_prompt=getattr(self.cfg, 'enable_final_turn_prompt', True),
                    final_turn_threshold=getattr(self.cfg, 'final_turn_threshold', 1.0),
                ):
                    LOG.info(f"Injecting final turn instruction at step {cur_step + 1}/{total_steps}")
                    prepared_data_point['turns'] = inject_final_turn_instruction(
                        prepared_data_point['turns'],
                        is_final_turn=True,
                        instruction_type=getattr(self.cfg, 'final_turn_instruction_type', 'standard'),
                    )

                try:
                    LOG.info(f"Sending {len(prepared_data_point['turns'])} turns to LLM")
                    llm_output = await super().process_single_datapoint(prepared_data_point, all_data)
                # TODO: this is a hack (as not all servers return that),
                # but eventually we should support handling errors like this globally for all generations
                except openai.BadRequestError as e:
                    if 'Please reduce the length of the messages or completion' in str(e):
                        LOG.warning(
                            "LocAgent generation failed due to running out of context. "
                            "Failing for subsequent subtasks automatically.",
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

                # Loop detection - check if agent is stuck in a repetitive pattern
                if (
                    LOOP_DETECTION_AVAILABLE
                    and self.cfg.enable_loop_detection
                    and len(chat_history) >= self.cfg.loop_detection_threshold
                ):
                    is_loop, loop_info = detect_repetitive_tool_calls(chat_history, self.cfg.loop_detection_threshold)

                    if is_loop:
                        LOG.warning(
                            f"Loop detected! Agent has repeated the same tool call {loop_info['total_repetitions']} times"
                        )
                        LOG.debug(f"Loop details: {loop_info}")

                        # Inject intervention to help break the loop
                        data_point['turns'] = inject_loop_intervention(data_point['turns'], loop_info)

                        # Also add a warning to the generation for visibility
                        loop_warning = {'_loop_detected': True, '_loop_info': loop_info, '_intervention_added': True}
                        chat_history[-1].update(loop_warning)

                        # Analyze patterns for debugging
                        pattern_analysis = analyze_loop_patterns(chat_history)
                        LOG.debug(f"Pattern analysis: {pattern_analysis}")

                if self.cfg.remove_thinking:
                    remove_thinking(llm_output, 'generation', self.cfg.thinking_begin, self.cfg.thinking_end)

                # Try to extract response with error handling
                try:
                    extracted_block = DialogProcessor.extract_response(llm_output['generation'], self.cfg)
                except Exception as e:
                    LOG.error(f"Error extracting response from LLM output: {e}")
                    LOG.debug(
                        f"LLM output was: {llm_output.get('generation', 'None')[:500]}..."
                    )  # Log first 500 chars
                    status = "failed"
                    reason = f"response_extraction_error: {str(e)}"
                    break

                if not extracted_block:
                    LOG.warning("Model failed to generate a tool use or location. Ending generation.")
                    # todo (hov): add resampling with different temperature if necessary.
                    status = "failed"
                    reason = "no_tool_or_location_generated"
                    break

                # Safely add assistant response to the current turn
                try:
                    if data_point['turns'] and len(data_point['turns']) > 0:
                        current_turn = data_point['turns'][-1]
                        if isinstance(current_turn, dict):
                            # Store raw LLM generation
                            current_turn['assistant'] = llm_output['generation']
                            current_turn['assistant_raw'] = llm_output.get('raw_generation', llm_output['generation'])
                            current_turn['assistant_raw_w_think'] = llm_output.get(
                                '_full_generation', llm_output['generation']
                            )

                            # Store extracted structured data
                            if extracted_block:
                                if extracted_block.get("type") == "tool_calls":
                                    current_turn['tool_call'] = extracted_block.get("tool_call", None)
                                elif extracted_block.get("type") == "locations":
                                    current_turn['locations'] = extracted_block.get("locations", [])
                        else:
                            LOG.error(f"Current turn is not a dict: {type(current_turn)}")
                            status = "failed"
                            reason = "invalid_turn_structure"
                            break
                    else:
                        LOG.error("No turns available to add assistant response")
                        status = "failed"
                        reason = "no_turns_available"
                        break
                except Exception as e:
                    LOG.error(f"Error adding assistant response to turn: {e}")
                    status = "failed"
                    reason = f"turn_update_error: {str(e)}"
                    break
                if extracted_block.get("type") == "tool_calls":
                    if "tool_call" not in extracted_block:
                        LOG.error(f"Missing 'tool_call' in extracted block: {extracted_block}")
                        status = "failed"
                        reason = "missing_tool_call_in_extracted_block"
                        break
                    tool_call_result = self.tool_executor.execute_tool(extracted_block["tool_call"], repo_dict)

                    tool_output_to_store = tool_call_result

                    # CRITICAL FIX: Add tool output to the CURRENT turn, not a new one
                    # This maintains the association between tool_call and tool_output
                    if data_point['turns'] and len(data_point['turns']) > 0:
                        current_turn = data_point['turns'][-1]
                        if isinstance(current_turn, dict):
                            current_turn['tool_output'] = tool_output_to_store
                            LOG.debug(f"Added tool output to current turn {len(data_point['turns'])-1}")

                            # Now create a new turn for the next iteration
                            # The new turn has the tool output as input for the assistant to analyze
                            new_turn = {
                                "inputs": tool_output_to_store,
                                "assistant": "",
                                "tool_call": None,
                                "tool_output": "",
                            }
                            data_point['turns'].append(new_turn)
                            LOG.debug(f"Added new turn for next iteration, total turns: {len(data_point['turns'])}")
                        else:
                            LOG.error(f"Current turn is not a dict: {type(current_turn)}")
                            status = "failed"
                            reason = "invalid_turn_structure_for_tool_output"
                            break
                    else:
                        LOG.error("No turns available to add tool output")
                        status = "failed"
                        reason = "no_turns_for_tool_output"
                        break
                elif extracted_block.get("type") == "locations":
                    if "locations" not in extracted_block:
                        LOG.error(f"Missing 'locations' in extracted block: {extracted_block}")
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
                    LOG.debug(
                        f"Current turn count: {len(data_point['turns'])}, last turn has assistant: {has_assistant}"
                    )
                else:
                    LOG.debug("No turns available to check for assistant field")

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
            import traceback

            full_traceback = traceback.format_exc()
            LOG.error(f"Full traceback:\n{full_traceback}")

            # Debug the state when error occurs
            LOG.error("=== DEBUG STATE AT ERROR ===")
            LOG.error(f"Error type: {type(e).__name__}")
            LOG.error(f"Error message: {str(e)}")

            # Also print for immediate visibility
            print(f"\n{'='*60}")
            print(f"ERROR in process_single_datapoint: {e}")
            print(f"Error type: {type(e).__name__}")
            print(f"Full traceback:\n{full_traceback}")

            # Check data_point structure
            if isinstance(data_point, dict):
                LOG.error(f"data_point keys: {list(data_point.keys())}")
                print(f"data_point keys: {list(data_point.keys())}")

                if 'turns' in data_point:
                    LOG.error(f"Number of turns: {len(data_point['turns'])}")
                    print(f"Number of turns: {len(data_point['turns'])}")

                    for i, turn in enumerate(data_point['turns'][:5]):  # Show first 5 turns
                        if isinstance(turn, dict):
                            LOG.error(f"Turn {i} keys: {list(turn.keys())}")
                            LOG.error(f"Turn {i} has 'assistant': {'assistant' in turn}")
                            print(f"Turn {i} keys: {list(turn.keys())}")
                            print(f"  - has 'assistant': {'assistant' in turn}")
                            print(f"  - has 'inputs': {'inputs' in turn}")
                            print(f"  - has 'tool_call': {'tool_call' in turn}")
                            print(f"  - has 'tool_output': {'tool_output' in turn}")
                        else:
                            LOG.error(f"Turn {i} is not a dict: {type(turn)}")
                            print(f"Turn {i} is not a dict: {type(turn)}, value: {turn}")
                else:
                    LOG.error("No 'turns' key in data_point")
                    print("No 'turns' key in data_point")
            else:
                LOG.error(f"data_point is not a dict: {type(data_point)}")
                print(f"data_point is not a dict: {type(data_point)}")

            print(f"{'='*60}\n")
            LOG.error("=== END DEBUG STATE ===")

            status = "failed"
            reason = f"exception: {str(e)}"

        # Ensure turns is always properly structured even in error cases
        if 'turns' not in data_point:
            LOG.warning("Missing 'turns' in data_point at return time, initializing empty structure")
            data_point['turns'] = []

        # Final validation and repair of turn structure
        for i, turn in enumerate(data_point.get('turns', [])):
            if not isinstance(turn, dict):
                LOG.error(f"Turn {i} is not a dictionary: {type(turn)}")
                # Convert to proper structure
                data_point['turns'][i] = {
                    "inputs": str(turn) if turn else "",
                    "assistant": "",
                    "tool_call": None,
                    "tool_output": "",
                }
            else:
                # Ensure all required fields exist with proper defaults
                if 'inputs' not in turn:
                    turn['inputs'] = ''
                if 'assistant' not in turn:
                    turn['assistant'] = ''
                if 'tool_call' not in turn:
                    turn['tool_call'] = None
                if 'tool_output' not in turn:
                    turn['tool_output'] = ''

                LOG.debug(
                    f"Final turn {i} validation - has assistant: {'assistant' in turn}, keys: {list(turn.keys())}"
                )

        # generation is a dict["problem_id.subtask_step": full_solution] here
        # """
        # [
        #     User: Problem statement,
        #     Assistant: {"generation": generation with reasoning trace, "tool_call": … },
        #     User: {"tool_output": ""},
        #     Assistant: {"generation": … "location": …}
        # ]
        # """

        # Debug log the turns structure before returning
        if 'turns' in data_point:
            LOG.debug(f"Returning {len(data_point['turns'])} turns")

        # Get the pre-calculated ground truth percentage and missing files
        ground_truth_in_repo_percentage = data_point.get('_ground_truth_in_repo_percentage', 0.0)
        missing_ground_truth_files = data_point.get('_missing_ground_truth_files', [])

        # Clean up temporary variables from data_point
        data_point.pop('_ground_truth_in_repo_percentage', None)
        data_point.pop('_missing_ground_truth_files', None)

        return {
            'generation': chat_history,
            'total_generated_tokens': total_generated_tokens,
            'num_turns': len(chat_history),
            'status': status,
            'reason': reason,
            'turns': data_point.get('turns', []),  # Include the turns
            'ground_truth_in_repo_percentage': ground_truth_in_repo_percentage,  # Percentage of ground truth files that exist in repo
            'missing_ground_truth_files': missing_ground_truth_files,  # Details about which GT files are missing and why
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
