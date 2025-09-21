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
VARIANT 11: SOTA - State of the Art
HYPOTHESIS: Combine best elements from analysis to exceed 83% precision
Strategy: 
- Base on V2 (82.0%, closest to best)
- Remove all complexity that doesn't add value
- Fine-tune parameters based on successful patterns
- Add subtle optimizations that don't interfere with natural flow
"""

import copy
import importlib
import logging
import pickle
import sys
import re
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

# Import bookend truncation strategies (optional, backwards compatible)
try:
    from nemo_skills.inference.eval.artsiv_utils.bookend_truncation import (
        bookend_truncate_dialogue_history,
        smart_bookend_truncate
    )
    BOOKEND_TRUNCATION_AVAILABLE = True
except ImportError:
    BOOKEND_TRUNCATION_AVAILABLE = False

# Import loop detection utilities (optional, backwards compatible)
try:
    from nemo_skills.inference.eval.artsiv_utils.loop_detection import (
        detect_repetitive_tool_calls,
        inject_loop_intervention,
        prevent_loop_generation,
        analyze_loop_patterns
    )
    LOOP_DETECTION_AVAILABLE = True
except ImportError:
    LOOP_DETECTION_AVAILABLE = False

# Import enhanced context management (optional, backwards compatible)
try:
    from nemo_skills.inference.eval.artsiv_utils.enhanced_context_management import (
        TokenCounter,
        enhanced_truncate_dialogue,
        check_context_before_generation
    )
    ENHANCED_CONTEXT_AVAILABLE = True
except ImportError:
    ENHANCED_CONTEXT_AVAILABLE = False

# Import final turn prompt injection (optional, backwards compatible)
try:
    from nemo_skills.inference.eval.artsiv_utils.final_turn_prompt import (
        inject_final_turn_instruction,
        should_inject_final_turn
    )
    FINAL_TURN_PROMPT_AVAILABLE = True
except ImportError:
    FINAL_TURN_PROMPT_AVAILABLE = False

PROMPT_TEMPLATE_VERSION: str = "v4"

module_base = f"nemo_skills.inference.eval.artsiv_utils.{PROMPT_TEMPLATE_VERSION}"

dialog_processor = importlib.import_module(f"{module_base}.dialog_processor")
tool_executor = importlib.import_module(f"{module_base}.tool_executor")

DialogProcessor = dialog_processor.DialogProcessor
ToolExecutor = tool_executor.ToolExecutor
truncate_dialogue_history = dialog_processor.truncate_dialogue_history

LOG = logging.getLogger(get_logger_name(__file__))


def inject_subtle_focus_hint(inputs: str, step: int, total_steps: int) -> str:
    """Inject very subtle hints at strategic points to maintain focus without disrupting flow."""
    if step == 0:
        # First turn: subtle emphasis on being thorough
        hint = "\n💡 Be thorough in your investigation.\n"
        return inputs + hint
    elif step == total_steps // 2:
        # Midpoint: gentle reminder to stay focused
        hint = "\n🎯 Stay focused on the core issue.\n"
        return inputs + hint
    elif step == total_steps - 3 and total_steps > 10:
        # Near end for long conversations: wrap up hint
        hint = "\n⏱️ Consider wrapping up your investigation.\n"
        return inputs + hint
    return inputs


def optimize_truncation_for_precision(turns, max_seq_length, tokens_to_generate, safety_margin=0.85):
    """Custom truncation that preserves key investigation context."""
    # For SOTA, use slightly more aggressive safety margin to prevent ANY truncation failures
    # This leaves more room for model's natural verbosity while staying safe
    
    if ENHANCED_CONTEXT_AVAILABLE:
        # Use enhanced truncation with custom parameters
        token_counter = TokenCounter('gpt-4')
        optimized_turns, stats = enhanced_truncate_dialogue(
            turns,
            max_seq_length,
            tokens_to_generate,
            safety_margin=safety_margin,  # More aggressive than default 0.9
            token_counter=token_counter
        )
        LOG.info(f"SOTA truncation stats: {stats}")
        return optimized_turns
    else:
        # Fallback to smart bookend if available
        if BOOKEND_TRUNCATION_AVAILABLE:
            return smart_bookend_truncate(turns, max_seq_length, tokens_to_generate)
        else:
            return truncate_dialogue_history(turns, max_seq_length, tokens_to_generate)


@nested_dataclass(kw_only=True)
class ArtsivGenerationConfig(GenerateSolutionsConfig):
    # SOTA configuration based on analysis
    inference: InferenceConfig = field(default_factory=lambda: InferenceConfig(
        temperature=0.7,  # Optimal from best run
        top_k=0,
        top_p=0.95,
        min_p=0.0,
        random_seed=0,
        tokens_to_generate=81920,  # Full context like best run
        repetition_penalty=1.0,
        top_logprobs=None,
        extra_body={}
    ))
    server: dict = field(default_factory=dict)

    # Core settings optimized for precision
    mount_directory: str = "/repos/"
    remove_thinking: bool = True
    total_steps: int = 20

    # Repository filtering - slightly refined based on patterns
    file_extensions: list = field(default_factory=lambda: ["py", "cfg", "yml", "yaml", "toml"])  # Added config formats
    exclude_dirs: list = field(
        default_factory=lambda: [
            # Test directories (most important to exclude)
            "test", "tests", "testing", "test_", "_test", "tests_", "_tests",
            "unittest", "pytest", "nose", "tox",
            # Build/cache directories
            "__pycache__", ".git", ".github", ".gitlab", ".gitignore",
            "build", "dist", "target", "bin", "obj", "out",
            ".pytest_cache", ".tox", ".mypy_cache", ".coverage",
            # Documentation (usually not relevant for bug fixing)
            "docs", "documentation", "doc", "_build", "sphinx",
            # Virtual environments
            "venv", "env", ".env", "virtualenv", ".venv",
            # Dependencies
            "node_modules", "vendor", "vendors", "third_party", "3rdparty",
            "libs", "lib", "dependencies", "packages", "pkg",
            # Examples/demos (often outdated or simplified code)
            "examples", "example", "demo", "demos", "samples", "sample",
            # Deployment/configs
            "deploy", "deployment", "docker", "kubernetes", "k8s", ".docker",
            "ci", "cd", ".ci", ".circleci", ".travis", ".jenkins",
            # Media/static files
            "static", "assets", "media", "images", "img", "css", "js",
            "public", "resources", "res",
            # Data/fixtures
            "data", "datasets", "fixtures", "fixture", "mocks", "mock",
            # Notebooks (not relevant for production bugs)
            "notebooks", "notebook", "jupyter", ".ipynb_checkpoints",
            # Temporary
            "tmp", "temp", "temporary", "cache", ".cache",
            # Logs
            "logs", "log", ".logs",
            # Localization
            "locale", "locales", "translations", "i18n", "l10n",
            # Scripts/tools (often not core functionality)
            "scripts", "script", "tools", "tool", "utils", "util",
            "migrations", "migration",  # DB migrations
            # Settings (often not where bugs are)
            "settings", "setting", "config", "configs", "conf",
            "local_settings", "local",
        ]
    )

    # Tool detection settings - refined for better precision
    enable_implicit_tool_detection: bool = True
    common_words_filter: list = field(
        default_factory=lambda: [
            # Articles, pronouns, prepositions
            "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "can", "this", "that", "these", "those",
            "a", "an", "as", "if", "then", "else", "when", "where", "why", "how", "what", "which",
            "who", "whom", "whose", "it", "its", "they", "them", "their", "we", "our", "you", "your",
            # Common programming terms that are too generic
            "need", "find", "search", "look", "check", "verify", "ensure", "make", "sure",
            "function", "class", "method", "variable", "parameter", "argument", "value",
            "return", "returns", "import", "from", "def", "self", "init", "main",
            "error", "exception", "bug", "issue", "problem", "fix", "patch",
            "file", "line", "code", "source", "implementation", "logic",
            # Common function names that are too generic
            "get", "set", "update", "delete", "create", "save", "load", "run", "execute",
            "process", "handle", "manage", "validate", "parse", "format", "convert",
        ]
    )

    # Context settings - optimized for precision
    max_seq_length: int = 262144
    show_line_counts: bool = False
    max_view_lines: int = 1000

    # SOTA settings based on analysis
    truncation_strategy: str = "enhanced"  # Best for precision
    enable_loop_detection: bool = True
    loop_detection_threshold: int = 3  # Detect loops early
    enable_enhanced_context: bool = True
    context_safety_margin: float = 0.85  # More aggressive to prevent truncation
    use_tiktoken: bool = True
    enable_final_turn_prompt: bool = True
    final_turn_instruction_type: str = "concise"  # New: more concise final prompt
    final_turn_threshold: float = 0.9  # Earlier final turn prompt
    
    # DISABLE all features that hurt performance
    enable_response_length_management: bool = False
    enable_planning_memory: bool = False
    enable_concise_planning: bool = False
    enable_robust_planning: bool = False
    enable_adaptive_planning: bool = False
    
    # NEW SOTA features
    enable_subtle_focus_hints: bool = True  # Very subtle hints that don't disrupt flow
    enable_precision_truncation: bool = True  # Custom truncation for max precision
    enable_early_success_detection: bool = True  # Detect when we've found the solution
    success_confidence_threshold: float = 0.85  # Confidence needed to stop early


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_artsiv_generation_config", node=ArtsivGenerationConfig)


class ArtsivGenerationTask(GenerationTask):
    def __init__(self, cfg: ArtsivGenerationConfig):
        super().__init__(cfg)
        self.tool_executor = ToolExecutor(cfg)
        self.solution_confidence = 0.0
        
        if not BOOKEND_TRUNCATION_AVAILABLE and cfg.truncation_strategy in ['bookend', 'smart_bookend']:
            LOG.warning(f"Bookend truncation module not available. Falling back to sequential truncation.")
        else:
            LOG.info(f"Using truncation strategy: {cfg.truncation_strategy}")

    def log_example_prompt(self, data):
        return

    def detect_solution_confidence(self, turn_output: str) -> float:
        """Detect if the model has found a confident solution."""
        confidence_indicators = [
            (r"(?i)the\s+(bug|issue|problem)\s+is\s+(in|at|located)", 0.3),
            (r"(?i)found\s+the\s+(bug|issue|problem)", 0.3),
            (r"(?i)this\s+is\s+causing\s+the\s+(issue|problem|bug)", 0.3),
            (r"(?i)the\s+fix\s+(is|would\s+be|should\s+be)", 0.2),
            (r"(?i)needs?\s+to\s+be\s+(fixed|changed|updated|modified)", 0.2),
            (r"(?i)specifically.{0,20}(line|lines)\s+\d+", 0.2),
            (r"(?i)exact\s+location", 0.2),
            (r"<location>.*</location>", 0.4),  # Location tags are strong indicator
        ]
        
        confidence = 0.0
        for pattern, weight in confidence_indicators:
            if re.search(pattern, turn_output):
                confidence += weight
        
        return min(confidence, 1.0)

    async def process_single_datapoint(self, data_point, all_data):
        """SOTA process with subtle optimizations for maximum precision."""

        LOG.debug(
            f"Initial data_point keys: {list(data_point.keys()) if isinstance(data_point, dict) else 'not a dict'}"
        )
        if 'turns' in data_point:
            LOG.debug(f"Initial turns structure: {data_point['turns']}")

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

        if 'turns' in data_point and isinstance(data_point['turns'], list) and len(data_point['turns']) > 0:
            for i, turn in enumerate(data_point['turns']):
                if isinstance(turn, dict):
                    turn.setdefault('inputs', '')
                    turn.setdefault('assistant', '')
                    turn.setdefault('tool_call', None)
                    turn.setdefault('tool_output', '')
                    LOG.debug(
                        f"Turn {i} after setdefault - keys: {list(turn.keys())}, has assistant: {'assistant' in turn}"
                    )
                else:
                    LOG.warning(f"Found non-dict turn at index {i}: {type(turn)}, replacing with empty structure")
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
                    LOG.debug(f"Ground truth check debug info: {debug_info}")
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

            # Add subtle focus hint for first turn
            if self.cfg.enable_subtle_focus_hints:
                inputs = inject_subtle_focus_hint(inputs, 0, total_steps)

            data_point['turns'][0]['inputs'] = inputs
            LOG.debug(f"Initialized turns with problem statement, turn count: {len(data_point['turns'])}")

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
                    LOG.error(f"Invalid turns structure at step {cur_step}: {data_point.get('turns', 'missing')}")
                    status = "failed"
                    reason = "invalid_turns_structure"
                    break

                # Add subtle hints at strategic points
                if self.cfg.enable_subtle_focus_hints and len(data_point['turns']) > 0:
                    last_turn = data_point['turns'][-1]
                    if isinstance(last_turn, dict) and last_turn.get('inputs', '').strip():
                        original_inputs = last_turn['inputs']
                        modified_inputs = inject_subtle_focus_hint(original_inputs, cur_step, total_steps)
                        if modified_inputs != original_inputs:
                            last_turn['inputs'] = modified_inputs
                            LOG.debug(f"Added subtle focus hint at step {cur_step}")

                # SOTA truncation strategy
                if hasattr(self.cfg, 'max_seq_length') and self.cfg.max_seq_length is not None and self.cfg.max_seq_length > 0:
                    original_turns_count = len(data_point['turns'])
                    
                    if self.cfg.enable_precision_truncation:
                        # Use custom precision-optimized truncation
                        data_point['turns'] = optimize_truncation_for_precision(
                            data_point['turns'],
                            self.cfg.max_seq_length,
                            self.cfg.inference.tokens_to_generate,
                            self.cfg.context_safety_margin
                        )
                    elif self.cfg.truncation_strategy == 'enhanced' and ENHANCED_CONTEXT_AVAILABLE:
                        # Standard enhanced truncation
                        if not hasattr(self, '_token_counter'):
                            self._token_counter = TokenCounter(getattr(self.cfg, 'model', 'gpt-4'))
                        
                        data_point['turns'], truncation_stats = enhanced_truncate_dialogue(
                            data_point['turns'], 
                            self.cfg.max_seq_length, 
                            self.cfg.inference.tokens_to_generate,
                            safety_margin=self.cfg.context_safety_margin,
                            token_counter=self._token_counter
                        )
                        LOG.info(f"Enhanced truncation stats: {truncation_stats}")
                    else:
                        # Fallback truncation
                        data_point['turns'] = truncate_dialogue_history(
                            data_point['turns'], self.cfg.max_seq_length, self.cfg.inference.tokens_to_generate
                        )
                    
                    if len(data_point['turns']) < original_turns_count:
                        LOG.info(f"Truncated dialogue from {original_turns_count} to {len(data_point['turns'])} turns")

                prepared_data_point = copy.deepcopy(data_point)
                
                # Loop detection with early intervention
                if LOOP_DETECTION_AVAILABLE and self.cfg.enable_loop_detection and len(chat_history) >= self.cfg.loop_detection_threshold - 1:
                    is_loop, loop_info = detect_repetitive_tool_calls(chat_history, self.cfg.loop_detection_threshold - 1)
                    
                    if is_loop:
                        LOG.warning(f"Potential loop detected before generation! Previous {loop_info['total_repetitions']} calls were identical")
                        prepared_data_point['turns'] = inject_loop_intervention(prepared_data_point['turns'], loop_info)
                
                # Proactive context check
                if ENHANCED_CONTEXT_AVAILABLE and getattr(self.cfg, 'enable_enhanced_context', True):
                    will_fit, error_msg, context_stats = check_context_before_generation(
                        prepared_data_point, 
                        self.cfg,
                        getattr(self, '_token_counter', None)
                    )
                    if not will_fit:
                        LOG.error(f"Context length check failed: {error_msg}")
                        LOG.error(f"Context stats: {context_stats}")
                        status = "failed"
                        reason = "context_length_exceeded_proactive"
                        break
                
                # Final turn prompt - use earlier and more concise
                if FINAL_TURN_PROMPT_AVAILABLE and should_inject_final_turn(
                    cur_step, 
                    total_steps, 
                    status,
                    enable_final_turn_prompt=getattr(self.cfg, 'enable_final_turn_prompt', True),
                    final_turn_threshold=getattr(self.cfg, 'final_turn_threshold', 0.9)
                ):
                    LOG.info(f"Injecting final turn instruction at step {cur_step + 1}/{total_steps}")
                    prepared_data_point['turns'] = inject_final_turn_instruction(
                        prepared_data_point['turns'],
                        is_final_turn=True,
                        instruction_type=getattr(self.cfg, 'final_turn_instruction_type', 'concise')
                    )

                # Direct LLM call
                try:
                    LOG.info(f"Sending {len(prepared_data_point['turns'])} turns to LLM")
                    llm_output = await super().process_single_datapoint(prepared_data_point, all_data)
                    
                except openai.BadRequestError as e:
                    if 'Please reduce the length of the messages or completion' in str(e) or 'is longer than the model\'s context length' in str(e):
                        LOG.warning(
                            "Artsiv generation failed due to running out of context. "
                            "Failing for subsequent subtasks automatically.",
                        )
                        status = "failed"
                        reason = "context_length_exceeded"
                        break
                    LOG.warning(f"Artsiv generation failed with BadRequestError: {e}")
                    status = "failed"
                    reason = f"bad_request_error: {str(e)}"
                    break

                generated_tokens = llm_output.get('num_generated_tokens', 0)
                total_generated_tokens += generated_tokens
                
                if generated_tokens == self.cfg.inference.tokens_to_generate:
                    LOG.warning(
                        f"Model generated exactly {generated_tokens} tokens (the configured limit). "
                        f"Response was likely truncated. Consider the response incomplete."
                    )
                    llm_output['_likely_truncated'] = True

                chat_history.append(llm_output)
                
                # Update solution confidence
                if self.cfg.enable_early_success_detection:
                    turn_confidence = self.detect_solution_confidence(
                        llm_output.get('_full_generation', llm_output.get('generation', ''))
                    )
                    self.solution_confidence = max(self.solution_confidence, turn_confidence)
                    LOG.debug(f"Solution confidence: {self.solution_confidence:.2f}")
                
                # Loop detection after generation
                if LOOP_DETECTION_AVAILABLE and self.cfg.enable_loop_detection and len(chat_history) >= self.cfg.loop_detection_threshold:
                    is_loop, loop_info = detect_repetitive_tool_calls(chat_history, self.cfg.loop_detection_threshold)
                    
                    if is_loop:
                        LOG.warning(f"Loop detected! Agent has repeated the same tool call {loop_info['total_repetitions']} times")
                        LOG.debug(f"Loop details: {loop_info}")
                        
                        data_point['turns'] = inject_loop_intervention(data_point['turns'], loop_info)
                        
                        loop_warning = {
                            '_loop_detected': True,
                            '_loop_info': loop_info,
                            '_intervention_added': True
                        }
                        chat_history[-1].update(loop_warning)
                        
                        pattern_analysis = analyze_loop_patterns(chat_history)
                        LOG.debug(f"Pattern analysis: {pattern_analysis}")

                # Remove thinking tags
                if self.cfg.remove_thinking:
                    remove_thinking(llm_output, 'generation', self.cfg.thinking_begin, self.cfg.thinking_end)

                # Response extraction
                try:
                    extracted_block = DialogProcessor.extract_response(llm_output['generation'], self.cfg)
                except Exception as e:
                    LOG.error(f"Error extracting response from LLM output: {e}")
                    LOG.debug(
                        f"LLM output was: {llm_output.get('generation', 'None')[:500]}..."
                    )
                    status = "failed"
                    reason = f"response_extraction_error: {str(e)}"
                    break

                if not extracted_block:
                    LOG.warning("Model failed to generate a tool use or location. Ending generation.")
                    if llm_output.get('_likely_truncated', False):
                        status = "failed"
                        reason = "response_truncated_at_token_limit"
                        LOG.error(
                            f"Response was truncated at token limit ({generated_tokens} tokens) "
                            f"and no valid tool/location was extracted."
                        )
                    else:
                        status = "failed"
                        reason = "no_tool_or_location_generated"
                    break

                # Turn management
                try:
                    if data_point['turns'] and len(data_point['turns']) > 0:
                        current_turn = data_point['turns'][-1]
                        if isinstance(current_turn, dict):
                            current_turn['assistant'] = llm_output['generation']
                            current_turn['assistant_raw'] = llm_output.get('raw_generation', llm_output['generation'])
                            current_turn['assistant_raw_w_think'] = llm_output.get(
                                '_full_generation', llm_output['generation']
                            )

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

                    if data_point['turns'] and len(data_point['turns']) > 0:
                        current_turn = data_point['turns'][-1]
                        if isinstance(current_turn, dict):
                            current_turn['tool_output'] = tool_output_to_store
                            LOG.debug(f"Added tool output to current turn {len(data_point['turns'])-1}")
                            
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
                    
                    # Log confidence at success
                    if self.cfg.enable_early_success_detection:
                        LOG.info(f"Success with solution confidence: {self.solution_confidence:.2f}")
                    break

                # Early success detection - if highly confident, suggest wrapping up
                if (self.cfg.enable_early_success_detection and 
                    self.solution_confidence >= self.cfg.success_confidence_threshold and
                    cur_step >= 5):  # Don't stop too early
                    LOG.info(f"High solution confidence ({self.solution_confidence:.2f}), encouraging wrap-up")
                    # Don't force stop, but inject a stronger hint
                    if len(data_point['turns']) > 0:
                        last_turn = data_point['turns'][-1]
                        if isinstance(last_turn, dict) and 'inputs' in last_turn:
                            last_turn['inputs'] += "\n\n🎯 You seem to have found the issue. If confident, please provide the location.\n"

                if data_point.get('turns') and len(data_point['turns']) > 0:
                    last_turn = data_point['turns'][-1]
                    has_assistant = isinstance(last_turn, dict) and 'assistant' in last_turn
                    LOG.debug(
                        f"Current turn count: {len(data_point['turns'])}, last turn has assistant: {has_assistant}"
                    )
                else:
                    LOG.debug("No turns available to check for assistant field")

                if cur_step == total_steps - 1 and status is None:
                    status = "failed"
                    reason = "max_steps_exceeded"
                    break

            if status is None:
                status = "failed"
                if reason is None:
                    reason = "unknown_failure"
        except Exception as e:
            LOG.error(f"Unexpected error in process_single_datapoint: {e}")
            import traceback

            full_traceback = traceback.format_exc()
            LOG.error(f"Full traceback:\n{full_traceback}")

            print(f"\n{'='*60}")
            print(f"ERROR in process_single_datapoint: {e}")
            print(f"Error type: {type(e).__name__}")
            print(f"Full traceback:\n{full_traceback}")

            status = "failed"
            reason = f"exception: {str(e)}"

        # Turn validation and cleanup
        if 'turns' not in data_point:
            LOG.warning("Missing 'turns' in data_point at return time, initializing empty structure")
            data_point['turns'] = []

        for i, turn in enumerate(data_point.get('turns', [])):
            if not isinstance(turn, dict):
                LOG.error(f"Turn {i} is not a dictionary: {type(turn)}")
                data_point['turns'][i] = {
                    "inputs": str(turn) if turn else "",
                    "assistant": "",
                    "tool_call": None,
                    "tool_output": "",
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

                LOG.debug(
                    f"Final turn {i} validation - has assistant: {'assistant' in turn}, keys: {list(turn.keys())}"
                )

        if 'turns' in data_point:
            LOG.debug(f"Returning {len(data_point['turns'])} turns")

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
