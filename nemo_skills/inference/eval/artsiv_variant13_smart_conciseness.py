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
VARIANT 13: Smart Conciseness Management
HYPOTHESIS: Prevent token exhaustion failures through intelligent verbosity control
Strategy: 
- Enable and optimize response length management
- Proactive conciseness prompts based on token usage patterns
- Dynamic limits based on investigation progress
- Smart thinking tag management
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

# Import response length management
try:
    from nemo_skills.inference.eval.artsiv_utils.v4.dialog_processor import (
        check_response_length,
        inject_length_warning,
        calculate_safe_token_limit,
        DEFAULT_MAX_RESPONSE_TOKENS
    )
    RESPONSE_LENGTH_AVAILABLE = True
except ImportError:
    RESPONSE_LENGTH_AVAILABLE = False

PROMPT_TEMPLATE_VERSION: str = "v4"

module_base = f"nemo_skills.inference.eval.artsiv_utils.{PROMPT_TEMPLATE_VERSION}"

dialog_processor = importlib.import_module(f"{module_base}.dialog_processor")
tool_executor = importlib.import_module(f"{module_base}.tool_executor")

DialogProcessor = dialog_processor.DialogProcessor
ToolExecutor = tool_executor.ToolExecutor
truncate_dialogue_history = dialog_processor.truncate_dialogue_history

LOG = logging.getLogger(get_logger_name(__file__))


def inject_smart_conciseness_prompt(inputs: str, step: int, total_steps: int, 
                                   verbosity_score: float = 0.0, 
                                   average_tokens_per_turn: float = 0.0) -> str:
    """Inject conciseness prompts based on verbosity patterns."""
    
    # Early gentle reminder
    if step == 2 and average_tokens_per_turn > 15000:
        prompt = """
📝 **Efficiency Reminder**: Your thinking is thorough, but please be more concise.
Focus on key findings and limit exploration to the most relevant areas.
"""
        return inputs + prompt
    
    # Stronger prompt if verbosity continues
    if step >= 5 and verbosity_score > 0.7:
        prompt = """
⚠️ **Conciseness Required**: You're using excessive tokens in thinking.
Please:
- Summarize findings briefly
- Focus only on the most likely bug locations
- Avoid repeating previous observations
- Keep thinking concise and to the point
"""
        return inputs + prompt
    
    # Critical intervention
    if average_tokens_per_turn > 20000:
        prompt = """
🚨 **TOKEN LIMIT WARNING**: Your responses are too verbose!
You MUST be more concise to avoid hitting token limits.
- Limit thinking to essential analysis only
- No lengthy explorations or repeated observations
- Get to the point quickly
- Focus on finding the bug location efficiently
"""
        return inputs + prompt
    
    # Near the end, demand brevity
    if step >= total_steps - 4:
        prompt = """
⏰ **Final Phase - Be Brief**: Approaching the end of investigation.
Provide concise analysis and specific bug locations.
No lengthy explorations - just key findings and locations.
"""
        return inputs + prompt
    
    return inputs


def calculate_verbosity_score(chat_history: list) -> float:
    """Calculate a verbosity score based on token usage patterns."""
    if not chat_history:
        return 0.0
    
    recent_history = chat_history[-5:]  # Last 5 turns
    total_tokens = sum(turn.get('num_generated_tokens', 0) for turn in recent_history)
    avg_tokens = total_tokens / len(recent_history) if recent_history else 0
    
    # Score based on average tokens per turn
    if avg_tokens > 25000:
        return 1.0  # Extremely verbose
    elif avg_tokens > 20000:
        return 0.8  # Very verbose
    elif avg_tokens > 15000:
        return 0.6  # Moderately verbose
    elif avg_tokens > 10000:
        return 0.4  # Slightly verbose
    else:
        return 0.2  # Acceptable
    

def calculate_dynamic_token_limit(step: int, total_steps: int, 
                                 verbosity_score: float,
                                 base_limit: int = 81920) -> int:
    """Calculate dynamic token limit based on progress and verbosity."""
    
    # Start with base limit
    limit = base_limit
    
    # Reduce limit for verbose models
    if verbosity_score > 0.6:
        limit = int(limit * 0.5)  # Cut in half for very verbose models
    elif verbosity_score > 0.4:
        limit = int(limit * 0.7)  # Reduce by 30% for moderately verbose
    
    # Further reduce as we approach the end
    progress = step / total_steps
    if progress > 0.8:
        limit = min(limit, 20000)  # Max 20k tokens near end
    elif progress > 0.6:
        limit = min(limit, 40000)  # Max 40k tokens in late stages
    
    # Never go below a minimum
    return max(limit, 10000)


@nested_dataclass(kw_only=True)
class ArtsivGenerationConfig(GenerateSolutionsConfig):
    # Based on V2 but with smart conciseness management
    inference: InferenceConfig = field(default_factory=lambda: InferenceConfig(
        temperature=0.7,
        top_k=0,
        top_p=0.95,
        min_p=0.0,
        random_seed=0,
        tokens_to_generate=81920,  # Base limit, dynamically adjusted
        repetition_penalty=1.0,
        top_logprobs=None,
        extra_body={}
    ))
    server: dict = field(default_factory=dict)

    # Core settings from V2
    mount_directory: str = "/repos/"
    remove_thinking: bool = True
    total_steps: int = 20

    # Repository filtering from V11/V12
    file_extensions: list = field(default_factory=lambda: ["py", "cfg", "yml", "yaml", "toml"])
    exclude_dirs: list = field(
        default_factory=lambda: [
            "test", "tests", "testing", "test_", "_test", "tests_", "_tests",
            "unittest", "pytest", "nose", "tox",
            "__pycache__", ".git", ".github", ".gitlab", ".gitignore",
            "build", "dist", "target", "bin", "obj", "out",
            ".pytest_cache", ".tox", ".mypy_cache", ".coverage",
            "docs", "documentation", "doc", "_build", "sphinx",
            "venv", "env", ".env", "virtualenv", ".venv",
            "node_modules", "vendor", "vendors", "third_party", "3rdparty",
            "libs", "lib", "dependencies", "packages", "pkg",
            "examples", "example", "demo", "demos", "samples", "sample",
            "deploy", "deployment", "docker", "kubernetes", "k8s", ".docker",
            "ci", "cd", ".ci", ".circleci", ".travis", ".jenkins",
            "static", "assets", "media", "images", "img", "css", "js",
            "public", "resources", "res",
            "data", "datasets", "fixtures", "fixture", "mocks", "mock",
            "notebooks", "notebook", "jupyter", ".ipynb_checkpoints",
            "tmp", "temp", "temporary", "cache", ".cache",
            "logs", "log", ".logs",
            "locale", "locales", "translations", "i18n", "l10n",
            "scripts", "script", "tools", "tool", "utils", "util",
            "migrations", "migration",
            "settings", "setting", "config", "configs", "conf",
            "local_settings", "local",
        ]
    )

    # Tool detection settings
    enable_implicit_tool_detection: bool = True
    common_words_filter: list = field(
        default_factory=lambda: [
            "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "can", "this", "that", "these", "those",
            "a", "an", "as", "if", "then", "else", "when", "where", "why", "how", "what", "which",
            "who", "whom", "whose", "it", "its", "they", "them", "their", "we", "our", "you", "your",
            "need", "find", "search", "look", "check", "verify", "ensure", "make", "sure",
            "function", "class", "method", "variable", "parameter", "argument", "value",
            "return", "returns", "import", "from", "def", "self", "init", "main",
            "error", "exception", "bug", "issue", "problem", "fix", "patch",
            "file", "line", "code", "source", "implementation", "logic",
            "get", "set", "update", "delete", "create", "save", "load", "run", "execute",
            "process", "handle", "manage", "validate", "parse", "format", "convert",
        ]
    )

    # Context settings
    max_seq_length: int = 262144
    show_line_counts: bool = False
    max_view_lines: int = 800  # Reasonable limit

    # V13 Smart Conciseness settings
    truncation_strategy: str = "bookend"  # Keep V2's strategy
    enable_loop_detection: bool = True
    loop_detection_threshold: int = 3
    enable_enhanced_context: bool = True
    context_safety_margin: float = 0.9  # Standard margin
    use_tiktoken: bool = True
    enable_final_turn_prompt: bool = True
    final_turn_instruction_type: str = "concise"
    final_turn_threshold: float = 1.0
    
    # ENABLE smart response length management
    enable_response_length_management: bool = True  # KEY CHANGE!
    enable_smart_conciseness: bool = True  # NEW
    enable_dynamic_token_limits: bool = True  # NEW
    
    # Response length settings
    max_normal_response_tokens: int = 10000  # Stricter than default 8192
    max_thinking_response_tokens: int = 20000  # Stricter than default 16384
    max_final_turn_tokens: int = 5000  # Slightly more than default 4096
    max_retry_tokens: int = 2000  # Same as default
    
    # Verbosity management
    verbosity_warning_threshold: float = 0.5  # Warn at 50% verbosity score
    verbosity_critical_threshold: float = 0.7  # Critical at 70% verbosity
    enable_verbosity_tracking: bool = True
    enable_conciseness_prompts: bool = True


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_artsiv_generation_config", node=ArtsivGenerationConfig)


class ArtsivGenerationTask(GenerationTask):
    def __init__(self, cfg: ArtsivGenerationConfig):
        super().__init__(cfg)
        self.tool_executor = ToolExecutor(cfg)
        self.verbosity_scores = []  # Track verbosity over time
        
        if not BOOKEND_TRUNCATION_AVAILABLE and cfg.truncation_strategy in ['bookend', 'smart_bookend']:
            LOG.warning(f"Bookend truncation module not available. Falling back to sequential truncation.")
        else:
            LOG.info(f"Using truncation strategy: {cfg.truncation_strategy}")

    def log_example_prompt(self, data):
        return

    async def process_single_datapoint(self, data_point, all_data):
        """Process with smart conciseness management."""

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

            # Add initial conciseness guidance
            if self.cfg.enable_conciseness_prompts:
                inputs += """

💡 **Note**: Please be concise in your investigation. Focus on the most relevant areas and avoid excessive exploration.
"""

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

                # Calculate verbosity metrics
                verbosity_score = 0.0
                avg_tokens_per_turn = 0.0
                
                if self.cfg.enable_verbosity_tracking and chat_history:
                    verbosity_score = calculate_verbosity_score(chat_history)
                    total_tokens_so_far = sum(turn.get('num_generated_tokens', 0) for turn in chat_history)
                    avg_tokens_per_turn = total_tokens_so_far / len(chat_history)
                    
                    LOG.debug(f"Step {cur_step}: verbosity_score={verbosity_score:.2f}, avg_tokens={avg_tokens_per_turn:.0f}")
                    
                    # Track for analysis
                    self.verbosity_scores.append(verbosity_score)

                # Inject conciseness prompts based on verbosity
                if self.cfg.enable_conciseness_prompts and len(data_point['turns']) > 0:
                    last_turn = data_point['turns'][-1]
                    if isinstance(last_turn, dict) and last_turn.get('inputs', '').strip():
                        original_inputs = last_turn['inputs']
                        modified_inputs = inject_smart_conciseness_prompt(
                            original_inputs, cur_step, total_steps, 
                            verbosity_score, avg_tokens_per_turn
                        )
                        if modified_inputs != original_inputs:
                            last_turn['inputs'] = modified_inputs
                            LOG.info(f"Injected conciseness prompt at step {cur_step}")

                # Dynamic token limit adjustment
                if self.cfg.enable_dynamic_token_limits:
                    dynamic_limit = calculate_dynamic_token_limit(
                        cur_step, total_steps, verbosity_score, 
                        self.cfg.inference.tokens_to_generate
                    )
                    if dynamic_limit < self.cfg.inference.tokens_to_generate:
                        LOG.info(f"Adjusting token limit from {self.cfg.inference.tokens_to_generate} to {dynamic_limit}")
                        # Temporarily adjust for this generation
                        original_limit = self.cfg.inference.tokens_to_generate
                        self.cfg.inference.tokens_to_generate = dynamic_limit

                # Context management with standard truncation
                if hasattr(self.cfg, 'max_seq_length') and self.cfg.max_seq_length is not None and self.cfg.max_seq_length > 0:
                    original_turns_count = len(data_point['turns'])
                    
                    truncation_strategy = getattr(self.cfg, 'truncation_strategy', 'sequential')
                    
                    if (ENHANCED_CONTEXT_AVAILABLE and 
                        getattr(self.cfg, 'enable_enhanced_context', True) and
                        (truncation_strategy == 'enhanced' or 
                         getattr(self.cfg, 'use_tiktoken', True))):
                        LOG.debug(f"Using enhanced context management")
                        if not hasattr(self, '_token_counter'):
                            self._token_counter = TokenCounter(getattr(self.cfg, 'model', 'gpt-4'))
                        
                        data_point['turns'], truncation_stats = enhanced_truncate_dialogue(
                            data_point['turns'], 
                            self.cfg.max_seq_length, 
                            self.cfg.inference.tokens_to_generate,
                            safety_margin=getattr(self.cfg, 'context_safety_margin', 0.9),
                            token_counter=self._token_counter
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
                        if truncation_strategy != 'sequential' and not BOOKEND_TRUNCATION_AVAILABLE:
                            LOG.warning(f"Truncation strategy '{truncation_strategy}' not available, using sequential")
                        LOG.debug(f"Using sequential truncation strategy")
                        data_point['turns'] = truncate_dialogue_history(
                            data_point['turns'], self.cfg.max_seq_length, self.cfg.inference.tokens_to_generate
                        )
                    
                    if len(data_point['turns']) < original_turns_count:
                        LOG.info(f"Truncated dialogue from {original_turns_count} to {len(data_point['turns'])} turns using {truncation_strategy} strategy")

                prepared_data_point = copy.deepcopy(data_point)
                
                # Loop detection
                if LOOP_DETECTION_AVAILABLE and self.cfg.enable_loop_detection and len(chat_history) >= self.cfg.loop_detection_threshold - 1:
                    is_loop, loop_info = detect_repetitive_tool_calls(chat_history, self.cfg.loop_detection_threshold - 1)
                    
                    if is_loop:
                        LOG.warning(f"Potential loop detected before generation! Previous {loop_info['total_repetitions']} calls were identical")
                        prepared_data_point['turns'] = inject_loop_intervention(prepared_data_point['turns'], loop_info)
                
                # Context check
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
                
                # Final turn prompt
                if FINAL_TURN_PROMPT_AVAILABLE and should_inject_final_turn(
                    cur_step, 
                    total_steps, 
                    status,
                    enable_final_turn_prompt=getattr(self.cfg, 'enable_final_turn_prompt', True),
                    final_turn_threshold=getattr(self.cfg, 'final_turn_threshold', 1.0)
                ):
                    LOG.info(f"Injecting final turn instruction at step {cur_step + 1}/{total_steps}")
                    prepared_data_point['turns'] = inject_final_turn_instruction(
                        prepared_data_point['turns'],
                        is_final_turn=True,
                        instruction_type=getattr(self.cfg, 'final_turn_instruction_type', 'concise')
                    )

                # LLM call
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
                
                # Check if we hit the token limit
                if generated_tokens == self.cfg.inference.tokens_to_generate:
                    LOG.warning(
                        f"Model generated exactly {generated_tokens} tokens (the configured limit). "
                        f"Response was likely truncated. This may cause no_tool_or_location_generated."
                    )
                    llm_output['_likely_truncated'] = True
                    
                    # If we hit token limit and have high verbosity, inject strong warning
                    if verbosity_score > 0.5 and self.cfg.enable_response_length_management:
                        LOG.warning("High verbosity model hit token limit - injecting strong conciseness warning")
                        warning_turn = {
                            "inputs": """
🛑 **CRITICAL**: Your last response was truncated due to token limits!
You MUST be MUCH more concise. Your thinking is too verbose.
- Drastically reduce thinking length
- Focus ONLY on essential analysis
- Provide tool calls/locations quickly
- No lengthy explorations

Be BRIEF or you will fail to complete the task!
""",
                            "assistant": "",
                            "tool_call": None,
                            "tool_output": "",
                        }
                        data_point['turns'].append(warning_turn)

                chat_history.append(llm_output)
                
                # Restore original token limit if it was adjusted
                if self.cfg.enable_dynamic_token_limits and 'original_limit' in locals():
                    self.cfg.inference.tokens_to_generate = original_limit
                
                # Response length check (if available)
                if RESPONSE_LENGTH_AVAILABLE and self.cfg.enable_response_length_management:
                    full_response = llm_output.get('_full_generation', llm_output.get('generation', ''))
                    
                    # Determine response type
                    has_thinking = '<think>' in full_response or '<thinking>' in full_response
                    is_final = cur_step >= total_steps - 3
                    response_type = 'final_turn' if is_final else ('thinking' if has_thinking else 'normal')
                    
                    # Check against appropriate limits
                    max_tokens = {
                        'normal': self.cfg.max_normal_response_tokens,
                        'thinking': self.cfg.max_thinking_response_tokens,
                        'final_turn': self.cfg.max_final_turn_tokens,
                        'retry': self.cfg.max_retry_tokens
                    }.get(response_type, self.cfg.max_normal_response_tokens)
                    
                    is_acceptable, warning_msg, stats = check_response_length(
                        full_response, response_type, max_tokens
                    )
                    
                    if not is_acceptable or warning_msg:
                        LOG.warning(f"Response length issue: {warning_msg}")
                        if not is_acceptable and cur_step < total_steps - 1:
                            # Inject conciseness warning for next turn
                            data_point['turns'] = inject_length_warning(data_point['turns'], warning_msg)
                
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
                            f"and no valid tool/location was extracted. This is likely due to excessive verbosity."
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
                    
                    # Log verbosity analysis
                    if self.verbosity_scores:
                        avg_verbosity = sum(self.verbosity_scores) / len(self.verbosity_scores)
                        LOG.info(f"Success with average verbosity score: {avg_verbosity:.2f}")
                    break

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
