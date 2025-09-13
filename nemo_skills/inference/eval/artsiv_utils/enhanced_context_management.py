"""
Enhanced context length management for Artsiv.

This module provides improved token counting and context management to prevent
context length exceeded errors.
"""

import logging
from typing import List, Dict, Optional, Tuple
import copy

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

LOG = logging.getLogger(__name__)


class TokenCounter:
    """Accurate token counting with fallback to character-based estimation."""
    
    def __init__(self, model_name: str = "gpt-4"):
        """Initialize token counter for specific model."""
        self.model_name = model_name
        self.encoding = None
        
        if TIKTOKEN_AVAILABLE:
            try:
                # Try to get encoding for the specific model
                self.encoding = tiktoken.encoding_for_model(model_name)
                LOG.info(f"Using tiktoken for accurate token counting (model: {model_name})")
            except Exception as e:
                LOG.warning(f"Failed to get tiktoken encoding for {model_name}: {e}")
                try:
                    # Fall back to cl100k_base encoding (used by GPT-4)
                    self.encoding = tiktoken.get_encoding("cl100k_base")
                    LOG.info("Using cl100k_base encoding as fallback")
                except Exception as e2:
                    LOG.warning(f"Failed to get any tiktoken encoding: {e2}")
                    self.encoding = None
        else:
            LOG.warning("tiktoken not available, using character-based estimation")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text, with fallback to estimation."""
        if not text:
            return 0
            
        if self.encoding is not None:
            try:
                # Use tiktoken for accurate counting
                return len(self.encoding.encode(text))
            except Exception as e:
                LOG.warning(f"tiktoken encoding failed: {e}, falling back to estimation")
        
        # Fallback to improved character-based estimation
        return self._estimate_tokens(text)
    
    def _estimate_tokens(self, text: str) -> int:
        """Improved character-based token estimation."""
        if not text:
            return 0
        
        # More accurate character-to-token ratios based on empirical data
        # These are calibrated for GPT-style tokenization
        
        # Count different types of content
        code_indicators = [
            'def ', 'class ', 'import ', 'from ', 'return ', 'if ', 'elif ', 'else:',
            'for ', 'while ', 'try:', 'except:', 'finally:', 'with ', 
            'function', 'const ', 'let ', 'var ', '===', '!==',  # JS patterns
            '()', '[]', '{}', '->', '=>', '...', '::',
        ]
        
        # Special characters that often create multiple tokens
        special_chars = sum(1 for c in text if c in '{}[]()<>|\\/@#$%^&*+=~`')
        newlines = text.count('\n')
        spaces = text.count(' ')
        
        # Calculate code likelihood
        text_lower = text.lower()
        code_score = sum(text_lower.count(ind) for ind in code_indicators)
        code_ratio = min(1.0, code_score / max(1, len(text) / 100))
        
        # Adjust for special characters and formatting
        special_ratio = special_chars / max(1, len(text))
        
        # More accurate ratios based on content type
        if code_ratio > 0.5:
            # Code with lots of symbols
            base_ratio = 3.0  # More tokens per character in code
        elif special_ratio > 0.1:
            # Text with many special characters (like logs, paths)
            base_ratio = 3.2
        else:
            # Regular text
            base_ratio = 3.8
        
        # Adjust for whitespace (which often doesn't create tokens)
        effective_chars = len(text) - (spaces * 0.5) - (newlines * 0.8)
        estimated_tokens = int(effective_chars / base_ratio)
        
        # Add safety margin (10% buffer)
        return int(estimated_tokens * 1.1)


def count_dialogue_tokens(turns: List[Dict], token_counter: Optional[TokenCounter] = None) -> int:
    """Count total tokens in dialogue with improved accuracy."""
    if token_counter is None:
        token_counter = TokenCounter()
    
    total_tokens = 0
    
    for turn in turns:
        if not isinstance(turn, dict):
            continue
            
        # Count all text fields in the turn
        for field in ['inputs', 'assistant', 'tool_output', 'assistant_raw', 'assistant_raw_w_think']:
            if field in turn and turn[field]:
                content = str(turn[field])
                tokens = token_counter.count_tokens(content)
                total_tokens += tokens
        
        # Count tool_call if present (JSON structure)
        if 'tool_call' in turn and turn['tool_call']:
            tool_call_str = str(turn['tool_call'])
            total_tokens += token_counter.count_tokens(tool_call_str)
    
    # Add overhead for message structure and formatting
    # Each turn adds ~10-20 tokens for role markers and structure
    message_overhead = len(turns) * 15
    total_tokens += message_overhead
    
    return total_tokens


def enhanced_truncate_dialogue(
    turns: List[Dict], 
    max_seq_length: int, 
    tokens_to_generate: int,
    safety_margin: float = 0.9,
    token_counter: Optional[TokenCounter] = None
) -> Tuple[List[Dict], Dict]:
    """
    Enhanced dialogue truncation with better token counting and safety margins.
    
    Args:
        turns: List of dialogue turns
        max_seq_length: Maximum context length in tokens
        tokens_to_generate: Tokens to reserve for generation
        safety_margin: Use only this fraction of available space (default 0.9)
        token_counter: Optional TokenCounter instance
        
    Returns:
        Tuple of (truncated_turns, stats_dict)
    """
    if token_counter is None:
        token_counter = TokenCounter()
    
    stats = {
        'original_turns': len(turns),
        'original_tokens': 0,
        'final_turns': 0,
        'final_tokens': 0,
        'truncation_applied': False,
        'truncation_strategy': None
    }
    
    if not turns:
        return turns, stats
    
    # Calculate safe target with margin
    safe_max_length = int(max_seq_length * safety_margin)
    target_tokens = safe_max_length - tokens_to_generate
    
    # Count current tokens accurately
    current_tokens = count_dialogue_tokens(turns, token_counter)
    stats['original_tokens'] = current_tokens
    
    LOG.info(f"Context check: {current_tokens} tokens, target: {target_tokens} "
             f"(safe max: {safe_max_length}, reserve: {tokens_to_generate})")
    
    if current_tokens <= target_tokens:
        stats['final_turns'] = len(turns)
        stats['final_tokens'] = current_tokens
        return turns, stats
    
    LOG.warning(f"Context length exceeded: {current_tokens} > {target_tokens}")
    stats['truncation_applied'] = True
    
    # Try different truncation strategies in order of preference
    strategies = [
        ('smart_bookend', _smart_bookend_truncate),
        ('aggressive_bookend', _aggressive_bookend_truncate),
        ('emergency', _emergency_truncate)
    ]
    
    truncated_turns = None
    for strategy_name, strategy_func in strategies:
        LOG.info(f"Trying {strategy_name} truncation strategy")
        candidate_turns = strategy_func(turns, target_tokens, token_counter)
        candidate_tokens = count_dialogue_tokens(candidate_turns, token_counter)
        
        if candidate_tokens <= target_tokens:
            truncated_turns = candidate_turns
            stats['truncation_strategy'] = strategy_name
            stats['final_tokens'] = candidate_tokens
            LOG.info(f"{strategy_name} succeeded: {candidate_tokens} tokens")
            break
        else:
            LOG.warning(f"{strategy_name} still too long: {candidate_tokens} tokens")
    
    if truncated_turns is None:
        # Emergency fallback - keep only problem and last turn
        LOG.error("All truncation strategies failed, using emergency minimum")
        truncated_turns = _emergency_truncate(turns, target_tokens // 2, token_counter)
        stats['truncation_strategy'] = 'emergency_minimum'
        stats['final_tokens'] = count_dialogue_tokens(truncated_turns, token_counter)
    
    stats['final_turns'] = len(truncated_turns)
    LOG.info(f"Truncation complete: {stats['original_turns']} -> {stats['final_turns']} turns, "
             f"{stats['original_tokens']} -> {stats['final_tokens']} tokens")
    
    return truncated_turns, stats


def _smart_bookend_truncate(turns: List[Dict], target_tokens: int, token_counter: TokenCounter) -> List[Dict]:
    """Smart bookend truncation - keep first and last turns, summarize middle."""
    if len(turns) <= 2:
        return turns
    
    result = []
    
    # Always keep first turn (problem statement)
    result.append(copy.deepcopy(turns[0]))
    
    # Keep last 2-3 turns based on available space
    last_turns_to_keep = 2
    if len(turns) > 10:
        last_turns_to_keep = 3
    
    # Add last turns
    for turn in turns[-last_turns_to_keep:]:
        result.append(copy.deepcopy(turn))
    
    # Remove raw fields to save space
    for turn in result:
        for field in ['assistant_raw', 'assistant_raw_w_think']:
            if field in turn:
                del turn[field]
    
    # If still too long, reduce assistant responses
    current_tokens = count_dialogue_tokens(result, token_counter)
    if current_tokens > target_tokens:
        for i, turn in enumerate(result):
            if i == 0:  # Skip first turn
                continue
            if 'assistant' in turn and turn['assistant']:
                # Keep only first 200 chars of assistant responses
                if len(turn['assistant']) > 200:
                    turn['assistant'] = turn['assistant'][:200] + "... [truncated]"
    
    return result


def _aggressive_bookend_truncate(turns: List[Dict], target_tokens: int, token_counter: TokenCounter) -> List[Dict]:
    """Aggressive bookend - minimal context preservation."""
    if len(turns) <= 1:
        return turns
    
    result = []
    
    # Keep only problem statement from first turn
    first_turn = copy.deepcopy(turns[0])
    # Truncate long problem statements
    if 'inputs' in first_turn and len(first_turn['inputs']) > 1000:
        first_turn['inputs'] = first_turn['inputs'][:1000] + "... [problem truncated]"
    
    # Preserve the turn structure - keep essential keys that prompt system expects
    minimal_first_turn = {'inputs': first_turn.get('inputs', '')}
    
    # If the original turn had an 'assistant' field, preserve it (even if empty)
    if 'assistant' in first_turn:
        minimal_first_turn['assistant'] = first_turn.get('assistant', '')[:200] if first_turn.get('assistant') else ''
    
    # Preserve tool-related fields if they exist
    if 'tool_call' in first_turn:
        minimal_first_turn['tool_call'] = first_turn['tool_call']
    if 'tool_output' in first_turn:
        minimal_first_turn['tool_output'] = first_turn['tool_output'][:200] + "... [output truncated]" if len(first_turn.get('tool_output', '')) > 200 else first_turn.get('tool_output', '')
    
    result.append(minimal_first_turn)
    
    # Keep only the very last turn with minimal content
    if len(turns) > 1:
        last_turn = copy.deepcopy(turns[-1])
        # Keep only essential fields, preserving structure
        minimal_turn = {'inputs': last_turn.get('inputs', '')[:500] if 'inputs' in last_turn else ''}
        
        # Always preserve 'assistant' field if it exists in the original
        if 'assistant' in last_turn:
            minimal_turn['assistant'] = last_turn.get('assistant', '')[:200] if last_turn.get('assistant') else ''
            
        # Preserve tool-related fields if they exist
        if 'tool_call' in last_turn:
            minimal_turn['tool_call'] = last_turn['tool_call']
        if 'tool_output' in last_turn:
            minimal_turn['tool_output'] = last_turn['tool_output'][:200] + "... [output truncated]" if len(last_turn.get('tool_output', '')) > 200 else last_turn.get('tool_output', '')
            
        result.append(minimal_turn)
    
    return result


def _emergency_truncate(turns: List[Dict], target_tokens: int, token_counter: TokenCounter) -> List[Dict]:
    """Emergency truncation - absolute minimum context."""
    if not turns:
        return turns
    
    # Keep only a summary of the problem
    problem = turns[0].get('inputs', '')[:500]
    return [{'inputs': problem + "... [context heavily truncated due to length]"}]


def check_context_before_generation(
    data_point: Dict,
    cfg,
    token_counter: Optional[TokenCounter] = None
) -> Tuple[bool, Optional[str], Dict]:
    """
    Proactively check if context will fit before making LLM call.
    
    Returns:
        Tuple of (will_fit, error_message, stats)
    """
    if token_counter is None:
        token_counter = TokenCounter(getattr(cfg, 'model', 'gpt-4'))
    
    turns = data_point.get('turns', [])
    current_tokens = count_dialogue_tokens(turns, token_counter)
    
    max_seq_length = getattr(cfg, 'max_seq_length', 32768)
    tokens_to_generate = getattr(cfg, 'tokens_to_generate', 8192)
    safety_margin = getattr(cfg, 'context_safety_margin', 0.9)
    
    safe_max = int(max_seq_length * safety_margin)
    available_tokens = safe_max - tokens_to_generate
    
    stats = {
        'current_tokens': current_tokens,
        'available_tokens': available_tokens,
        'max_seq_length': max_seq_length,
        'tokens_to_generate': tokens_to_generate,
        'safety_margin': safety_margin
    }
    
    if current_tokens > available_tokens:
        error_msg = (f"Context too long: {current_tokens} tokens > {available_tokens} available "
                    f"(max: {max_seq_length}, generate: {tokens_to_generate}, safety: {safety_margin})")
        return False, error_msg, stats
    
    return True, None, stats
