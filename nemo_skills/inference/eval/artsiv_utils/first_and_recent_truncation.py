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
First-and-recent truncation strategy for dialogue history.

This strategy always preserves:
1. The first turn (problem statement)
2. As many recent turns as possible

Middle turns are removed as needed, starting with the oldest.
"""

import logging
from typing import List, Dict, Optional, Tuple
import copy

LOG = logging.getLogger(__name__)


def estimate_turn_tokens(turn: Dict, token_counter: Optional['TokenCounter'] = None) -> int:
    """Estimate tokens in a single turn.
    
    Args:
        turn: A dialogue turn dictionary
        token_counter: Optional TokenCounter for accurate counting
        
    Returns:
        Estimated token count
    """
    if token_counter is not None:
        # Use accurate token counting if available
        from nemo_skills.inference.eval.artsiv_utils.enhanced_context_management import count_dialogue_tokens
        return count_dialogue_tokens([turn], token_counter)
    
    # Fallback to character-based estimation
    total_chars = 0
    if isinstance(turn, dict):
        for key, value in turn.items():
            if value and isinstance(value, str):
                total_chars += len(value)
            elif value and isinstance(value, dict):
                # For tool_call dictionaries
                total_chars += len(str(value))
    
    return total_chars // 4  # Rough estimation: ~1 token per 4 chars


def first_and_recent_truncate(
    turns: List[Dict], 
    max_seq_length: int, 
    tokens_to_generate: int,
    token_counter: Optional['TokenCounter'] = None
) -> Tuple[List[Dict], Dict]:
    """
    Truncate dialogue keeping first turn and as many recent turns as possible.
    
    Strategy:
    1. Always keep the first turn (problem statement)
    2. Keep recent turns in reverse chronological order
    3. Remove middle turns as needed to fit within token limit
    
    Args:
        turns: List of dialogue turns
        max_seq_length: Maximum context length in tokens
        tokens_to_generate: Tokens to reserve for the next generation
        token_counter: Optional TokenCounter for accurate counting
        
    Returns:
        Tuple of (truncated turns, statistics dictionary)
    """
    if not turns:
        return turns, {"removed_turns": 0, "token_reduction": 0}
    
    # Calculate available space
    target_tokens = max_seq_length - tokens_to_generate
    
    # Deep copy turns to avoid modifying original
    first_turn = copy.deepcopy(turns[0])
    first_turn_tokens = estimate_turn_tokens(first_turn, token_counter)
    
    # If even the first turn is too large, we have a problem
    if first_turn_tokens > target_tokens:
        LOG.error(f"First turn alone ({first_turn_tokens} tokens) exceeds target ({target_tokens} tokens)")
        return [first_turn], {
            "removed_turns": len(turns) - 1,
            "token_reduction": 100.0,
            "warning": "First turn exceeds token limit"
        }
    
    # Start building result with first turn
    result_turns = [first_turn]
    used_tokens = first_turn_tokens
    
    # Add recent turns in reverse order (most recent first)
    kept_indices = [0]  # Track which turn indices we're keeping
    
    for i in range(len(turns) - 1, 0, -1):  # Start from last turn, go backwards, skip first
        turn = copy.deepcopy(turns[i])
        turn_tokens = estimate_turn_tokens(turn, token_counter)
        
        # Check if adding this turn would exceed limit
        if used_tokens + turn_tokens <= target_tokens:
            result_turns.insert(1, turn)  # Insert after first turn
            used_tokens += turn_tokens
            kept_indices.insert(1, i)
        else:
            # Can't fit any more turns
            break
    
    # Calculate statistics
    original_tokens = sum(estimate_turn_tokens(t, token_counter) for t in turns)
    removed_turns = len(turns) - len(result_turns)
    token_reduction = ((original_tokens - used_tokens) / original_tokens * 100) if original_tokens > 0 else 0
    
    # Log the truncation
    if removed_turns > 0:
        LOG.info(f"First-and-recent truncation: {len(turns)} → {len(result_turns)} turns")
        LOG.info(f"Kept turns: {kept_indices}")
        LOG.info(f"Token reduction: {original_tokens} → {used_tokens} ({token_reduction:.1f}%)")
    else:
        LOG.debug(f"No truncation needed: {used_tokens} tokens <= {target_tokens} target")
    
    stats = {
        "original_turns": len(turns),
        "kept_turns": len(result_turns),
        "removed_turns": removed_turns,
        "original_tokens": original_tokens,
        "final_tokens": used_tokens,
        "token_reduction": token_reduction,
        "kept_indices": kept_indices
    }
    
    return result_turns, stats


def get_truncation_preview(
    turns: List[Dict], 
    max_seq_length: int, 
    tokens_to_generate: int,
    token_counter: Optional['TokenCounter'] = None
) -> str:
    """
    Get a preview of what would be truncated without actually doing it.
    
    Useful for logging/debugging.
    """
    _, stats = first_and_recent_truncate(turns, max_seq_length, tokens_to_generate, token_counter)
    
    if stats["removed_turns"] == 0:
        return "No truncation needed"
    
    preview = f"Would truncate {stats['removed_turns']} turns:\n"
    preview += f"  Keep: Turn 0 (problem statement)\n"
    
    # Show which turns would be kept
    for idx in stats["kept_indices"][1:]:  # Skip first turn
        preview += f"  Keep: Turn {idx}"
        if idx == len(turns) - 1:
            preview += " (most recent)"
        preview += "\n"
    
    # Show which turns would be removed
    removed_indices = [i for i in range(len(turns)) if i not in stats["kept_indices"]]
    if removed_indices:
        preview += f"  Remove: Turns {removed_indices}\n"
    
    preview += f"  Token reduction: {stats['token_reduction']:.1f}%"
    
    return preview
