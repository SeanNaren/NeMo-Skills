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
Bookend truncation strategy for dialogue history.

This module implements the truncation pattern that was accidentally discovered
when summarization failed - keeping only the first and last turns, which 
surprisingly led to better performance.
"""

import logging
from typing import List, Dict
import copy

LOG = logging.getLogger(__name__)


def estimate_dialogue_tokens(turns: List[Dict]) -> int:
    """Estimate the number of tokens in a dialogue.
    
    Simple estimation: ~1 token per 4 characters
    """
    total_chars = 0
    for turn in turns:
        if isinstance(turn, dict):
            for key, value in turn.items():
                if value and isinstance(value, str):
                    total_chars += len(value)
                elif value and isinstance(value, dict):
                    # For tool_call dictionaries
                    total_chars += len(str(value))
    
    return total_chars // 4  # Rough estimation


def bookend_truncate_dialogue_history(turns: List[dict], max_seq_length: int, tokens_to_generate: int) -> List[dict]:
    """Truncate dialogue using bookend strategy - keep first and last turns only.
    
    This implements the pattern that emerged when summarization failed:
    - Always keep the problem statement (turn 0)
    - Keep only the most recent 1-2 turns
    - Remove all middle turns
    
    Args:
        turns: List of dialogue turns
        max_seq_length: Maximum context length in tokens
        tokens_to_generate: Tokens to reserve for the next generation
        
    Returns:
        Truncated list of turns
    """
    if not turns:
        return turns
    
    target_tokens = max_seq_length - tokens_to_generate
    current_tokens = estimate_dialogue_tokens(turns)
    
    LOG.info(f"Bookend truncation: {current_tokens} tokens, target: {target_tokens}")
    
    if current_tokens <= target_tokens:
        LOG.info("No truncation needed")
        return turns
    
    # Deep copy to avoid modifying the original
    result_turns = []
    
    # Always keep the first turn (problem statement)
    result_turns.append(copy.deepcopy(turns[0]))
    LOG.info("Kept turn 0 (problem statement)")
    
    # Determine how many recent turns to keep
    if len(turns) <= 1:
        return result_turns
    
    # Check if we should include the second-to-last turn
    include_penultimate = False
    if len(turns) >= 2:
        last_turn = turns[-1]
        # If last turn has tool output, include the assistant request that triggered it
        if isinstance(last_turn, dict) and last_turn.get('tool_output'):
            include_penultimate = True
    
    # Add recent turns
    if include_penultimate and len(turns) >= 3:
        result_turns.append(copy.deepcopy(turns[-2]))
        LOG.info(f"Kept turn {len(turns)-2} (assistant request before tool output)")
    
    # Always include the last turn
    result_turns.append(copy.deepcopy(turns[-1]))
    LOG.info(f"Kept turn {len(turns)-1} (most recent)")
    
    # Log the reduction
    original_turns = len(turns)
    final_turns = len(result_turns)
    removed_turns = original_turns - final_turns
    
    final_tokens = estimate_dialogue_tokens(result_turns)
    token_reduction = ((current_tokens - final_tokens) / current_tokens * 100) if current_tokens > 0 else 0
    
    LOG.info(f"Bookend truncation complete: {original_turns} → {final_turns} turns (removed {removed_turns} middle turns)")
    LOG.info(f"Token reduction: {current_tokens} → {final_tokens} ({token_reduction:.1f}% reduction)")
    
    # Warn if still over limit
    if final_tokens > target_tokens:
        LOG.warning(f"Still over token limit after bookend truncation: {final_tokens} > {target_tokens}")
        LOG.warning("Consider reducing problem statement or implementing further truncation")
    
    return result_turns


def smart_bookend_truncate(turns: List[dict], max_seq_length: int, tokens_to_generate: int) -> List[dict]:
    """Smart bookend truncation with fallback to standard truncation if needed.
    
    This tries bookend truncation first, but if the first and last turns alone
    exceed the token limit, it falls back to keeping just the last turn(s).
    
    Args:
        turns: List of dialogue turns
        max_seq_length: Maximum context length in tokens
        tokens_to_generate: Tokens to reserve for the next generation
        
    Returns:
        Truncated list of turns
    """
    # First try standard bookend truncation
    result = bookend_truncate_dialogue_history(turns, max_seq_length, tokens_to_generate)
    
    target_tokens = max_seq_length - tokens_to_generate
    result_tokens = estimate_dialogue_tokens(result)
    
    if result_tokens <= target_tokens:
        return result
    
    # If still too long, try keeping only the last turn(s)
    LOG.warning("Bookend truncation still exceeds limit, trying last turns only")
    
    result_turns = []
    
    # Start with just the last turn
    if turns:
        result_turns = [copy.deepcopy(turns[-1])]
        
        # If last turn is tool output and we have space, add the request
        if (len(turns) >= 2 and 
            turns[-1].get('tool_output') and 
            estimate_dialogue_tokens([turns[-2], turns[-1]]) <= target_tokens):
            result_turns = [copy.deepcopy(turns[-2]), copy.deepcopy(turns[-1])]
    
    final_tokens = estimate_dialogue_tokens(result_turns)
    LOG.info(f"Final truncation: kept last {len(result_turns)} turn(s), ~{final_tokens} tokens")
    
    return result_turns


# For easy import and use
__all__ = ['bookend_truncate_dialogue_history', 'smart_bookend_truncate']
