"""
Loop detection and prevention utilities for LocAgent.

This module helps prevent the agent from getting stuck in repetitive loops
where it generates the same tool call multiple times.
"""

from typing import List, Dict, Optional, Tuple
import json
from collections import Counter
import logging

LOG = logging.getLogger(__name__)


def detect_repetitive_tool_calls(generations: List[Dict], threshold: int = 3) -> Tuple[bool, Optional[Dict]]:
    """
    Detect if the agent is stuck in a loop generating the same tool call repeatedly.
    
    Args:
        generations: List of generation dictionaries from the agent
        threshold: Number of identical calls to consider it a loop (default: 3)
        
    Returns:
        Tuple of (is_loop_detected, loop_info)
        where loop_info contains details about the repeated call if a loop is detected
    """
    if len(generations) < threshold:
        return False, None
    
    # Extract tool calls from generations
    tool_calls = []
    for gen in generations:
        gen_text = gen.get('generation', '')
        if '<tool_call>' in gen_text and '</tool_call>' in gen_text:
            # Extract the JSON between tool_call tags
            start = gen_text.find('<tool_call>') + len('<tool_call>')
            end = gen_text.find('</tool_call>')
            if start < end:
                try:
                    tool_call_json = gen_text[start:end].strip()
                    # Normalize the JSON to handle formatting differences
                    tool_call_obj = json.loads(tool_call_json)
                    tool_call_normalized = json.dumps(tool_call_obj, sort_keys=True)
                    tool_calls.append(tool_call_normalized)
                except (json.JSONDecodeError, Exception) as e:
                    LOG.debug(f"Failed to parse tool call JSON: {e}")
                    tool_calls.append(gen_text[start:end].strip())
    
    if not tool_calls:
        return False, None
    
    # Count occurrences of each tool call
    call_counts = Counter(tool_calls)
    
    # Check if any call appears more than threshold times
    for call, count in call_counts.items():
        if count >= threshold:
            # Calculate what percentage of recent calls are this repeated call
            recent_window = min(10, len(tool_calls))  # Look at last 10 calls
            recent_calls = tool_calls[-recent_window:]
            recent_repetitions = recent_calls.count(call)
            
            loop_info = {
                'repeated_call': call,
                'total_repetitions': count,
                'recent_repetitions': recent_repetitions,
                'recent_window': recent_window,
                'loop_percentage': recent_repetitions / recent_window * 100
            }
            
            # Consider it a loop if more than 70% of recent calls are the same
            if loop_info['loop_percentage'] >= 70:
                return True, loop_info
    
    return False, None


def inject_loop_intervention(turns: List[Dict], loop_info: Dict) -> List[Dict]:
    """
    Inject a system message to help the agent break out of a loop.
    
    Args:
        turns: Current conversation turns
        loop_info: Information about the detected loop
        
    Returns:
        Modified turns with intervention message
    """
    # Parse the repeated call to provide specific guidance
    try:
        repeated_call = json.loads(loop_info['repeated_call'])
        tool_name = list(repeated_call.keys())[0]
        tool_params = repeated_call[tool_name]
        
        # Create specific guidance based on the tool
        if tool_name == 'view_file':
            file_path = tool_params.get('path', 'unknown')
            view_range = tool_params.get('view_range', [])
            
            intervention = f"""SYSTEM INTERVENTION: Loop detected! You have attempted to view '{file_path}' {loop_info['total_repetitions']} times with the same parameters.

The file appears to be too large or the output is being truncated. Please try a different approach:
1. View a specific section using line numbers (e.g., view_range: [1000, 1200])
2. Search for specific content using grep or find
3. Look at the file structure first with list_directory
4. Check if there's a more specific file related to your task

DO NOT repeat the same view_file command. Think of an alternative strategy."""
        else:
            intervention = f"""SYSTEM INTERVENTION: Loop detected! You have repeated the same {tool_name} command {loop_info['total_repetitions']} times.

This suggests the current approach isn't working. Please:
1. Analyze why the previous attempts didn't provide useful information
2. Try a completely different tool or approach
3. Break down the problem into smaller steps
4. Consider if you're looking in the wrong place

DO NOT repeat the same command. Think of an alternative strategy."""
    except Exception as e:
        LOG.debug(f"Failed to parse repeated call for specific guidance: {e}")
        intervention = f"""SYSTEM INTERVENTION: Loop detected! You have repeated the same command {loop_info['total_repetitions']} times.

Please try a different approach. The current strategy is not working."""
    
    # Add the intervention as a system turn
    intervention_turn = {
        'inputs': intervention,
        'tool_call': None,
        'tool_output': '',
        'assistant': ''  # Agent will respond to this
    }
    
    # Insert the intervention before the last turn
    modified_turns = turns.copy()
    if len(modified_turns) > 0:
        modified_turns.insert(-1, intervention_turn)
    else:
        modified_turns.append(intervention_turn)
    
    return modified_turns


def prevent_loop_generation(current_prompt: str, loop_info: Dict) -> str:
    """
    Modify the prompt to prevent the agent from generating the same tool call again.
    
    Args:
        current_prompt: The current prompt being sent to the model
        loop_info: Information about the detected loop
        
    Returns:
        Modified prompt that discourages repetition
    """
    try:
        repeated_call = json.loads(loop_info['repeated_call'])
        tool_name = list(repeated_call.keys())[0]
        
        loop_warning = f"""\n\nIMPORTANT: You have already tried {tool_name} with these exact parameters {loop_info['total_repetitions']} times. 
DO NOT repeat this command. You MUST try a different approach or tool.
Previous attempts have not yielded useful results - the output may be truncated or the approach may be wrong.
Think creatively about alternative ways to solve this problem.\n\n"""
        
        # Prepend the warning to the prompt
        return loop_warning + current_prompt
    except Exception as e:
        LOG.debug(f"Failed to create specific loop warning: {e}")
        return f"\n\nWARNING: Loop detected. Do not repeat the previous command.\n\n" + current_prompt


def analyze_loop_patterns(generations: List[Dict]) -> Dict:
    """
    Analyze patterns in tool call generations to identify potential issues.
    
    Args:
        generations: List of generation dictionaries
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {
        'total_generations': len(generations),
        'unique_calls': 0,
        'most_common_call': None,
        'repetition_ratio': 0.0,
        'potential_issues': []
    }
    
    if not generations:
        return analysis
    
    # Extract all tool calls
    tool_calls = []
    for gen in generations:
        gen_text = gen.get('generation', '')
        if '<tool_call>' in gen_text and '</tool_call>' in gen_text:
            start = gen_text.find('<tool_call>') + len('<tool_call>')
            end = gen_text.find('</tool_call>')
            if start < end:
                tool_calls.append(gen_text[start:end].strip())
    
    if not tool_calls:
        analysis['potential_issues'].append('No tool calls found in generations')
        return analysis
    
    # Analyze patterns
    call_counts = Counter(tool_calls)
    analysis['unique_calls'] = len(call_counts)
    
    if call_counts:
        most_common = call_counts.most_common(1)[0]
        analysis['most_common_call'] = most_common[0]
        analysis['repetition_ratio'] = most_common[1] / len(tool_calls)
        
        # Identify issues
        if analysis['repetition_ratio'] > 0.5:
            analysis['potential_issues'].append(f'High repetition: {most_common[1]}/{len(tool_calls)} calls are identical')
        
        if analysis['unique_calls'] == 1 and len(tool_calls) > 3:
            analysis['potential_issues'].append('All tool calls are identical - severe loop detected')
        
        # Check for truncation indicators
        for call in tool_calls:
            if 'view_file' in call and '[1, -1]' in call:
                analysis['potential_issues'].append('Attempting to view entire files - may cause truncation')
    
    return analysis
