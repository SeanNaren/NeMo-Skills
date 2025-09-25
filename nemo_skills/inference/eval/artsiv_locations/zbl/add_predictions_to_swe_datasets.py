#!/usr/bin/env python3
"""
Script to add prediction information to SWE-bench datasets.

This script reads a SWE-bench output.jsonl file (with Artsiv results),
extracts the final assistant response (reasoning trace and predicted locations) from the turns,
and adds this information to the problem_statement field to show what the model actually generated.

Usage:
    python add_predictions_to_swe_datasets.py --input-file INPUT_FILE --output-file OUTPUT_FILE

Args:
    --input-file: Path to the input output.jsonl file (Artsiv results)
    --output-file: Path to the output JSONL file
"""

import argparse
import json
import os
import sys
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add the parent directory to sys.path to import the utils module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'artsiv_utils'))
from patch_processor import PatchProcessor


def extract_locations_from_assistant_response(assistant_text: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Extract location information from the assistant's final response.
    
    Args:
        assistant_text: The assistant's response text
        
    Returns:
        Tuple of (locations list, reasoning trace)
    """
    locations = []
    reasoning_trace = assistant_text
    
    # Look for <locations> tags
    location_pattern = r'<locations>\s*(.*?)\s*</locations>'
    location_match = re.search(location_pattern, assistant_text, re.DOTALL | re.IGNORECASE)
    
    if location_match:
        location_content = location_match.group(1).strip()
        
        # Remove the locations block from reasoning trace
        reasoning_trace = re.sub(location_pattern, '', assistant_text, flags=re.DOTALL | re.IGNORECASE).strip()
        
        # Parse the location content
        # Look for file:line patterns
        file_line_patterns = [
            r'([^\s]+\.py):(\d+)-(\d+)',  # file.py:start-end
            r'([^\s]+\.py):(\d+)',        # file.py:line (assume single line)
            r'([^\s]+):(\d+)-(\d+)',      # file:start-end (any extension)
            r'([^\s]+):(\d+)',            # file:line (any extension)
        ]
        
        for pattern in file_line_patterns:
            matches = re.findall(pattern, location_content)
            for match in matches:
                if len(match) == 3:  # file, start, end
                    file_path, start_line, end_line = match
                    locations.append({
                        'file_path': file_path,
                        'start_line': int(start_line),
                        'end_line': int(end_line)
                    })
                elif len(match) == 2:  # file, single line
                    file_path, line = match
                    locations.append({
                        'file_path': file_path,
                        'start_line': int(line),
                        'end_line': int(line)
                    })
    
    # If no locations found in <locations> tags, try to find them in the text
    if not locations:
        # Look for common location formats in the text
        location_patterns = [
            r'(?:modify|edit|change|update)\s+([^\s]+\.py)\s+(?:line|lines)\s+(\d+)(?:-(\d+))?',
            r'([^\s]+\.py)\s*:\s*(?:line|lines)\s+(\d+)(?:-(\d+))?',
            r'([^\s]+\.py)\s+(?:at\s+)?(?:line|lines)\s+(\d+)(?:-(\d+))?',
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, assistant_text, re.IGNORECASE)
            for match in matches:
                if len(match) == 3 and match[2]:  # file, start, end
                    file_path, start_line, end_line = match
                    locations.append({
                        'file_path': file_path,
                        'start_line': int(start_line),
                        'end_line': int(end_line)
                    })
                elif len(match) >= 2:  # file, single line
                    file_path, line = match[0], match[1]
                    locations.append({
                        'file_path': file_path,
                        'start_line': int(line),
                        'end_line': int(line)
                    })
    
    return locations, reasoning_trace


def get_final_assistant_response(turns: List[Dict[str, Any]]) -> Optional[str]:
    """
    Extract the final assistant response from the turns.
    
    Args:
        turns: List of conversation turns
        
    Returns:
        The final assistant response text or None if not found
    """
    if not turns:
        return None
    
    # Look for the last turn with assistant content
    for turn in reversed(turns):
        assistant_content = turn.get('assistant', '').strip()
        if assistant_content:
            return assistant_content
    
    return None


def format_prediction_prompt(locations: List[Dict[str, Any]], reasoning_trace: str, 
                           status: str, num_turns: int) -> str:
    """
    Format the predictions into a helpful prompt to add to the problem statement.
    
    Args:
        locations: List of predicted location dictionaries
        reasoning_trace: The model's reasoning trace
        status: The generation status (success/failed)
        num_turns: Number of turns in the conversation
        
    Returns:
        Formatted string with prediction information
    """
    prompt_parts = ["\n\n--- MODEL PREDICTION ---"]
    prompt_parts.append(f"Generation Status: {status.upper()}")
    prompt_parts.append(f"Number of Turns: {num_turns}")
    
    if locations:
        # Group locations by file
        files_dict = {}
        for loc in locations:
            file_path = loc['file_path']
            if file_path not in files_dict:
                files_dict[file_path] = []
            files_dict[file_path].append(f"L{loc['start_line']}-L{loc['end_line']}")
        
        prompt_parts.append("\nPredicted Locations:")
        for file_path, line_ranges in files_dict.items():
            line_ranges_str = ", ".join(line_ranges)
            prompt_parts.append(f"• {file_path}: {line_ranges_str}")
    else:
        prompt_parts.append("\nPredicted Locations: None found")
    
    if reasoning_trace:
        # Truncate very long reasoning traces
        max_reasoning_length = 2000
        if len(reasoning_trace) > max_reasoning_length:
            reasoning_trace = reasoning_trace[:max_reasoning_length] + "... [truncated]"
        
        prompt_parts.append(f"\nModel Reasoning Trace:")
        prompt_parts.append(f"```")
        prompt_parts.append(reasoning_trace)
        prompt_parts.append(f"```")
    
    return "\n".join(prompt_parts)


def process_output_jsonl_file(input_file: Path, output_file: Path) -> Dict[str, int]:
    """
    Process an output JSONL file and add prediction information to the problem_statement field.
    
    Args:
        input_file: Path to input output.jsonl file
        output_file: Path to output JSONL file
        
    Returns:
        Dictionary with processing statistics
    """
    stats = {
        'total_instances': 0,
        'instances_with_turns': 0,
        'instances_with_final_response': 0,
        'instances_with_predicted_locations': 0,
        'total_predicted_locations': 0,
        'instances_with_problem_statement': 0,
        'successful_instances': 0,
        'failed_instances': 0
    }
    
    print(f"Processing {input_file} -> {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse JSON line
                instance = json.loads(line.strip())
                stats['total_instances'] += 1
                
                # Get basic info
                status = instance.get('status', 'unknown')
                turns = instance.get('turns', [])
                num_turns = len(turns)
                
                if status == 'success':
                    stats['successful_instances'] += 1
                elif status == 'failed':
                    stats['failed_instances'] += 1
                
                if turns:
                    stats['instances_with_turns'] += 1
                    
                    # Extract final assistant response
                    final_response = get_final_assistant_response(turns)
                    
                    if final_response:
                        stats['instances_with_final_response'] += 1
                        
                        # Extract locations and reasoning from the response
                        predicted_locations, reasoning_trace = extract_locations_from_assistant_response(final_response)
                        
                        if predicted_locations:
                            stats['instances_with_predicted_locations'] += 1
                            stats['total_predicted_locations'] += len(predicted_locations)
                        
                        # Add prediction information to problem_statement
                        problem_statement = instance.get('problem_statement', '')
                        if problem_statement:
                            stats['instances_with_problem_statement'] += 1
                            prediction_prompt = format_prediction_prompt(
                                predicted_locations, reasoning_trace, status, num_turns
                            )
                            instance['problem_statement'] = problem_statement + prediction_prompt
                
                # Write the updated instance
                outfile.write(json.dumps(instance) + '\n')
                
                # Progress indicator
                if line_num % 100 == 0:
                    print(f"  Processed {line_num} instances...")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Add prediction information to SWE-bench datasets from Artsiv output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--input-file',
        type=str,
        required=True,
        help='Path to the input output.jsonl file (Artsiv results)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='Path to the output JSONL file'
    )
    parser.add_argument(
        '--max-reasoning-length',
        type=int,
        default=2000,
        help='Maximum length of reasoning trace to include (default: 2000 chars)'
    )
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    
    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist")
        return
    
    # Create output directory if it doesn't exist
    output_dir = output_file.parent
    if output_dir != Path('.'):
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # Process the file
    stats = process_output_jsonl_file(input_file, output_file)
    
    # Print file statistics
    print(f"\nResults for {input_file.name}:")
    print(f"  Total instances: {stats['total_instances']}")
    print(f"  Successful instances: {stats['successful_instances']}")
    print(f"  Failed instances: {stats['failed_instances']}")
    print(f"  Instances with turns: {stats['instances_with_turns']}")
    print(f"  Instances with final response: {stats['instances_with_final_response']}")
    print(f"  Instances with predicted locations: {stats['instances_with_predicted_locations']}")
    print(f"  Total predicted locations: {stats['total_predicted_locations']}")
    print(f"  Instances with problem_statement updated: {stats['instances_with_problem_statement']}")
    
    if stats['instances_with_final_response'] > 0:
        avg_locations = stats['total_predicted_locations'] / stats['instances_with_final_response']
        print(f"  Average predicted locations per instance: {avg_locations:.2f}")
    
    if stats['instances_with_turns'] > 0:
        response_rate = (stats['instances_with_final_response'] / stats['instances_with_turns'] * 100)
        print(f"\nFinal response extraction rate: {response_rate:.1f}%")
    
    if stats['instances_with_final_response'] > 0:
        location_rate = (stats['instances_with_predicted_locations'] / stats['instances_with_final_response'] * 100)
        print(f"Location prediction rate: {location_rate:.1f}%")
    
    if stats['instances_with_predicted_locations'] > 0:
        update_rate = (stats['instances_with_problem_statement'] / stats['instances_with_predicted_locations'] * 100)
        print(f"Problem statement update rate: {update_rate:.1f}%")
    
    print(f"\nOutput saved to: {output_file}")


if __name__ == '__main__':
    main()
