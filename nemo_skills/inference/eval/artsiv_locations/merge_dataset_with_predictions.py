#!/usr/bin/env python3
"""
Script to merge original SWE-bench dataset with Artsiv predictions.

This script reads:
1. Original SWE-bench dataset (with problem statements)
2. Artsiv output.jsonl file (with predictions and reasoning)

And creates a merged dataset that includes both the original problem statement
and the model's reasoning trace and predicted locations.

Usage:
    python merge_dataset_with_predictions.py --original-file ORIGINAL_FILE --predictions-file PREDICTIONS_FILE --output-file OUTPUT_FILE

Args:
    --original-file: Path to the original JSONL dataset file
    --predictions-file: Path to the Artsiv output.jsonl file with predictions
    --output-file: Path to the output merged JSONL file
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


def format_locations_prompt(locations: List[Dict[str, Any]], title: str) -> str:
    """
    Format locations into a prompt section.
    
    Args:
        locations: List of location dictionaries
        title: Title for this section (e.g., "GROUND TRUTH LOCATIONS", "PREDICTED LOCATIONS")
        
    Returns:
        Formatted string with location information
    """
    if not locations:
        return f"\n\n--- {title} ---\nNo locations found."
    
    # Group locations by file
    files_dict = {}
    for loc in locations:
        file_path = loc['file_path']
        if file_path not in files_dict:
            files_dict[file_path] = []
        files_dict[file_path].append(f"L{loc['start_line']}-L{loc['end_line']}")
    
    # Format the prompt
    prompt_parts = [f"\n\n--- {title} ---"]
    
    for file_path, line_ranges in files_dict.items():
        line_ranges_str = ", ".join(line_ranges)
        prompt_parts.append(f"• {file_path}: {line_ranges_str}")
    
    return "\n".join(prompt_parts)


def format_merged_prompt(ground_truth_locations: List[Dict[str, Any]], 
                        predicted_locations: List[Dict[str, Any]], 
                        reasoning_trace: str, 
                        status: str, 
                        num_turns: int,
                        max_reasoning_length: int = 2000) -> str:
    """
    Format the merged information into a comprehensive prompt.
    
    Args:
        ground_truth_locations: List of ground truth location dictionaries
        predicted_locations: List of predicted location dictionaries
        reasoning_trace: The model's reasoning trace
        status: The generation status (success/failed)
        num_turns: Number of turns in the conversation
        max_reasoning_length: Maximum length of reasoning trace
        
    Returns:
        Formatted string with all information
    """
    prompt_parts = []
    
    # Ground truth locations
    gt_prompt = format_locations_prompt(ground_truth_locations, "GROUND TRUTH LOCATIONS")
    prompt_parts.append(gt_prompt)
    
    # Model prediction info
    prompt_parts.append(f"\n\n--- MODEL PREDICTION ---")
    prompt_parts.append(f"Generation Status: {status.upper()}")
    prompt_parts.append(f"Number of Turns: {num_turns}")
    
    # Predicted locations
    if predicted_locations:
        files_dict = {}
        for loc in predicted_locations:
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
    
    # Reasoning trace
    if reasoning_trace:
        # Truncate very long reasoning traces
        if len(reasoning_trace) > max_reasoning_length:
            reasoning_trace = reasoning_trace[:max_reasoning_length] + "... [truncated]"
        
        prompt_parts.append(f"\nModel Reasoning Trace:")
        prompt_parts.append(f"```")
        prompt_parts.append(reasoning_trace)
        prompt_parts.append(f"```")
    
    return "\n".join(prompt_parts)


def load_predictions_by_instance_id(predictions_file: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load predictions from output.jsonl file indexed by instance_id.
    
    Args:
        predictions_file: Path to the predictions file
        
    Returns:
        Dictionary mapping instance_id to prediction data
    """
    predictions = {}
    
    print(f"Loading predictions from {predictions_file}")
    
    with open(predictions_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                instance = json.loads(line.strip())
                instance_id = instance.get('instance_id')
                
                if instance_id:
                    # Extract prediction information
                    status = instance.get('status', 'unknown')
                    turns = instance.get('turns', [])
                    num_turns = len(turns)
                    
                    # Get final assistant response
                    final_response = get_final_assistant_response(turns)
                    predicted_locations = []
                    reasoning_trace = ""
                    
                    if final_response:
                        predicted_locations, reasoning_trace = extract_locations_from_assistant_response(final_response)
                    
                    predictions[instance_id] = {
                        'status': status,
                        'num_turns': num_turns,
                        'predicted_locations': predicted_locations,
                        'reasoning_trace': reasoning_trace,
                        'final_response': final_response
                    }
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    print(f"Loaded predictions for {len(predictions)} instances")
    return predictions


def merge_datasets(original_file: Path, predictions_file: Path, output_file: Path,
                  max_reasoning_length: int = 2000, exclude_new_files: bool = True) -> Dict[str, int]:
    """
    Merge original dataset with predictions.
    
    Args:
        original_file: Path to original dataset file
        predictions_file: Path to predictions file
        output_file: Path to output merged file
        max_reasoning_length: Maximum length of reasoning trace
        exclude_new_files: Whether to exclude locations from newly created files
        
    Returns:
        Dictionary with processing statistics
    """
    stats = {
        'total_instances': 0,
        'instances_with_predictions': 0,
        'instances_with_predicted_locations': 0,
        'instances_with_ground_truth': 0,
        'instances_merged_successfully': 0,
        'successful_predictions': 0,
        'failed_predictions': 0
    }
    
    # Load predictions
    predictions = load_predictions_by_instance_id(predictions_file)
    
    print(f"Merging {original_file} with predictions -> {output_file}")
    
    with open(original_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse original instance
                instance = json.loads(line.strip())
                stats['total_instances'] += 1
                
                instance_id = instance.get('instance_id')
                problem_statement = instance.get('problem_statement', '')
                
                # Extract ground truth locations from patch
                ground_truth_locations = []
                patch = instance.get('patch', '')
                if patch:
                    ground_truth_locations = PatchProcessor.extract_locations_from_patch(
                        patch, exclude_new_files=exclude_new_files
                    )
                    if ground_truth_locations:
                        stats['instances_with_ground_truth'] += 1
                
                # Get prediction data
                prediction_data = predictions.get(instance_id, {})
                if prediction_data:
                    stats['instances_with_predictions'] += 1
                    
                    if prediction_data.get('predicted_locations'):
                        stats['instances_with_predicted_locations'] += 1
                    
                    if prediction_data.get('status') == 'success':
                        stats['successful_predictions'] += 1
                    elif prediction_data.get('status') == 'failed':
                        stats['failed_predictions'] += 1
                    
                    # Create merged prompt
                    if problem_statement:
                        merged_prompt = format_merged_prompt(
                            ground_truth_locations,
                            prediction_data.get('predicted_locations', []),
                            prediction_data.get('reasoning_trace', ''),
                            prediction_data.get('status', 'unknown'),
                            prediction_data.get('num_turns', 0),
                            max_reasoning_length
                        )
                        
                        instance['problem_statement'] = problem_statement + merged_prompt
                        stats['instances_merged_successfully'] += 1
                
                # Write the merged instance
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
        description="Merge original SWE-bench dataset with Artsiv predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--original-file',
        type=str,
        required=True,
        help='Path to the original JSONL dataset file'
    )
    parser.add_argument(
        '--predictions-file',
        type=str,
        required=True,
        help='Path to the Artsiv output.jsonl file with predictions'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='Path to the output merged JSONL file'
    )
    parser.add_argument(
        '--max-reasoning-length',
        type=int,
        default=2000,
        help='Maximum length of reasoning trace to include (default: 2000 chars)'
    )
    parser.add_argument(
        '--include-new-files',
        action='store_true',
        help='Include locations from newly created files in ground truth (default: exclude them)'
    )
    
    args = parser.parse_args()
    
    original_file = Path(args.original_file)
    predictions_file = Path(args.predictions_file)
    output_file = Path(args.output_file)
    
    # Check if input files exist
    if not original_file.exists():
        print(f"Error: Original file {original_file} does not exist")
        return
    
    if not predictions_file.exists():
        print(f"Error: Predictions file {predictions_file} does not exist")
        return
    
    # Create output directory if it doesn't exist
    output_dir = output_file.parent
    if output_dir != Path('.'):
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # Merge the datasets
    exclude_new_files = not args.include_new_files
    stats = merge_datasets(original_file, predictions_file, output_file, 
                          args.max_reasoning_length, exclude_new_files)
    
    # Print statistics
    print(f"\nMerge Results:")
    print(f"  Total instances: {stats['total_instances']}")
    print(f"  Instances with predictions: {stats['instances_with_predictions']}")
    print(f"  Instances with predicted locations: {stats['instances_with_predicted_locations']}")
    print(f"  Instances with ground truth locations: {stats['instances_with_ground_truth']}")
    print(f"  Instances merged successfully: {stats['instances_merged_successfully']}")
    print(f"  Successful predictions: {stats['successful_predictions']}")
    print(f"  Failed predictions: {stats['failed_predictions']}")
    
    if stats['instances_with_predictions'] > 0:
        prediction_rate = (stats['instances_with_predictions'] / stats['total_instances'] * 100)
        print(f"\nPrediction coverage: {prediction_rate:.1f}%")
        
        success_rate = (stats['successful_predictions'] / stats['instances_with_predictions'] * 100)
        print(f"Prediction success rate: {success_rate:.1f}%")
        
        location_rate = (stats['instances_with_predicted_locations'] / stats['instances_with_predictions'] * 100)
        print(f"Location prediction rate: {location_rate:.1f}%")
    
    exclude_new_files_str = "excluded" if not args.include_new_files else "included"
    print(f"\nNew files were {exclude_new_files_str} from ground truth location extraction.")
    print(f"Output saved to: {output_file}")


if __name__ == '__main__':
    main()


