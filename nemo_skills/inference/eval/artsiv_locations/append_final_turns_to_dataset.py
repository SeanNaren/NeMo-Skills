#!/usr/bin/env python3
"""
Script to add edit_locations data to SWE-bench dataset instances.

This script reads:
1. Original SWE-bench dataset file (default_bp.jsonl)
2. Artsiv output.jsonl file (with trajectories)

And creates a new dataset where each instance gets an 'edit_locations' key containing:
- raw_response: The full assistant_raw_w_think response
- reasoning: The extracted reasoning content from <think> tags  
- locations: Parsed location information from the response

Usage:
    python append_final_turns_to_dataset.py --original-file ORIGINAL_FILE --trajectories-file TRAJECTORIES_FILE --output-file OUTPUT_FILE

Args:
    --original-file: Path to the original JSONL dataset file (e.g., default_bp.jsonl)
    --trajectories-file: Path to the Artsiv output.jsonl file with trajectories
    --output-file: Path to the output JSONL file
"""

import argparse
import json
import re
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add the parent directory to sys.path to import the utils module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'artsiv_utils'))
from patch_processor import PatchProcessor

def get_final_assistant_raw_w_think(turns: List[Dict[str, Any]]) -> Optional[str]:
    """
    Extract the final assistant_raw_w_think response from the turns.
    
    Args:
        turns: List of conversation turns
        
    Returns:
        The final assistant_raw_w_think response text with content extracted from <think> tags or None if not found
    """
    if not turns:
        return None
    
    # Look for the last turn with assistant_raw_w_think content
    for turn in reversed(turns):
        assistant_raw_w_think = turn.get('assistant_raw_w_think', '').strip()
        if assistant_raw_w_think:
            # Extract content from <think></think> tags
            think_content = _extract_tag_content(assistant_raw_w_think, "think")
            if think_content:
                return think_content
            else:
                # If no think tags, return the content as is
                return assistant_raw_w_think
    
    return None


def _extract_tag_content(text: str, tag_name: str) -> str:
    """Extract content from XML-style tags with fallback for unclosed tags (from dialog_processor.py)."""
    # Try properly closed tag first
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()

    # Fallback: unclosed tag - extract everything after the tag
    pattern = f"<{tag_name}>\\s*(.*?)$"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()

    return None


def extract_locations_from_response(response_text: str) -> List[Dict[str, Any]]:
    """
    Extract location information from the assistant's response using the same logic as DialogProcessor._extract_locations.
    
    Args:
        response_text: The assistant's response text
        
    Returns:
        List of location dictionaries
    """
    # Use unified tag extraction (copied from dialog_processor.py)
    locations_text = _extract_tag_content(response_text, "locations")
    if not locations_text:
        return []

    # Parse location lines (copied from dialog_processor.py)
    locations = []
    for line in locations_text.split("\n"):
        line = line.strip()
        if line and not line.startswith("#"):
            # Try to match range format first: file:L<start>-L<end>
            location_match = re.match(r"([^:]+):L(\d+)-L(\d+)", line)
            if location_match:
                file_path, start_line, end_line = location_match.groups()
                locations.append({
                    "file_path": file_path,
                    "start_line": int(start_line),
                    "end_line": int(end_line),
                    "raw": line
                })
            else:
                # Try to match single line format: file:L<line>
                single_line_match = re.match(r"([^:]+):L(\d+)", line)
                if single_line_match:
                    file_path, line_num = single_line_match.groups()
                    locations.append({
                        "file_path": file_path,
                        "start_line": int(line_num),
                        "end_line": int(line_num),  # Same as start for single line
                        "raw": line
                    })
                else:
                    # Keep raw line even if we can't parse it
                    locations.append({"raw": line})

    # Return only the successfully parsed locations (filter out ones with just 'raw')
    return [loc for loc in locations if 'file_path' in loc]


def format_locations_prompt(locations: List[Dict[str, Any]]) -> str:
    """
    Format the locations into a helpful prompt (using the same template as add_locations_to_swe_datasets.py).
    
    Args:
        locations: List of location dictionaries
        
    Returns:
        Formatted string with location information
    """
    if not locations:
        return ""
    
    # Group locations by file
    files_dict = {}
    for loc in locations:
        file_path = loc['file_path']
        if file_path not in files_dict:
            files_dict[file_path] = []
        files_dict[file_path].append(f"L{loc['start_line']}-L{loc['end_line']}")
    
    # Format the prompt using the exact template
    prompt_parts = ["\n\n--- EDIT LOCATIONS ---"]
    prompt_parts.append("The following locations in the codebase need to be modified to address this issue:")
    
    for file_path, line_ranges in files_dict.items():
        line_ranges_str = ", ".join(line_ranges)
        prompt_parts.append(f"• {file_path}: {line_ranges_str}")
    
    prompt_parts.append("Focus your changes on these specific locations when generating the patch.")
    
    return "\n".join(prompt_parts)


def create_edit_locations_data(final_turn: str, full_response: str) -> Dict[str, Any]:
    """
    Create the edit_locations data structure with raw response, reasoning, and locations.
    
    Args:
        final_turn: The reasoning content from think tags
        full_response: The full assistant_raw_w_think response
        
    Returns:
        Dictionary with raw_response, reasoning, and locations fields
    """
    # Extract and parse predicted locations from the full response
    predicted_locations = extract_locations_from_response(full_response)
    
    return {
        "raw_response": full_response,
        "reasoning": final_turn,
        "locations": predicted_locations
    }


def load_trajectories_by_instance_id(trajectories_file: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load trajectories from output.jsonl file indexed by instance_id.
    
    Args:
        trajectories_file: Path to the trajectories file
        
    Returns:
        Dictionary mapping instance_id to trajectory data
    """
    trajectories = {}
    
    print(f"Loading trajectories from {trajectories_file}")
    
    with open(trajectories_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                instance = json.loads(line.strip())
                instance_id = instance.get('instance_id')
                
                if instance_id:
                    # Extract trajectory information
                    status = instance.get('status', 'unknown')
                    turns = instance.get('turns', [])
                    num_turns = len(turns)
                    
                    # Get final assistant response from assistant_raw_w_think
                    final_turn = get_final_assistant_raw_w_think(turns)
                    
                    # Also get the full response for location extraction
                    full_response = None
                    for turn in reversed(turns):
                        full_response = turn.get('assistant_raw_w_think', '').strip()
                        if full_response:
                            break
                    
                    trajectories[instance_id] = {
                        'status': status,
                        'num_turns': num_turns,
                        'final_turn': final_turn,
                        'full_response': full_response,
                        'turns': turns
                    }
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    print(f"Loaded trajectories for {len(trajectories)} instances")
    return trajectories


def append_final_turns_to_dataset(original_file: Path, trajectories_file: Path, 
                                 output_file: Path) -> Dict[str, int]:
    """
    Add edit_locations data to instances in the original dataset.
    
    Args:
        original_file: Path to original dataset file
        trajectories_file: Path to trajectories file
        output_file: Path to output file
        
    Returns:
        Dictionary with processing statistics
    """
    stats = {
        'total_instances': 0,
        'instances_with_trajectories': 0,
        'instances_with_final_turns': 0,
        'instances_updated': 0,
        'successful_trajectories': 0,
        'failed_trajectories': 0,
        'missing_trajectories': 0
    }
    
    # Load trajectories
    trajectories = load_trajectories_by_instance_id(trajectories_file)
    
    print(f"Processing {original_file} with trajectories -> {output_file}")
    
    with open(original_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse original instance
                instance = json.loads(line.strip())
                stats['total_instances'] += 1
                
                instance_id = instance.get('instance_id')
                problem_statement = instance.get('problem_statement', '')
                
                # Get trajectory data
                trajectory_data = trajectories.get(instance_id)
                
                if trajectory_data:
                    stats['instances_with_trajectories'] += 1
                    
                    # Track status
                    if trajectory_data['status'] == 'success':
                        stats['successful_trajectories'] += 1
                    elif trajectory_data['status'] == 'failed':
                        stats['failed_trajectories'] += 1
                    
                    # Get final turn
                    final_turn = trajectory_data.get('final_turn')
                    
                    if final_turn:
                        stats['instances_with_final_turns'] += 1
                    
                    # Add edit_locations data as a separate key
                    if final_turn:
                        edit_locations_data = create_edit_locations_data(
                            final_turn,
                            trajectory_data.get('full_response', '')
                        )
                        
                        instance['edit_locations'] = edit_locations_data
                        stats['instances_updated'] += 1
                
                else:
                    stats['missing_trajectories'] += 1
                    if instance_id:
                        print(f"Warning: No trajectory found for instance_id: {instance_id}")
                
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
        description="Add edit_locations data to SWE-bench dataset instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--original-file',
        type=str,
        required=True,
        help='Path to the original JSONL dataset file (e.g., default_bp.jsonl)'
    )
    parser.add_argument(
        '--trajectories-file',
        type=str,
        required=True,
        help='Path to the Artsiv output.jsonl file with trajectories'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='Path to the output JSONL file'
    )
    
    args = parser.parse_args()
    
    original_file = Path(args.original_file)
    trajectories_file = Path(args.trajectories_file)
    output_file = Path(args.output_file)
    
    # Check if input files exist
    if not original_file.exists():
        print(f"Error: Original file {original_file} does not exist")
        return
    
    if not trajectories_file.exists():
        print(f"Error: Trajectories file {trajectories_file} does not exist")
        return
    
    # Create output directory if it doesn't exist
    output_dir = output_file.parent
    if output_dir != Path('.'):
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # Process the datasets
    stats = append_final_turns_to_dataset(original_file, trajectories_file, output_file)
    
    # Print statistics
    print(f"\nProcessing Results:")
    print(f"  Total instances: {stats['total_instances']}")
    print(f"  Instances with trajectories: {stats['instances_with_trajectories']}")
    print(f"  Instances with final turns: {stats['instances_with_final_turns']}")
    print(f"  Instances updated: {stats['instances_updated']}")
    print(f"  Missing trajectories: {stats['missing_trajectories']}")
    print(f"  Successful trajectories: {stats['successful_trajectories']}")
    print(f"  Failed trajectories: {stats['failed_trajectories']}")
    
    if stats['total_instances'] > 0:
        coverage_rate = (stats['instances_with_trajectories'] / stats['total_instances'] * 100)
        print(f"\nTrajectory coverage: {coverage_rate:.1f}%")
        
        if stats['instances_with_trajectories'] > 0:
            success_rate = (stats['successful_trajectories'] / stats['instances_with_trajectories'] * 100)
            print(f"Trajectory success rate: {success_rate:.1f}%")
            
            final_turn_rate = (stats['instances_with_final_turns'] / stats['instances_with_trajectories'] * 100)
            print(f"Final turn extraction rate: {final_turn_rate:.1f}%")
    
    print(f"\nOutput saved to: {output_file}")


if __name__ == '__main__':
    main()
