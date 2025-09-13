#!/usr/bin/env python3
"""
Script to add location information to SWE-bench datasets.

This script reads SWE-bench dataset files (swe-lite, swe-verified, etc.),
extracts location information from the patches using the extract_locations_from_patch function,
and adds this information to the problem_statement field as a helpful prompt about where changes should be made.

Usage:
    python add_locations_to_swe_datasets.py [--input-dir INPUT_DIR] [--output-dir OUTPUT_DIR]

Args:
    --input-dir: Directory containing the input JSONL files (default: current directory)
    --output-dir: Directory to save the output files with locations (default: current directory)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add the parent directory to sys.path to import the utils module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'locagent_utils'))
from utils import extract_locations_from_patch


def format_locations_prompt(locations: List[Dict[str, Any]]) -> str:
    """
    Format the locations into a helpful prompt to add to the problem statement.
    
    Args:
        locations: List of location dictionaries from extract_locations_from_patch
        
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
    
    # Format the prompt
    prompt_parts = ["\n\n--- EDIT LOCATIONS ---"]
    prompt_parts.append("The following locations in the codebase need to be modified to address this issue:")
    
    for file_path, line_ranges in files_dict.items():
        line_ranges_str = ", ".join(line_ranges)
        prompt_parts.append(f"• {file_path}: {line_ranges_str}")
    
    prompt_parts.append("Focus your changes on these specific locations when generating the patch.")
    
    return "\n".join(prompt_parts)


def process_jsonl_file(input_file: Path, output_file: Path, exclude_new_files: bool = True) -> Dict[str, int]:
    """
    Process a JSONL file and add location information to the problem_statement field.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        exclude_new_files: Whether to exclude locations from newly created files
        
    Returns:
        Dictionary with processing statistics
    """
    stats = {
        'total_instances': 0,
        'instances_with_patch': 0,
        'instances_with_locations': 0,
        'total_locations': 0,
        'instances_with_problem_statement': 0
    }
    
    print(f"Processing {input_file} -> {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse JSON line
                instance = json.loads(line.strip())
                stats['total_instances'] += 1
                
                # Extract locations from patch if present
                patch = instance.get('patch', '')
                if patch:
                    stats['instances_with_patch'] += 1
                    locations = extract_locations_from_patch(patch, exclude_new_files=exclude_new_files)
                    
                    if locations:
                        stats['instances_with_locations'] += 1
                        stats['total_locations'] += len(locations)
                        
                        # Add location information to problem_statement
                        problem_statement = instance.get('problem_statement', '')
                        if problem_statement:
                            stats['instances_with_problem_statement'] += 1
                            location_prompt = format_locations_prompt(locations)
                            instance['problem_statement'] = problem_statement + location_prompt
                
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
        description="Add location information to SWE-bench datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='.',
        help='Directory containing the input JSONL files (default: current directory)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='datasets_with_edit_locations_v1',
        help='Directory to save the output files with locations (default: datasets_with_edit_locations)'
    )
    parser.add_argument(
        '--include-new-files',
        action='store_true',
        help='Include locations from newly created files (default: exclude them)'
    )
    parser.add_argument(
        '--suffix',
        type=str,
        default='_with_locations',
        help='Suffix to add to output filenames (default: _with_locations)'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    # If output_dir is relative, make it relative to the script's directory
    if not Path(args.output_dir).is_absolute():
        script_dir = Path(__file__).parent
        output_dir = script_dir / args.output_dir
    else:
        output_dir = Path(args.output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Define input and output files - specifically for swe-lite-test and swe-verified-test
    possible_files = [
        'swe-lite-test.jsonl',
        'swe-verified-test.jsonl'
    ]
    
    files_to_process = []
    for filename in possible_files:
        input_file = input_dir / filename
        if input_file.exists():
            output_filename = filename.replace('.jsonl', f'{args.suffix}.jsonl')
            files_to_process.append((filename, output_filename))
    
    if not files_to_process:
        print(f"No SWE-bench dataset files found in {input_dir}")
        print(f"Looking for: {', '.join(possible_files)}")
        return
    
    total_stats = {
        'total_instances': 0,
        'instances_with_patch': 0,
        'instances_with_locations': 0,
        'total_locations': 0,
        'instances_with_problem_statement': 0
    }
    
    for input_filename, output_filename in files_to_process:
        input_file = input_dir / input_filename
        output_file = output_dir / output_filename
        
        if not input_file.exists():
            print(f"Warning: {input_file} not found, skipping...")
            continue
        
        # Process the file
        exclude_new_files = not args.include_new_files
        stats = process_jsonl_file(input_file, output_file, exclude_new_files)
        
        # Update total stats
        for key in total_stats:
            total_stats[key] += stats[key]
        
        # Print file statistics
        print(f"\nResults for {input_filename}:")
        print(f"  Total instances: {stats['total_instances']}")
        print(f"  Instances with patch: {stats['instances_with_patch']}")
        print(f"  Instances with locations: {stats['instances_with_locations']}")
        print(f"  Total locations extracted: {stats['total_locations']}")
        print(f"  Instances with problem_statement updated: {stats['instances_with_problem_statement']}")
        if stats['instances_with_patch'] > 0:
            print(f"  Average locations per instance with patch: {stats['total_locations'] / stats['instances_with_patch']:.2f}")
        print()
    
    # Print overall statistics
    print("="*50)
    print("OVERALL RESULTS:")
    print(f"Total instances processed: {total_stats['total_instances']}")
    print(f"Instances with patch: {total_stats['instances_with_patch']}")
    print(f"Instances with locations: {total_stats['instances_with_locations']}")
    print(f"Total locations extracted: {total_stats['total_locations']}")
    print(f"Problem statements updated: {total_stats['instances_with_problem_statement']}")
    if total_stats['instances_with_patch'] > 0:
        print(f"Average locations per instance with patch: {total_stats['total_locations'] / total_stats['instances_with_patch']:.2f}")
    
    coverage_rate = (total_stats['instances_with_locations'] / total_stats['instances_with_patch'] * 100) if total_stats['instances_with_patch'] > 0 else 0
    print(f"Location extraction coverage: {coverage_rate:.1f}%")
    
    update_rate = (total_stats['instances_with_problem_statement'] / total_stats['instances_with_locations'] * 100) if total_stats['instances_with_locations'] > 0 else 0
    print(f"Problem statement update rate: {update_rate:.1f}%")
    
    exclude_new_files_str = "excluded" if not args.include_new_files else "included"
    print(f"New files were {exclude_new_files_str} from location extraction.")


if __name__ == '__main__':
    main()
