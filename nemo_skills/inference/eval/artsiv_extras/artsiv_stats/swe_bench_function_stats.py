"""
Script to analyze SWE-Bench Lite and Verified datasets to extract statistics about functions and patches.

This script:
1. Loads SWE-Bench Lite and Verified datasets from HuggingFace
2. Extracts patches from each sample
3. Analyzes patches to identify functions and other types of changes
4. Provides statistics on:
   - Number of files in each sample
   - Number of functions modified/added
   - Number of patches without function changes (imports, variables, etc.)
"""

import re
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from datasets import load_dataset
import json
import os

# Note: extract_locations_from_patch is not used in the core function detection logic

# Set a custom cache directory to avoid permission issues
os.environ['HF_HOME'] = './hf_cache'
os.environ['HF_DATASETS_CACHE'] = './hf_cache/datasets'
os.makedirs('./hf_cache/datasets', exist_ok=True)


def detect_function_changes_in_patch(patch: str) -> Dict[str, Any]:
    """
    Analyze a patch to detect if changes occur within function bodies.
    
    Returns:
        Dict containing:
        - num_files: Number of files modified
        - num_changes_in_functions: Number of changes that occur within function bodies
        - num_changes_outside_functions: Number of changes that occur outside function bodies
        - has_function_changes: Boolean indicating if any changes occur within functions
        - change_types: List of change types detected (imports, variables, classes, etc.)
        - changes_by_file: Dict mapping file paths to change locations and context
    """
    stats = {
        "num_files": 0,
        "num_changes_in_functions": 0,
        "num_changes_outside_functions": 0,
        "has_function_changes": False,
        "change_types": set(),
        "changes_by_file": defaultdict(list),
        "files": set()
    }
    
    if not patch:
        return stats
    
    # Parse the patch to understand the structure and context
    current_file = None
    hunks = []
    
    for line in patch.splitlines():
        # File detection
        if line.startswith("--- ") or line.startswith("+++ "):
            file_path = line[4:]
            if file_path != "/dev/null":
                if file_path.startswith(("a/", "b/")):
                    file_path = file_path[2:]
                current_file = file_path
                stats["files"].add(file_path)
        
        # Hunk header detection (e.g., @@ -1,4 +1,6 @@)
        elif line.startswith("@@"):
            if current_file and current_file.endswith('.py'):
                # Extract line numbers from hunk header
                hunk_match = re.match(r'@@ -(\d+),?(\d+)? \+(\d+),?(\d+)? @@', line)
                if hunk_match:
                    old_start = int(hunk_match.group(1))
                    old_count = int(hunk_match.group(2)) if hunk_match.group(2) else 1
                    new_start = int(hunk_match.group(3))
                    new_count = int(hunk_match.group(4)) if hunk_match.group(4) else 1
                    
                    hunk_info = {
                        'file': current_file,
                        'old_start': old_start,
                        'old_count': old_count,
                        'new_start': new_start,
                        'new_count': new_count,
                        'lines': [],
                        'context_lines': [],
                        'changes': []
                    }
                    hunks.append(hunk_info)
        
        # Collect lines within hunks
        elif hunks and current_file and current_file.endswith('.py'):
            current_hunk = hunks[-1]
            if line.startswith(('+', '-', ' ')):
                current_hunk['lines'].append(line)
                if line.startswith(' '):  # Context line
                    current_hunk['context_lines'].append(line[1:])
                elif line.startswith(('+', '-')):  # Actual change
                    current_hunk['changes'].append(line)
    
    # Analyze each hunk to determine if changes are within functions
    for hunk in hunks:
        if not hunk['changes']:  # Skip hunks with no actual changes
            continue
            
        # Build a full context with line numbers and track functions
        lines_with_context = []
        old_line_num = hunk['old_start']
        new_line_num = hunk['new_start']
        
        for line in hunk['lines']:
            if line.startswith(' '):  # Context line
                lines_with_context.append({
                    'type': 'context',
                    'content': line[1:],
                    'old_line': old_line_num,
                    'new_line': new_line_num
                })
                old_line_num += 1
                new_line_num += 1
            elif line.startswith('-'):  # Removed line
                lines_with_context.append({
                    'type': 'removed',
                    'content': line[1:],
                    'old_line': old_line_num,
                    'new_line': None
                })
                old_line_num += 1
            elif line.startswith('+'):  # Added line
                lines_with_context.append({
                    'type': 'added',
                    'content': line[1:],
                    'old_line': None,
                    'new_line': new_line_num
                })
                new_line_num += 1
        
        # Track function context through the hunk
        current_function = None
        function_indent = None
        changes_in_functions = 0
        changes_outside_functions = 0
        
        for i, line_info in enumerate(lines_with_context):
            content = line_info['content']
            stripped = content.strip()
            
            # Skip empty lines and comments for function tracking
            if not stripped or stripped.startswith('#'):
                continue
                
            line_indent = len(content) - len(content.lstrip())
            
            # Check for function definition
            if re.match(r'^\s*(?:async\s+)?def\s+(\w+)\s*\(', content):
                func_match = re.match(r'^\s*(?:async\s+)?def\s+(\w+)\s*\(', content)
                if func_match:
                    current_function = func_match.group(1)
                    function_indent = line_indent
                    # Function definition itself is outside function body
                    if line_info['type'] in ['added', 'removed']:
                        changes_outside_functions += 1
                    continue
            
            # Check for class definition
            elif re.match(r'^\s*class\s+\w+', content):
                current_function = None
                function_indent = None
                if line_info['type'] in ['added', 'removed']:
                    changes_outside_functions += 1
                continue
            
            # Check if we're still inside the current function
            if current_function is not None and function_indent is not None:
                # If indentation is greater than function definition, we're inside
                if line_indent > function_indent:
                    if line_info['type'] in ['added', 'removed']:
                        changes_in_functions += 1
                        continue
                # If indentation is equal or less, we might be outside
                elif line_indent <= function_indent:
                    # Check if it's a decorator, docstring, or continuation
                    if not (stripped.startswith('@') or 
                           stripped.startswith('"""') or 
                           stripped.startswith("'''")):
                        current_function = None
                        function_indent = None
            
            # If we reach here and it's a change, it's outside a function
            if line_info['type'] in ['added', 'removed']:
                changes_outside_functions += 1
        
        # Update stats
        stats["num_changes_in_functions"] += changes_in_functions
        stats["num_changes_outside_functions"] += changes_outside_functions
        
        if changes_in_functions > 0:
            stats["has_function_changes"] = True
            stats["changes_by_file"][hunk['file']].append({
                'function': current_function,
                'type': 'within_function',
                'changes': changes_in_functions
            })
        
        if changes_outside_functions > 0:
            stats["changes_by_file"][hunk['file']].append({
                'function': None,
                'type': 'outside_function', 
                'changes': changes_outside_functions
            })
        
        # Analyze change types
        for change_line in hunk['changes']:
            if change_line.startswith('+'):
                line_content = change_line[1:]
                if re.match(r'\s*import\s+', line_content) or re.match(r'\s*from\s+\S+\s+import\s+', line_content):
                    stats["change_types"].add("import")
                elif re.match(r'\s*class\s+(\w+)', line_content):
                    stats["change_types"].add("class")
                elif re.match(r'\s*def\s+(\w+)', line_content) or re.match(r'\s*async\s+def\s+(\w+)', line_content):
                    stats["change_types"].add("function_definition")
                elif re.match(r'\s*#', line_content):
                    stats["change_types"].add("comment")
                elif re.match(r'\s*$', line_content):
                    stats["change_types"].add("whitespace")
                else:
                    stats["change_types"].add("code")
    
    stats["num_files"] = len(stats["files"])
    stats["change_types"] = list(stats["change_types"])
    stats["files"] = list(stats["files"])
    
    return stats


def analyze_swe_bench_dataset(dataset_name: str, split: str = "test") -> Dict[str, Any]:
    """
    Analyze a SWE-Bench dataset and extract statistics.
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., "princeton-nlp/SWE-bench_Lite")
        split: Dataset split to analyze (default: "test")
    
    Returns:
        Dict containing overall statistics and per-sample details
    """
    print(f"\nLoading dataset: {dataset_name} (split: {split})...")
    # Try to load with custom cache directory
    cache_dir = os.path.abspath('./hf_cache')
    os.makedirs(cache_dir, exist_ok=True)
    dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    
    overall_stats = {
        "dataset_name": dataset_name,
        "split": split,
        "total_samples": len(dataset),
        "total_files_changed": 0,
        "total_changes_in_functions": 0,
        "total_changes_outside_functions": 0,
        "samples_with_function_changes": 0,
        "samples_without_function_changes": 0,
        "change_type_counts": defaultdict(int),
        "avg_files_per_sample": 0,
        "avg_changes_in_functions_per_sample": 0,
        "avg_changes_outside_functions_per_sample": 0,
        "max_files_in_sample": 0,
        "max_changes_in_functions_in_sample": 0,
        "samples": []
    }
    
    for idx, sample in enumerate(dataset):
        instance_id = sample.get("instance_id", f"sample_{idx}")
        patch = sample.get("patch", "")
        
        # Extract patch statistics
        patch_stats = detect_function_changes_in_patch(patch)
        
        # Note: locations extraction functionality removed to focus on function detection
        locations = []
        
        sample_info = {
            "instance_id": instance_id,
            "repo": sample.get("repo", ""),
            "num_files": patch_stats["num_files"],
            "num_changes_in_functions": patch_stats["num_changes_in_functions"],
            "num_changes_outside_functions": patch_stats["num_changes_outside_functions"],
            "has_function_changes": patch_stats["has_function_changes"],
            "change_types": patch_stats["change_types"],
            "changes_by_file": dict(patch_stats["changes_by_file"]),
            "files": patch_stats["files"],
            "num_locations": len(locations)
        }
        
        # Update overall statistics
        overall_stats["total_files_changed"] += patch_stats["num_files"]
        overall_stats["total_changes_in_functions"] += patch_stats["num_changes_in_functions"]
        overall_stats["total_changes_outside_functions"] += patch_stats["num_changes_outside_functions"]
        
        if patch_stats["has_function_changes"]:
            overall_stats["samples_with_function_changes"] += 1
        else:
            overall_stats["samples_without_function_changes"] += 1
        
        for change_type in patch_stats["change_types"]:
            overall_stats["change_type_counts"][change_type] += 1
        
        overall_stats["max_files_in_sample"] = max(overall_stats["max_files_in_sample"], patch_stats["num_files"])
        overall_stats["max_changes_in_functions_in_sample"] = max(overall_stats["max_changes_in_functions_in_sample"], patch_stats["num_changes_in_functions"])
        
        overall_stats["samples"].append(sample_info)
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} samples...")
    
    # Calculate averages
    if overall_stats["total_samples"] > 0:
        overall_stats["avg_files_per_sample"] = overall_stats["total_files_changed"] / overall_stats["total_samples"]
        overall_stats["avg_changes_in_functions_per_sample"] = overall_stats["total_changes_in_functions"] / overall_stats["total_samples"]
        overall_stats["avg_changes_outside_functions_per_sample"] = overall_stats["total_changes_outside_functions"] / overall_stats["total_samples"]
    
    overall_stats["change_type_counts"] = dict(overall_stats["change_type_counts"])
    
    return overall_stats


def print_statistics(stats: Dict[str, Any]):
    """Pretty print the statistics."""
    print(f"\n{'='*60}")
    print(f"Dataset: {stats['dataset_name']} ({stats['split']} split)")
    print(f"{'='*60}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Total files changed: {stats['total_files_changed']}")
    print(f"Total changes in functions: {stats['total_changes_in_functions']}")
    print(f"Total changes outside functions: {stats['total_changes_outside_functions']}")
    
    total_changes = stats['total_changes_in_functions'] + stats['total_changes_outside_functions']
    if total_changes > 0:
        func_percentage = stats['total_changes_in_functions'] / total_changes * 100
        outside_percentage = stats['total_changes_outside_functions'] / total_changes * 100
        print(f"\nChanges within functions: {stats['total_changes_in_functions']} ({func_percentage:.1f}%)")
        print(f"Changes outside functions: {stats['total_changes_outside_functions']} ({outside_percentage:.1f}%)")
    
    print(f"\nSamples with function changes: {stats['samples_with_function_changes']} ({stats['samples_with_function_changes']/stats['total_samples']*100:.1f}%)")
    print(f"Samples without function changes: {stats['samples_without_function_changes']} ({stats['samples_without_function_changes']/stats['total_samples']*100:.1f}%)")
    print(f"\nAverage files per sample: {stats['avg_files_per_sample']:.2f}")
    print(f"Average changes in functions per sample: {stats['avg_changes_in_functions_per_sample']:.2f}")
    print(f"Average changes outside functions per sample: {stats['avg_changes_outside_functions_per_sample']:.2f}")
    print(f"Max files in a single sample: {stats['max_files_in_sample']}")
    print(f"Max changes in functions in a single sample: {stats['max_changes_in_functions_in_sample']}")
    
    print(f"\nChange type distribution:")
    for change_type, count in sorted(stats['change_type_counts'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {change_type}: {count}")
    
    # Show some examples of samples without function changes
    print(f"\nExamples of samples without function changes:")
    non_func_samples = [s for s in stats['samples'] if not s['has_function_changes']][:5]
    for sample in non_func_samples:
        print(f"  - {sample['instance_id']}: {sample['change_types']} (files: {sample['num_files']})")
    
    # Show some examples of samples with function changes
    print(f"\nExamples of samples with function changes:")
    func_samples = [s for s in stats['samples'] if s['has_function_changes']][:5]
    for sample in func_samples:
        print(f"  - {sample['instance_id']}: {sample['num_changes_in_functions']} changes in functions, {sample['num_changes_outside_functions']} outside")


def main():
    """Main function to analyze both SWE-Bench Lite and Verified datasets."""
    datasets = [
        "princeton-nlp/SWE-bench_Lite",
        "princeton-nlp/SWE-bench_Verified"
    ]
    
    all_stats = {}
    
    for dataset_name in datasets:
        try:
            stats = analyze_swe_bench_dataset(dataset_name)
            all_stats[dataset_name] = stats
            print_statistics(stats)
            
            # Save detailed results to JSON
            output_file = f"swe_bench_stats_{dataset_name.split('/')[-1].lower()}.json"
            with open(output_file, 'w') as f:
                # Remove the full samples list for the summary file
                summary_stats = {k: v for k, v in stats.items() if k != 'samples'}
                json.dump(summary_stats, f, indent=2)
            print(f"\nSaved summary statistics to: {output_file}")
            
            # Save detailed per-sample data
            detailed_file = f"swe_bench_detailed_{dataset_name.split('/')[-1].lower()}.json"
            with open(detailed_file, 'w') as f:
                json.dump(stats['samples'], f, indent=2)
            print(f"Saved detailed per-sample data to: {detailed_file}")
            
        except Exception as e:
            print(f"\nError processing {dataset_name}: {e}")
    
    # Print comparison between datasets
    if len(all_stats) == 2:
        print(f"\n{'='*60}")
        print("COMPARISON BETWEEN DATASETS")
        print(f"{'='*60}")
        
        for metric in ["total_samples", "samples_with_function_changes", "samples_without_function_changes", 
                      "avg_files_per_sample", "avg_changes_in_functions_per_sample", "avg_changes_outside_functions_per_sample"]:
            print(f"\n{metric.replace('_', ' ').title()}:")
            for dataset_name, stats in all_stats.items():
                value = stats[metric]
                if isinstance(value, float):
                    print(f"  {dataset_name.split('/')[-1]}: {value:.2f}")
                else:
                    print(f"  {dataset_name.split('/')[-1]}: {value}")


if __name__ == "__main__":
    main()
