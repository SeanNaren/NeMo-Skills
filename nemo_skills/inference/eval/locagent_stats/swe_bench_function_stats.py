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

# Import the existing function from utils.py
from utils import extract_locations_from_patch

# Set a custom cache directory to avoid permission issues
os.environ['HF_HOME'] = './hf_cache'
os.environ['HF_DATASETS_CACHE'] = './hf_cache/datasets'
os.makedirs('./hf_cache/datasets', exist_ok=True)


def detect_function_changes_in_patch(patch: str) -> Dict[str, Any]:
    """
    Analyze a patch to detect function-related changes and other statistics.
    
    Returns:
        Dict containing:
        - num_files: Number of files modified
        - num_functions: Number of functions added/modified/removed
        - has_function_changes: Boolean indicating if any function changes exist
        - change_types: List of change types detected (imports, variables, classes, etc.)
        - functions_by_file: Dict mapping file paths to function changes
    """
    stats = {
        "num_files": 0,
        "num_functions": 0,
        "has_function_changes": False,
        "change_types": set(),
        "functions_by_file": defaultdict(list),
        "files": set()
    }
    
    if not patch:
        return stats
    
    # Function definition patterns for different languages
    function_patterns = {
        'python': [
            r'^\s*def\s+(\w+)\s*\(',  # Python function
            r'^\s*async\s+def\s+(\w+)\s*\(',  # Python async function
        ],
        'javascript': [
            r'^\s*function\s+(\w+)\s*\(',  # JavaScript function
            r'^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(',  # JS with export/async
            r'^\s*(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(',  # Arrow function
            r'^\s*(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?function',  # Function expression
        ],
        'typescript': [
            r'^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(',  # TypeScript function
            r'^\s*(?:private|public|protected)?\s*(?:static)?\s*(?:async)?\s*(\w+)\s*\(',  # Class method
            r'^\s*(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(',  # Arrow function
        ],
        'java': [
            r'^\s*(?:public|private|protected)?\s*(?:static)?\s*(?:final)?\s*(?:\w+(?:<[^>]+>)?)\s+(\w+)\s*\(',  # Java method
        ],
        'cpp': [
            r'^\s*(?:inline\s+)?(?:static\s+)?(?:\w+(?:::\w+)?(?:<[^>]+>)?)\s+(\w+)\s*\(',  # C++ function
            r'^\s*(\w+)::(\w+)\s*\(',  # C++ method implementation
        ],
        'go': [
            r'^\s*func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(',  # Go function
        ],
        'ruby': [
            r'^\s*def\s+(\w+)',  # Ruby method
        ],
        'php': [
            r'^\s*(?:public|private|protected)?\s*(?:static)?\s*function\s+(\w+)\s*\(',  # PHP function
        ]
    }
    
    # Other code patterns to detect
    import_patterns = [
        r'^\s*import\s+',  # Python/Java import
        r'^\s*from\s+\S+\s+import\s+',  # Python from import
        r'^\s*(?:const|let|var)\s+(?:\{[^}]+\}|\w+)\s*=\s*require\(',  # JS require
        r'^\s*import\s+(?:\{[^}]+\}|\w+)\s+from\s+',  # ES6 import
        r'^\s*#include\s*[<"]',  # C/C++ include
    ]
    
    class_patterns = [
        r'^\s*class\s+(\w+)',  # Python/Java/JS class
        r'^\s*(?:export\s+)?(?:abstract\s+)?class\s+(\w+)',  # TypeScript class
        r'^\s*struct\s+(\w+)',  # C/C++/Go struct
    ]
    
    variable_patterns = [
        r'^\s*(?:const|let|var)\s+\w+\s*=',  # JS/TS variable
        r'^\s*\w+\s*=\s*[^=]',  # Python/other assignment
        r'^\s*(?:public|private|protected)?\s*(?:static)?\s*(?:final)?\s*\w+\s+\w+\s*=',  # Java field
    ]
    
    current_file = None
    in_hunk = False
    
    for line in patch.splitlines():
        # File detection
        if line.startswith("--- ") or line.startswith("+++ "):
            file_path = line[4:]
            if file_path != "/dev/null":
                if file_path.startswith(("a/", "b/")):
                    file_path = file_path[2:]
                current_file = file_path
                stats["files"].add(file_path)
        
        # Check for actual changes (lines starting with + or -)
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            if current_file:
                # Determine file type based on extension
                file_ext = current_file.split('.')[-1] if '.' in current_file else ''
                
                # Map extensions to language groups
                lang_map = {
                    'py': 'python',
                    'js': 'javascript',
                    'jsx': 'javascript',
                    'ts': 'typescript',
                    'tsx': 'typescript',
                    'java': 'java',
                    'cpp': 'cpp',
                    'cc': 'cpp',
                    'cxx': 'cpp',
                    'c': 'cpp',
                    'h': 'cpp',
                    'hpp': 'cpp',
                    'go': 'go',
                    'rb': 'ruby',
                    'php': 'php'
                }
                
                lang = lang_map.get(file_ext, 'python')  # Default to Python
                
                # Check for function definitions
                function_found = False
                line_content = line[1:]  # Skip the +/- character
                for pattern in function_patterns.get(lang, []):
                    # Remove the ^ anchor since we're not at the beginning of the original line
                    pattern_no_anchor = pattern.lstrip('^')
                    match = re.match(pattern_no_anchor, line_content)
                    if match:
                        func_name = match.group(1)
                        stats["functions_by_file"][current_file].append(func_name)
                        stats["num_functions"] += 1
                        stats["has_function_changes"] = True
                        stats["change_types"].add("function")
                        function_found = True
                        break
                
                if not function_found:
                    # Check for imports
                    for pattern in import_patterns:
                        pattern_no_anchor = pattern.lstrip('^')
                        if re.match(pattern_no_anchor, line_content):
                            stats["change_types"].add("import")
                            break
                    
                    # Check for classes
                    for pattern in class_patterns:
                        pattern_no_anchor = pattern.lstrip('^')
                        if re.match(pattern_no_anchor, line_content):
                            stats["change_types"].add("class")
                            break
                    
                    # Check for variables
                    for pattern in variable_patterns:
                        pattern_no_anchor = pattern.lstrip('^')
                        if re.match(pattern_no_anchor, line_content):
                            stats["change_types"].add("variable")
                            break
                    
                    # Check for other common patterns
                    if re.match(r'\s*#', line_content) or re.match(r'\s*//', line_content) or re.match(r'\s*/\*', line_content):
                        stats["change_types"].add("comment")
                    elif re.match(r'\s*$', line_content):
                        stats["change_types"].add("whitespace")
                    elif not stats["change_types"]:
                        stats["change_types"].add("other")
    
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
        "total_functions_changed": 0,
        "samples_with_functions": 0,
        "samples_without_functions": 0,
        "change_type_counts": defaultdict(int),
        "avg_files_per_sample": 0,
        "avg_functions_per_sample": 0,
        "max_files_in_sample": 0,
        "max_functions_in_sample": 0,
        "samples": []
    }
    
    for idx, sample in enumerate(dataset):
        instance_id = sample.get("instance_id", f"sample_{idx}")
        patch = sample.get("patch", "")
        
        # Extract patch statistics
        patch_stats = detect_function_changes_in_patch(patch)
        
        # Extract locations using the existing function
        locations = extract_locations_from_patch(patch, exclude_new_files=False)
        
        sample_info = {
            "instance_id": instance_id,
            "repo": sample.get("repo", ""),
            "num_files": patch_stats["num_files"],
            "num_functions": patch_stats["num_functions"],
            "has_function_changes": patch_stats["has_function_changes"],
            "change_types": patch_stats["change_types"],
            "functions_by_file": dict(patch_stats["functions_by_file"]),
            "files": patch_stats["files"],
            "num_locations": len(locations)
        }
        
        # Update overall statistics
        overall_stats["total_files_changed"] += patch_stats["num_files"]
        overall_stats["total_functions_changed"] += patch_stats["num_functions"]
        
        if patch_stats["has_function_changes"]:
            overall_stats["samples_with_functions"] += 1
        else:
            overall_stats["samples_without_functions"] += 1
        
        for change_type in patch_stats["change_types"]:
            overall_stats["change_type_counts"][change_type] += 1
        
        overall_stats["max_files_in_sample"] = max(overall_stats["max_files_in_sample"], patch_stats["num_files"])
        overall_stats["max_functions_in_sample"] = max(overall_stats["max_functions_in_sample"], patch_stats["num_functions"])
        
        overall_stats["samples"].append(sample_info)
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} samples...")
    
    # Calculate averages
    if overall_stats["total_samples"] > 0:
        overall_stats["avg_files_per_sample"] = overall_stats["total_files_changed"] / overall_stats["total_samples"]
        overall_stats["avg_functions_per_sample"] = overall_stats["total_functions_changed"] / overall_stats["total_samples"]
    
    overall_stats["change_type_counts"] = dict(overall_stats["change_type_counts"])
    
    return overall_stats


def print_statistics(stats: Dict[str, Any]):
    """Pretty print the statistics."""
    print(f"\n{'='*60}")
    print(f"Dataset: {stats['dataset_name']} ({stats['split']} split)")
    print(f"{'='*60}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Total files changed: {stats['total_files_changed']}")
    print(f"Total functions changed: {stats['total_functions_changed']}")
    print(f"\nSamples with function changes: {stats['samples_with_functions']} ({stats['samples_with_functions']/stats['total_samples']*100:.1f}%)")
    print(f"Samples without function changes: {stats['samples_without_functions']} ({stats['samples_without_functions']/stats['total_samples']*100:.1f}%)")
    print(f"\nAverage files per sample: {stats['avg_files_per_sample']:.2f}")
    print(f"Average functions per sample: {stats['avg_functions_per_sample']:.2f}")
    print(f"Max files in a single sample: {stats['max_files_in_sample']}")
    print(f"Max functions in a single sample: {stats['max_functions_in_sample']}")
    
    print(f"\nChange type distribution:")
    for change_type, count in sorted(stats['change_type_counts'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {change_type}: {count}")
    
    # Show some examples of samples without functions
    print(f"\nExamples of samples without function changes:")
    non_func_samples = [s for s in stats['samples'] if not s['has_function_changes']][:5]
    for sample in non_func_samples:
        print(f"  - {sample['instance_id']}: {sample['change_types']} (files: {sample['num_files']})")


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
        
        for metric in ["total_samples", "samples_with_functions", "samples_without_functions", 
                      "avg_files_per_sample", "avg_functions_per_sample"]:
            print(f"\n{metric.replace('_', ' ').title()}:")
            for dataset_name, stats in all_stats.items():
                value = stats[metric]
                if isinstance(value, float):
                    print(f"  {dataset_name.split('/')[-1]}: {value:.2f}")
                else:
                    print(f"  {dataset_name.split('/')[-1]}: {value}")


if __name__ == "__main__":
    main()
