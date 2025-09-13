#!/usr/bin/env python3
"""
Script to find the most similar file paths to problem descriptions in SWE-bench datasets
using the SWERankEmbed-Large model.

Usage:
    python swerank_embed.py --dataset <dataset_name> --top_k <k> --output_path <output.json>
    
Where dataset_name can be 'princeton-nlp/SWE-bench_Lite' or 'princeton-nlp/SWE-bench_Verified'

Examples:
    # Use full repository contents for similarity (default: 5 parallel GPUs)
    python swerank_embed.py --dataset princeton-nlp/SWE-bench_Lite --top_k 5 --output_path results.json
    
    # Use all 10 GPUs for maximum parallelization
    python swerank_embed.py --dataset princeton-nlp/SWE-bench_Lite --top_k 5 --output_path results.json --num_workers 10
    
    # Conservative settings for memory-constrained systems
    python swerank_embed.py --dataset princeton-nlp/SWE-bench_Lite --top_k 5 --output_path results.json --num_workers 3 --batch_size 4
    
    # Specify custom repository structures directory
    python swerank_embed.py --dataset princeton-nlp/SWE-bench_Lite --top_k 5 --output_path results.json --repo_structures_dir /path/to/repo_structures
"""

import argparse
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from typing import List, Tuple, Dict
import json
import sys
import os
import pickle
import re
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Set up local cache directory
CACHE_DIR = "/home/htamoyan/cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = CACHE_DIR

# Add the parent directory to sys.path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from locagent_utils.utils import extract_locations_from_patch


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool the last token from the hidden states."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format the query with task instruction."""
    return f'Instruct: {task_description}\nQuery: {query}'


def load_repo_structure_from_pickle(instance_id: str, repo_structures_dir: str) -> Dict[str, str]:
    """Load repository structure from pickle file and extract all file contents.
    
    Returns a dictionary mapping file paths to their contents.
    """
    file_contents = {}
    
    # Try both dataset directories
    for dataset_name in ['SWE-bench_Lite', 'SWE-bench_Verified']:
        pickle_path = Path(repo_structures_dir) / dataset_name / f"{instance_id}.pkl"
        if pickle_path.exists():
            try:
                with open(pickle_path, 'rb') as f:
                    repo_data = pickle.load(f)
                
                # Extract files from the structure
                if 'structure' in repo_data:
                    file_contents = extract_files_from_structure(repo_data['structure'])
                    print(f"  Loaded repository structure from {pickle_path}")
                    return file_contents
            except Exception as e:
                print(f"  Error loading {pickle_path}: {e}")
                continue
    
    print(f"  No repository structure found for {instance_id}")
    return {}


def extract_files_from_structure(structure: Dict, current_path: str = "") -> Dict[str, str]:
    """Recursively extract all files from the repository structure."""
    file_contents = {}
    
    for name, content in structure.items():
        # Skip .git and other non-code directories
        if name.startswith('.git'):
            continue
            
        file_path = f"{current_path}/{name}" if current_path else name
        
        if isinstance(content, dict):
            if 'text' in content and isinstance(content['text'], list):
                # This is a file with content
                text_lines = content['text']
                
                # Clean line numbers from the beginning of each line (format: "5: content")
                cleaned_lines = []
                for line in text_lines:
                    # Remove line numbers at the beginning (pattern: number + ": ")
                    # Match patterns like "5: " or "    42: " at the start of lines
                    cleaned_line = re.sub(r'^\s*\d+:\s*', '', line)
                    cleaned_lines.append(cleaned_line)
                
                file_text = '\n'.join(cleaned_lines)
                # Only include files with actual content and reasonable file extensions
                if file_text.strip() and should_include_file(file_path):
                    file_contents[file_path] = file_text
            else:
                # This is a directory, recurse
                sub_files = extract_files_from_structure(content, file_path)
                file_contents.update(sub_files)
    
    return file_contents


def should_include_file(file_path: str) -> bool:
    """Check if a file should be included based on its extension.
    
    Only processes .py and .cfg files for maximum speed.
    """
    file_ext = Path(file_path).suffix.lower()
    return file_ext in {'.py', '.cfg'}


def extract_file_contents_from_instance(instance: Dict) -> Dict[str, str]:
    """Extract file contents from a SWE-bench instance.
    
    Returns a dictionary mapping file paths to their contents.
    Since SWE-bench dataset doesn't contain full repo contents directly,
    we'll extract available code snippets from patches and related fields.
    """
    file_contents = {}
    
    # Try to extract file contents from environment_setup_commit field if available
    if 'environment_setup_commit' in instance:
        # This might contain some repo information
        pass
    
    # Extract code from patch context
    if 'patch' in instance and instance['patch']:
        file_contents.update(extract_code_from_patch(instance['patch']))
    
    # Extract code from test_patch if available  
    if 'test_patch' in instance and instance['test_patch']:
        file_contents.update(extract_code_from_patch(instance['test_patch']))
    
    # Extract any code snippets from problem statement
    if 'problem_statement' in instance and instance['problem_statement']:
        code_snippets = extract_code_snippets_from_text(instance['problem_statement'])
        for i, snippet in enumerate(code_snippets):
            file_contents[f"snippet_{i}.py"] = snippet
    
    return file_contents


def extract_code_from_patch(patch: str) -> Dict[str, str]:
    """Extract file contents from git patch format."""
    file_contents = {}
    current_file = None
    current_content = []
    
    for line in patch.splitlines():
        # File path from --- or +++ line
        if line.startswith("--- ") or line.startswith("+++ "):
            # Save previous file if we have one
            if current_file and current_content:
                file_contents[current_file] = '\n'.join(current_content)
                current_content = []
            
            # Extract new file path
            file_path = line[4:]  # Remove "--- " or "+++ "
            if file_path.startswith(("a/", "b/")):
                file_path = file_path[2:]
            if file_path != "/dev/null":
                current_file = file_path
        
        # Skip hunk headers
        elif line.startswith("@@"):
            continue
        
        # Extract content lines (context and additions)
        elif current_file and line:
            if line.startswith(" ") or line.startswith("+"):
                # Context line or addition - extract the actual content
                content = line[1:] if line.startswith((" ", "+")) else line
                current_content.append(content)
    
    # Save the last file
    if current_file and current_content:
        file_contents[current_file] = '\n'.join(current_content)
    
    return file_contents


def extract_code_snippets_from_text(text: str) -> List[str]:
    """Extract code snippets from markdown-style code blocks."""
    import re
    code_blocks = re.findall(r'```(?:python|py)?\n(.*?)\n```', text, re.DOTALL)
    # Also try to find indented code blocks
    indented_blocks = re.findall(r'\n((?:    .+\n)+)', text)
    return code_blocks + [block.replace('    ', '') for block in indented_blocks]


def extract_file_paths(instance: Dict) -> List[str]:
    """Extract all file paths from a SWE-bench instance."""
    file_paths = []
    
    # Get file paths from patch if available
    if 'patch' in instance and instance['patch']:
        import re
        # Extract file paths from patch headers (--- a/file_path and +++ b/file_path)
        patch_file_pattern = r'(?:---|\+\+\+) [ab]/(.+?)(?:\s|$)'
        matches = re.findall(patch_file_pattern, instance['patch'])
        file_paths.extend(matches)
    
    # Get file paths from test_patch if available
    if 'test_patch' in instance and instance['test_patch']:
        import re
        patch_file_pattern = r'(?:---|\+\+\+) [ab]/(.+?)(?:\s|$)'
        matches = re.findall(patch_file_pattern, instance['test_patch'])
        file_paths.extend(matches)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_file_paths = []
    for path in file_paths:
        if path not in seen:
            seen.add(path)
            unique_file_paths.append(path)
    
    return unique_file_paths


def extract_ground_truth_files(instance: Dict) -> List[str]:
    """Extract ground truth file paths from the patch using the utility function."""
    ground_truth_files = []
    
    # Extract from main patch
    if 'patch' in instance and instance['patch']:
        locations = extract_locations_from_patch(instance['patch'])
        for loc in locations:
            if loc['file_path'] not in ground_truth_files:
                ground_truth_files.append(loc['file_path'])
    
    return ground_truth_files


def check_ground_truth_in_topk(ground_truth_files: List[str], top_k_files: List[Tuple[str, float]]) -> Dict:
    """Check if ground truth files are in the top-k predictions."""
    top_k_file_paths = [fp for fp, _ in top_k_files]
    
    hits = []
    for gt_file in ground_truth_files:
        if gt_file in top_k_file_paths:
            rank = top_k_file_paths.index(gt_file) + 1  # 1-indexed rank
            hits.append({'file_path': gt_file, 'rank': rank})
    
    return {
        'total_ground_truth': len(ground_truth_files),
        'hits_in_topk': len(hits),
        'hit_rate': len(hits) / len(ground_truth_files) if ground_truth_files else 0.0,
        'hits': hits,
        'missed_files': [gt for gt in ground_truth_files if gt not in top_k_file_paths]
    }


def compute_similarity_scores(
    problem_description: str,
    file_contents: Dict[str, str],
    model,
    tokenizer,
    task_instruction: str,
    max_length: int = 8192,
    device: str = 'cpu',
    batch_size: int = 8
) -> List[Tuple[str, float]]:
    """Compute similarity scores between problem description and file contents."""
    
    if not file_contents:
        return []
    
    file_paths = list(file_contents.keys())
    file_texts = list(file_contents.values())
    
    # Format the query with instruction
    query_with_prefix = get_detailed_instruct(task_instruction, problem_description)
    
    # Tokenize query once
    query_inputs = tokenizer(
        [query_with_prefix], 
        padding=True, 
        truncation=True, 
        return_tensors='pt', 
        max_length=max_length
    ).to(device)
    
    # Compute query embedding once
    with torch.no_grad():
        query_embeddings = last_token_pool(
            model(**query_inputs).last_hidden_state, 
            query_inputs["attention_mask"]
        )
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    
    # Process file contents in batches for better GPU utilization and memory management
    all_scores = []
    total_batches = (len(file_texts) + batch_size - 1) // batch_size
    
    for batch_idx in range(0, len(file_texts), batch_size):
        current_batch = (batch_idx // batch_size) + 1
        print(f"    Processing batch {current_batch}/{total_batches} ({batch_size} files)")
        
        batch_texts = file_texts[batch_idx:batch_idx+batch_size]
        
        # Prepare documents with file path context for better embeddings
        batch_documents = []
        for j, text in enumerate(batch_texts):
            file_path = file_paths[batch_idx + j]
            # Include file path as context for better understanding
            # Truncate very long files to prevent memory issues
            if len(text) > 50000:  # ~50KB limit per file
                text = text[:50000] + "... [truncated]"
            document = f"File: {file_path}\n\n{text}"
            batch_documents.append(document)
        
        try:
            # Tokenize batch of file contents
            document_inputs = tokenizer(
                batch_documents, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=max_length
            ).to(device)
            
            # Compute embeddings for this batch
            with torch.no_grad():
                document_embeddings = last_token_pool(
                    model(**document_inputs).last_hidden_state, 
                    document_inputs["attention_mask"]
                )
                document_embeddings = F.normalize(document_embeddings, p=2, dim=1)
                
                # Compute similarity scores for this batch
                batch_scores = torch.mm(query_embeddings, document_embeddings.transpose(0, 1))
                all_scores.extend(batch_scores[0].cpu().tolist())
                
                # Clear GPU cache after each batch
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                    
        except torch.cuda.OutOfMemoryError as e:
            print(f"    Warning: GPU OOM in batch {current_batch}, skipping batch: {e}")
            # Add zero scores for skipped files
            all_scores.extend([0.0] * len(batch_texts))
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
    
    # Convert to list of (file_path, score) tuples
    file_score_pairs = list(zip(file_paths, all_scores))
    
    # Sort by score in descending order
    file_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    return file_score_pairs


def process_single_instance(args_tuple):
    """Process a single instance on a specific GPU."""
    (instance_idx, instance, gpu_id, repo_structures_dir, max_files_per_instance, 
     batch_size, task_instruction, max_length, model_name) = args_tuple
    
    try:
        # Set up GPU for this worker
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = f"cuda:0"  # Will be the only visible GPU for this process
        
        # Load model on this GPU
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=CACHE_DIR)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, cache_dir=CACHE_DIR)
        model.eval()
        model.to(device)
        
        print(f"Worker GPU {gpu_id}: Processing instance {instance_idx+1}: {instance['instance_id']}")
        
        # Extract problem description
        problem_description = instance.get('problem_statement', '')
        if not problem_description:
            print(f"Worker GPU {gpu_id}: Warning: No problem statement found for instance {instance['instance_id']}")
            return None
        
        # Extract ground truth files
        ground_truth_files = extract_ground_truth_files(instance)
        
        # Try to load from repository structures first
        file_contents = load_repo_structure_from_pickle(instance['instance_id'], repo_structures_dir)
        
        # Fallback to extracting from instance if no repo structure found
        if not file_contents:
            print(f"Worker GPU {gpu_id}: Falling back to extracting from patches/problem statement")
            file_contents = extract_file_contents_from_instance(instance)
        
        file_paths = list(file_contents.keys())
        
        if not file_contents:
            print(f"Worker GPU {gpu_id}: Warning: No file contents found for instance {instance['instance_id']}")
            return {
                'instance_id': instance['instance_id'],
                'problem_statement': problem_description,
                'ground_truth_files': ground_truth_files,
                'top_k_files': [],
                'all_file_paths': [],
                'available_files': [],
                'source': 'none',
                'gpu_id': gpu_id,
                'ground_truth_analysis': {
                    'total_ground_truth': len(ground_truth_files),
                    'hits_in_topk': 0,
                    'hit_rate': 0.0,
                    'hits': [],
                    'missed_files': ground_truth_files
                }
            }
        
        source = 'repository_structure' if len(file_contents) > 10 else 'patches_and_problem'
        
        # Process all files unless explicitly limited (SWERank approach)
        if max_files_per_instance is not None and len(file_contents) > max_files_per_instance:
            print(f"Worker GPU {gpu_id}: Found {len(file_contents)} files, limiting to {max_files_per_instance} for memory management")
            
            # Prioritize files more intelligently (following SWERank approach)
            def file_priority(item):
                file_path, content = item
                score = 0
                
                # Prioritize source files over test files
                if '/test' not in file_path.lower() and 'test_' not in file_path.lower():
                    score += 1000
                
                # Prioritize Python files (for most SWE-bench instances)
                if file_path.endswith('.py'):
                    score += 500
                
                # Prioritize files in main package directories
                if any(pkg in file_path for pkg in ['src/', 'lib/', 'core/', 'main/']):
                    score += 300
                
                # Prioritize smaller files (easier to process, often more focused)
                score += max(0, 1000 - len(content) // 100)
                
                return score
            
            # Sort by priority and take the top files
            sorted_files = sorted(file_contents.items(), key=file_priority, reverse=True)
            file_contents = dict(sorted_files[:max_files_per_instance])
        else:
            print(f"Worker GPU {gpu_id}: Processing ALL {len(file_contents)} files (SWERank approach)")
        
        print(f"Worker GPU {gpu_id}: Processing {len(file_contents)} files with content (source: {source})")
        
        # Compute similarity scores using file contents
        file_score_pairs = compute_similarity_scores(
            problem_description=problem_description,
            file_contents=file_contents,
            model=model,
            tokenizer=tokenizer,
            task_instruction=task_instruction,
            max_length=max_length,
            device=device,
            batch_size=batch_size
        )
        
        print(f"Worker GPU {gpu_id}: Found {len(ground_truth_files)} ground truth files: {ground_truth_files}")
        
        # Get top-k results (we'll determine top_k in main process)
        file_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Check ground truth in results
        gt_analysis = check_ground_truth_in_topk(ground_truth_files, file_score_pairs)
        
        print(f"Worker GPU {gpu_id}: Completed instance {instance['instance_id']} - Ground truth hit rate: {gt_analysis['hit_rate']:.2%}")
        
        return {
            'instance_id': instance['instance_id'],
            'problem_statement': problem_description,
            'ground_truth_files': ground_truth_files,
            'file_score_pairs': file_score_pairs,  # Return full scores, filter top_k later
            'all_file_paths': file_paths,
            'available_files': list(file_contents.keys()),
            'source': source,
            'gpu_id': gpu_id,
            'ground_truth_analysis': gt_analysis
        }
        
    except Exception as e:
        print(f"Worker GPU {gpu_id}: Error processing instance {instance['instance_id']}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Find most similar file paths to SWE-bench problem descriptions")
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        choices=["princeton-nlp/SWE-bench_Lite", "princeton-nlp/SWE-bench_Verified"],
        help="Name of the SWE-bench dataset"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        required=True,
        help="Number of top similar file paths to return for each task"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run the model on ('auto', 'cpu', 'cuda', 'cuda:0', etc.). 'auto' will use best available GPU"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for processing files (larger batches utilize GPU better but use more memory)"
    )
    parser.add_argument(
        "--max_files_per_instance",
        type=int,
        default=None,
        help="Maximum number of files to process per instance (None means process all files)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=5,
        help="Number of parallel GPU workers (should be <= number of available GPUs)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Salesforce/SweRankEmbed-Large",
        help="Name of the embedding model to use"
    )
    parser.add_argument(
        "--repo_structures_dir",
        type=str,
        default="/mnt/ssd/htamoyan/repo_structures",
        help="Directory containing pickled repository structures"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output file path to save results JSON"
    )
    parser.add_argument(
        "--max_instances",
        type=int,
        default=None,
        help="Maximum number of instances to process (for testing)"
    )
    
    args = parser.parse_args()
    
    print(f"Processing mode: Full file contents with line number cleaning")
    print(f"Multi-GPU processing with {args.num_workers} workers")
    
    # Initialize suitable GPUs list
    suitable_gpus = []
    
    # Check available GPUs
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Found {gpu_count} GPUs available")
        
        # Filter GPUs by memory (need at least 8GB for SWERankEmbed-Large)
        min_memory_gb = 8.0
        
        for i in range(gpu_count):
            torch.cuda.set_device(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory
            free_memory = total_memory - torch.cuda.memory_allocated(i)
            total_memory_gb = total_memory / 1024**3
            free_memory_gb = free_memory / 1024**3
            
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name} - Total: {total_memory_gb:.2f} GB, Free: {free_memory_gb:.2f} GB")
            
            if total_memory_gb >= min_memory_gb:
                suitable_gpus.append(i)
                print(f"  ✓ GPU {i} suitable for processing")
            else:
                print(f"  ✗ GPU {i} insufficient memory (need {min_memory_gb:.0f}GB+)")
        
        if not suitable_gpus:
            print("No GPUs with sufficient memory found, falling back to CPU processing")
            args.num_workers = 1
        else:
            print(f"Found {len(suitable_gpus)} suitable GPUs: {suitable_gpus}")
            
            if args.num_workers > len(suitable_gpus):
                print(f"Warning: Requested {args.num_workers} workers but only {len(suitable_gpus)} suitable GPUs available. Using {len(suitable_gpus)} workers.")
                args.num_workers = len(suitable_gpus)
    else:
        print("No GPUs available, falling back to CPU processing")
        args.num_workers = 1
    
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="test", cache_dir=CACHE_DIR)
    
    if args.max_instances:
        dataset = dataset.select(range(min(args.max_instances, len(dataset))))
    
    # Task instruction for the embedding model (following SWERank paper approach)
    task_instruction = 'Given a software issue, retrieve the most relevant source code files that need to be modified to resolve the issue.'
    max_length = 8192
    
    # Prepare arguments for each instance
    instance_args = []
    gpu_assignment = []
    
    for i, instance in enumerate(dataset):
        if torch.cuda.is_available() and suitable_gpus:
            # Round-robin assignment among suitable GPUs only
            gpu_id = suitable_gpus[i % len(suitable_gpus)]
        else:
            gpu_id = 0  # CPU fallback
        gpu_assignment.append(gpu_id)
        
        instance_args.append((
            i, instance, gpu_id, args.repo_structures_dir, args.max_files_per_instance,
            args.batch_size, task_instruction, max_length, args.model_name
        ))
    
    if torch.cuda.is_available() and suitable_gpus:
        print(f"Processing {len(dataset)} instances across {args.num_workers} suitable GPUs: {suitable_gpus}")
        # Show assignment for first 10 instances as example
        sample_assignments = dict(list(zip(range(min(10, len(dataset))), gpu_assignment[:10])))
        print(f"Sample GPU assignments: {sample_assignments}{'...' if len(dataset) > 10 else ''}")
    else:
        print(f"Processing {len(dataset)} instances on CPU")
    
    # Process instances in parallel using multiprocessing
    results = []
    completed_count = 0
    start_time = time.time()
    
    # Use spawn method to avoid issues with CUDA and multiprocessing
    mp.set_start_method('spawn', force=True)
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all jobs
        future_to_idx = {executor.submit(process_single_instance, args): i for i, args in enumerate(instance_args)}
        
        # Collect results as they complete
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            completed_count += 1
            
            try:
                result = future.result()
                if result is not None:
                    # Extract top-k files from full results
                    top_k_files = result['file_score_pairs'][:args.top_k]
                    
                    # Recalculate ground truth analysis for top-k
                    gt_analysis = check_ground_truth_in_topk(result['ground_truth_files'], top_k_files)
                    
                    # Format final result - keep it simple and clean
                    final_result = {
                        'instance_id': result['instance_id'],
                        'ground_truth_files': result['ground_truth_files'],
                        'top_k_files': [fp for fp, score in top_k_files],  # Just file paths, no scores
                        'hit_rate': gt_analysis['hit_rate'],
                        'hits_in_topk': gt_analysis['hits_in_topk']
                    }
                    
                    results.append(final_result)
                    
                    print(f"Completed {completed_count}/{len(dataset)} instances - Instance: {result['instance_id']} (GPU {result['gpu_id']}) - Hit rate: {gt_analysis['hit_rate']:.2%}")
                else:
                    print(f"Completed {completed_count}/{len(dataset)} instances - Failed processing")
                    
            except Exception as e:
                print(f"Error processing instance {idx}: {e}")
                completed_count += 1
    
    elapsed_time = time.time() - start_time
    print(f"\nCompleted all {len(dataset)} instances in {elapsed_time:.2f} seconds ({elapsed_time/len(dataset):.2f}s per instance)")
    
    # Sort results by instance order (since they may complete out of order)
    results.sort(key=lambda x: x['instance_id'])
    
    # Calculate overall statistics
    total_instances = len(results)
    total_with_gt = sum(1 for r in results if r['ground_truth_files'])
    total_hits = sum(r['hits_in_topk'] for r in results)
    total_gt_files = sum(len(r['ground_truth_files']) for r in results)
    overall_hit_rate = total_hits / total_gt_files if total_gt_files > 0 else 0.0
    
    # Calculate accuracy@k (percentage of samples with at least one ground truth in top-k)
    samples_with_hits = sum(1 for r in results if r['hits_in_topk'] > 0)
    accuracy_at_k = samples_with_hits / total_instances if total_instances > 0 else 0.0
    
    # Add summary statistics - keep it simple
    summary = {
        'total_instances': total_instances,
        'accuracy_at_k': accuracy_at_k,
        'samples_with_hits': samples_with_hits,
        'overall_hit_rate': overall_hit_rate,
        'top_k': args.top_k,
        'dataset': args.dataset
    }
    
    output_data = {
        'summary': summary,
        'results': results
    }
    
    # Save results
    print(f"\nSaving results to {args.output_path}")
    with open(args.output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Done! Processed {len(results)} instances using {args.num_workers} GPUs.")
    print(f"Accuracy@{args.top_k}: {accuracy_at_k:.2%} ({samples_with_hits}/{total_instances} samples have ground truth in top-{args.top_k})")
    print(f"Overall hit rate: {overall_hit_rate:.2%} ({total_hits}/{total_gt_files} ground truth files found)")
    print(f"Average processing time: {elapsed_time/len(dataset):.2f}s per instance")
    print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
