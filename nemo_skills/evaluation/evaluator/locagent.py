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

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import field
from typing import List, Dict, Tuple, Set

from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.inference.eval.locagent_utils.utils import extract_locations_from_patch
from nemo_skills.utils import get_logger_name, nested_dataclass, unroll_files

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class LocalAgentEvaluatorConfig:
    timeout: float = 30.0
    num_parallel_requests: int = 20


def evaluate_file_level_accuracy(ground_truth_locations: List[Dict], predicted_locations: List[Dict]) -> Dict[str, float]:
    """
    Calculate file-level prediction accuracy metrics.
    
    Args:
        ground_truth_locations: List of ground truth location dictionaries
        predicted_locations: List of predicted location dictionaries
    
    Returns:
        Dictionary containing precision, recall, f1, and exact_match metrics
    """
    if not ground_truth_locations and not predicted_locations:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact_match": 1.0}
    
    if not ground_truth_locations:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "exact_match": 0.0}
    
    if not predicted_locations:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0}
    
    # Extract unique file paths (with defensive programming for missing file_path key)
    ground_truth_files = {loc['file_path'] for loc in ground_truth_locations if 'file_path' in loc}
    predicted_files = {loc['file_path'] for loc in predicted_locations if 'file_path' in loc}
    
    # Calculate metrics
    true_positives = len(ground_truth_files.intersection(predicted_files))
    false_positives = len(predicted_files - ground_truth_files)
    false_negatives = len(ground_truth_files - predicted_files)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_match = 1.0 if ground_truth_files == predicted_files else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match
    }


def calculate_line_overlap(gt_start: int, gt_end: int, pred_start: int, pred_end: int) -> float:
    """
    Calculate the overlap ratio between two line ranges.
    
    Args:
        gt_start, gt_end: Ground truth line range
        pred_start, pred_end: Predicted line range
    
    Returns:
        Overlap ratio as a float between 0 and 1
    """
    # Calculate intersection
    intersection_start = max(gt_start, pred_start)
    intersection_end = min(gt_end, pred_end)
    
    if intersection_start > intersection_end:
        return 0.0  # No overlap
    
    intersection_size = intersection_end - intersection_start + 1
    
    # Calculate union
    union_start = min(gt_start, pred_start)
    union_end = max(gt_end, pred_end)
    union_size = union_end - union_start + 1
    
    return intersection_size / union_size


def evaluate_line_level_accuracy(ground_truth_locations: List[Dict], predicted_locations: List[Dict], 
                                overlap_threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate line-level prediction accuracy metrics with overlap consideration.
    
    Args:
        ground_truth_locations: List of ground truth location dictionaries
        predicted_locations: List of predicted location dictionaries
        overlap_threshold: Minimum overlap ratio to consider a match
    
    Returns:
        Dictionary containing precision, recall, f1, exact_match, and average_overlap metrics
    """
    if not ground_truth_locations and not predicted_locations:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact_match": 1.0, "average_overlap": 1.0}
    
    if not ground_truth_locations:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "exact_match": 0.0, "average_overlap": 0.0}
    
    if not predicted_locations:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0, "average_overlap": 0.0}
    
    # Group locations by file path
    gt_by_file = {}
    pred_by_file = {}
    
    for loc in ground_truth_locations:
        if 'file_path' not in loc:
            continue  # Skip locations without file_path
        file_path = loc['file_path']
        if file_path not in gt_by_file:
            gt_by_file[file_path] = []
        gt_by_file[file_path].append(loc)
    
    for loc in predicted_locations:
        if 'file_path' not in loc:
            continue  # Skip locations without file_path
        file_path = loc['file_path']
        if file_path not in pred_by_file:
            pred_by_file[file_path] = []
        pred_by_file[file_path].append(loc)
    
    all_overlaps = []
    matched_gt = set()
    matched_pred = set()
    
    # For each ground truth location, find the best matching predicted location
    for file_path, gt_locs in gt_by_file.items():
        if file_path not in pred_by_file:
            continue
            
        pred_locs = pred_by_file[file_path]
        
        for i, gt_loc in enumerate(gt_locs):
            best_overlap = 0.0
            best_pred_idx = -1
            
            for j, pred_loc in enumerate(pred_locs):
                if (file_path, j) in matched_pred:
                    continue
                    
                overlap = calculate_line_overlap(
                    gt_loc['start_line'], gt_loc['end_line'],
                    pred_loc['start_line'], pred_loc['end_line']
                )
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_pred_idx = j
            
            if best_overlap >= overlap_threshold and best_pred_idx != -1:
                matched_gt.add((file_path, i))
                matched_pred.add((file_path, best_pred_idx))
                all_overlaps.append(best_overlap)
    
    # Calculate metrics
    total_gt = sum(len(locs) for locs in gt_by_file.values())
    total_pred = sum(len(locs) for locs in pred_by_file.values())
    
    true_positives = len(matched_gt)
    false_positives = total_pred - len(matched_pred)
    false_negatives = total_gt - len(matched_gt)
    
    precision = true_positives / total_pred if total_pred > 0 else 0.0
    recall = true_positives / total_gt if total_gt > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    average_overlap = sum(all_overlaps) / len(all_overlaps) if all_overlaps else 0.0
    
    # Exact match: all ground truth locations have exact line matches
    exact_matches = 0
    for file_path, gt_locs in gt_by_file.items():
        if file_path not in pred_by_file:
            continue
            
        pred_locs = pred_by_file[file_path]
        
        for gt_loc in gt_locs:
            for pred_loc in pred_locs:
                if (gt_loc['start_line'] == pred_loc['start_line'] and 
                    gt_loc['end_line'] == pred_loc['end_line']):
                    exact_matches += 1
                    break
    
    exact_match_ratio = exact_matches / total_gt if total_gt > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match_ratio,
        "average_overlap": average_overlap
    }


def _execute_single_test(args):
    """Helper function to execute a single test case."""
    eval_config, elem_idx, ground_truth_locations, locations = args
    
    # Calculate file-level accuracy
    file_level_metrics = evaluate_file_level_accuracy(ground_truth_locations, locations)
    
    # Calculate line-level accuracy with different overlap thresholds
    line_level_metrics_50 = evaluate_line_level_accuracy(ground_truth_locations, locations, overlap_threshold=0.5)
    line_level_metrics_25 = evaluate_line_level_accuracy(ground_truth_locations, locations, overlap_threshold=0.25)
    line_level_metrics_75 = evaluate_line_level_accuracy(ground_truth_locations, locations, overlap_threshold=0.75)
    
    output_dict = {
        "file_level": file_level_metrics,
        "line_level_overlap_50": line_level_metrics_50,
        "line_level_overlap_25": line_level_metrics_25,
        "line_level_overlap_75": line_level_metrics_75,
        "ground_truth_count": len(ground_truth_locations),
        "predicted_count": len(locations),
        "ground_truth_files": list({loc['file_path'] for loc in ground_truth_locations if 'file_path' in loc}),
        "predicted_files": list({loc['file_path'] for loc in locations if 'file_path' in loc})
    }
    
    # Log detailed information for debugging
    LOG.info(f"Element {elem_idx} evaluation:")
    LOG.info(f"  Ground truth locations: {len(ground_truth_locations)}")
    LOG.info(f"  Predicted locations: {len(locations)}")
    LOG.info(f"  File-level F1: {file_level_metrics['f1']:.3f}")
    LOG.info(f"  Line-level F1 (50% overlap): {line_level_metrics_50['f1']:.3f}")
    
    return elem_idx, output_dict


def eval_metrics(eval_config, locagent_data):
    json_idx = {}

    for prob_data in locagent_data:
        json_idx[prob_data['instance_id']] = locagent_data.index(prob_data)

    # Prepare all tasks for parallel execution
    tasks = []
    for elem_idx, elem in enumerate(locagent_data):        
        if elem["status"] != "success":
            continue
        ground_truth_locations = extract_locations_from_patch(elem["patch"])
        tasks.append((eval_config, elem_idx, ground_truth_locations, elem["locations"]))
        # for step_id, full_generation in elem['generation'].items():
        #     instance_id, subtask_step = step_id.split('.')
        #     json_content = locagent_data[json_idx[instance_id]]
        #     tasks.append((eval_config, elem_idx, full_generation, json_content, subtask_step))

    # Initialize status_lists with correct structure
    status_lists = [[] for _ in range(len(locagent_data))]

    # Execute tasks in parallel
    with ThreadPoolExecutor(max_workers=eval_config.num_parallel_requests) as executor:
        results = list(executor.map(_execute_single_test, tasks))

    # Organize results back into the original structure
    for elem_idx, output_dict in results:
        status_lists[elem_idx].append(output_dict)

    return status_lists


def eval_locagent(cfg):
    eval_config = LocalAgentEvaluatorConfig(**cfg.eval_config)
    for file in unroll_files(cfg.input_files):
        with open(file, 'rt', encoding='utf-8') as fin:
            data = [json.loads(line) for line in fin]
        status_lists = eval_metrics(eval_config, data)
        with open(file, 'wt', encoding='utf-8') as fout:
            for idx, elem in enumerate(data):
                elem['eval_status'] = status_lists[idx]
                fout.write(json.dumps(elem) + "\n")
