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
        Dictionary containing precision, recall, f1, exact_match, and accuracy metrics
    """
    if not ground_truth_locations and not predicted_locations:
        # Both empty - this is a perfect match in a sense (nothing needed, nothing predicted)
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact_match": 1.0, "accuracy": 1.0}
    
    if not ground_truth_locations:
        # No ground truth but made predictions - all predictions are false positives
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0, "accuracy": 0.0}
    
    if not predicted_locations:
        # Ground truth exists but no predictions - all ground truth are false negatives
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0, "accuracy": 0.0}
    
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
    
    # Calculate accuracy as Jaccard index (IoU) - intersection over union of file sets
    # This is a common measure for set similarity
    if len(ground_truth_files) == 0 and len(predicted_files) == 0:
        accuracy = 1.0  # Both empty
    elif len(ground_truth_files.union(predicted_files)) == 0:
        accuracy = 0.0  # Should not happen, but defensive programming
    else:
        accuracy = len(ground_truth_files.intersection(predicted_files)) / len(ground_truth_files.union(predicted_files))
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match,
        "accuracy": accuracy
    }


# Note: The following overlap-based functions are kept for potential future use,
# but are not currently used in favor of the chunk containment metrics which better
# align with the use case where predictions encompassing ground truth get full credit.

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


def evaluate_chunk_containment_metrics(ground_truth_locations: List[Dict], predicted_locations: List[Dict]) -> Dict[str, float]:
    """
    Calculate chunk-level containment metrics where predictions are rewarded for completely encompassing ground truth.
    
    Args:
        ground_truth_locations: List of ground truth location dictionaries
        predicted_locations: List of predicted location dictionaries
    
    Returns:
        Dictionary containing coverage_recall, avg_prediction_tightness, and precision metrics
    """
    if not ground_truth_locations and not predicted_locations:
        return {
            "coverage_recall": 0.0,  # No GT to cover, so 0
            "avg_prediction_tightness": 0.0,  # No predictions, so 0
            "precision": 0.0,  # No predictions can be useful, so 0
            "covered_chunks": 0,
            "total_chunks": 0,
            "useful_predictions": 0,
            "total_predictions": 0
        }
    
    if not ground_truth_locations:
        return {
            "coverage_recall": 0.0,  # No GT to cover, so 0
            "avg_prediction_tightness": 0.0,
            "precision": 0.0,  # All predictions are useless (no GT), so 0
            "covered_chunks": 0,
            "total_chunks": 0,
            "useful_predictions": 0,
            "total_predictions": len(predicted_locations)
        }
    
    if not predicted_locations:
        return {
            "coverage_recall": 0.0,
            "avg_prediction_tightness": 0.0,
            "precision": 0.0,
            "covered_chunks": 0,
            "total_chunks": len(ground_truth_locations),
            "useful_predictions": 0,
            "total_predictions": 0
        }
    
    # Track which ground truths are covered and by which predictions
    covered_gt_indices = set()
    useful_pred_indices = set()
    tightness_scores = []
    
    # For each ground truth, check if any prediction fully covers it
    for gt_idx, gt_loc in enumerate(ground_truth_locations):
        if 'file_path' not in gt_loc or 'start_line' not in gt_loc or 'end_line' not in gt_loc:
            continue
            
        gt_file = gt_loc['file_path']
        gt_start = gt_loc['start_line']
        gt_end = gt_loc['end_line']
        
        best_tightness = 0.0
        found_coverage = False
        
        # Check all predictions for this ground truth
        for pred_idx, pred_loc in enumerate(predicted_locations):
            if 'file_path' not in pred_loc or 'start_line' not in pred_loc or 'end_line' not in pred_loc:
                continue
                
            # Must be in the same file
            if pred_loc['file_path'] != gt_file:
                continue
                
            pred_start = pred_loc['start_line']
            pred_end = pred_loc['end_line']
            
            # Check if prediction fully contains ground truth
            if pred_start <= gt_start and pred_end >= gt_end:
                found_coverage = True
                useful_pred_indices.add(pred_idx)
                
                # Calculate tightness (how tight is the prediction around the ground truth)
                gt_length = gt_end - gt_start + 1
                pred_length = pred_end - pred_start + 1
                tightness = gt_length / pred_length if pred_length > 0 else 0.0
                
                # Keep the best (tightest) prediction for this ground truth
                if tightness > best_tightness:
                    best_tightness = tightness
        
        if found_coverage:
            covered_gt_indices.add(gt_idx)
            tightness_scores.append(best_tightness)
    
    # Calculate metrics
    total_gt = len([gt for gt in ground_truth_locations if all(k in gt for k in ['file_path', 'start_line', 'end_line'])])
    total_pred = len([pred for pred in predicted_locations if all(k in pred for k in ['file_path', 'start_line', 'end_line'])])
    
    coverage_recall = len(covered_gt_indices) / total_gt if total_gt > 0 else 0.0
    avg_tightness = sum(tightness_scores) / len(tightness_scores) if tightness_scores else 0.0
    precision = len(useful_pred_indices) / total_pred if total_pred > 0 else 0.0
    
    return {
        "coverage_recall": coverage_recall,
        "avg_prediction_tightness": avg_tightness,
        "precision": precision,
        "covered_chunks": len(covered_gt_indices),
        "total_chunks": total_gt,
        "useful_predictions": len(useful_pred_indices),
        "total_predictions": total_pred
    }


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
    
    # Calculate chunk containment metrics
    chunk_containment_metrics = evaluate_chunk_containment_metrics(ground_truth_locations, locations)
    
    output_dict = {
        "file_level": file_level_metrics,
        "chunk_containment": chunk_containment_metrics,
        "ground_truth_locations": ground_truth_locations,
        "ground_truth_count": len(ground_truth_locations),
        "predicted_count": len(locations),
        "ground_truth_files": list({loc['file_path'] for loc in ground_truth_locations if 'file_path' in loc}),
        "predicted_files": list({loc['file_path'] for loc in locations if 'file_path' in loc}),
    }
    
    # Log detailed information for debugging
    LOG.info(f"Element {elem_idx} evaluation:")
    LOG.info(f"  Ground truth locations: {len(ground_truth_locations)}")
    LOG.info(f"  Predicted locations: {len(locations)}")
    LOG.info(f"  File-level F1: {file_level_metrics['f1']:.3f}")
    LOG.info(f"  File-level Accuracy: {file_level_metrics['accuracy']:.3f}")
    LOG.info(f"  Chunk Coverage Recall: {chunk_containment_metrics['coverage_recall']:.3f}")
    LOG.info(f"  Avg Prediction Tightness: {chunk_containment_metrics['avg_prediction_tightness']:.3f}")
    
    return elem_idx, output_dict


def eval_metrics(eval_config, locagent_data):
    json_idx = {}

    for prob_data in locagent_data:
        json_idx[prob_data['instance_id']] = locagent_data.index(prob_data)

    # Initialize status_lists with correct structure
    status_lists = [[] for _ in range(len(locagent_data))]
    
    # Prepare all tasks for parallel execution
    tasks = []
    successful_samples = 0
    failed_samples = 0
    skipped_samples = 0
    
    for elem_idx, elem in enumerate(locagent_data):        
        if elem["status"] == "skipped":
            skipped_samples += 1
            # Assign zero metrics to skipped samples
            ground_truth_locations = extract_locations_from_patch(elem["patch"])
            skip_metrics = {
                "file_level": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0, "accuracy": 0.0},
                "chunk_containment": {
                    "coverage_recall": 0.0,
                    "avg_prediction_tightness": 0.0,
                    "precision": 0.0,
                    "covered_chunks": 0,
                    "total_chunks": len(ground_truth_locations),
                    "useful_predictions": 0,
                    "total_predictions": 0
                },
                "ground_truth_locations": ground_truth_locations,
                "ground_truth_count": len(ground_truth_locations),
                "predicted_count": 0,
                "ground_truth_files": list({loc['file_path'] for loc in ground_truth_locations if 'file_path' in loc}),
                "predicted_files": [],
                "is_skipped_sample": True,  # Mark this explicitly as a skipped sample
                "skip_reason": elem.get("reason", "unknown")
            }
            status_lists[elem_idx].append(skip_metrics)
        elif elem["status"] != "success":
            failed_samples += 1
            # Assign zero metrics to failed samples
            ground_truth_locations = extract_locations_from_patch(elem["patch"])
            zero_metrics = {
                "file_level": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0, "accuracy": 0.0},
                "chunk_containment": {
                    "coverage_recall": 0.0,
                    "avg_prediction_tightness": 0.0,
                    "precision": 0.0,
                    "covered_chunks": 0,
                    "total_chunks": len(ground_truth_locations),
                    "useful_predictions": 0,
                    "total_predictions": 0
                },
                "ground_truth_locations": ground_truth_locations,
                "ground_truth_count": len(ground_truth_locations),
                "predicted_count": 0,
                "ground_truth_files": list({loc['file_path'] for loc in ground_truth_locations if 'file_path' in loc}),
                "predicted_files": [],
                "is_failed_sample": True,  # Mark this explicitly as a failed sample
            }
            status_lists[elem_idx].append(zero_metrics)
            continue
        successful_samples += 1
        ground_truth_locations = extract_locations_from_patch(elem["patch"])
        tasks.append((eval_config, elem_idx, ground_truth_locations, elem["locations"]))
        # for step_id, full_generation in elem['generation'].items():
        #     instance_id, subtask_step = step_id.split('.')
        #     json_content = locagent_data[json_idx[instance_id]]
        #     tasks.append((eval_config, elem_idx, full_generation, json_content, subtask_step))

    # Log processing statistics
    total_samples = len(locagent_data)
    LOG.info(f"Processing statistics:")
    LOG.info(f"  Total samples: {total_samples}")
    LOG.info(f"  Successful samples: {successful_samples}")
    LOG.info(f"  Failed samples: {failed_samples}")
    LOG.info(f"  Skipped samples: {skipped_samples}")
    LOG.info(f"  Success rate: {(successful_samples / total_samples * 100):.1f}%")
    
    # Store processing statistics to be included in metrics
    processing_stats = {
        "total_samples": total_samples,
        "successful_samples": successful_samples, 
        "failed_samples": failed_samples,
        "skipped_samples": skipped_samples,
        "success_rate": (successful_samples / total_samples * 100) if total_samples > 0 else 0.0
    }

    # Execute tasks in parallel
    with ThreadPoolExecutor(max_workers=eval_config.num_parallel_requests) as executor:
        results = list(executor.map(_execute_single_test, tasks))

    # Organize results back into the original structure
    for elem_idx, output_dict in results:
        status_lists[elem_idx].append(output_dict)

    return status_lists, processing_stats


def eval_locagent(cfg):
    eval_config = LocalAgentEvaluatorConfig(**cfg.eval_config)
    for file in unroll_files(cfg.input_files):
        with open(file, 'rt', encoding='utf-8') as fin:
            data = []
            null_indices = []
            for line_idx, line in enumerate(fin):
                parsed = json.loads(line)
                if parsed is None:
                    LOG.warning(f"Skipping null entry at line {line_idx + 1} in {file}")
                    null_indices.append(line_idx)
                else:
                    data.append(parsed)
        
        if not data:
            LOG.warning(f"No valid data to evaluate in {file}")
            continue
            
        status_lists, processing_stats = eval_metrics(eval_config, data)
        
        # Reconstruct the full output including null entries
        all_outputs = []
        data_idx = 0
        for line_idx in range(len(data) + len(null_indices)):
            if line_idx in null_indices:
                all_outputs.append(None)
            else:
                elem = data[data_idx]
                elem['eval_status'] = status_lists[data_idx]
                # Add processing statistics to the first element (as metadata)
                if data_idx == 0:
                    elem['processing_stats'] = processing_stats
                all_outputs.append(elem)
                data_idx += 1
        
        with open(file, 'wt', encoding='utf-8') as fout:
            for elem in all_outputs:
                fout.write(json.dumps(elem) + "\n")
