# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_skills.evaluation.metrics.base import BaseMetrics


class CodeMetrics(BaseMetrics):
    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        return {
            "passing_base_tests": prediction['is_correct'],
            "passing_plus_tests": prediction['is_correct-plus'],
        }

    @classmethod
    def get_incorrect_sample(cls, prediction: dict) -> dict:
        return {"is_correct": False, "is_correct-plus": False}

    def update(self, predictions):
        super().update(predictions)
        self._compute_pass_at_k(predictions=predictions)


class LiveCodeBenchMetrics(BaseMetrics):
    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        return {
            "accuracy": prediction['graded_list'][0],
        }

    @classmethod
    def get_incorrect_sample(cls, prediction: dict) -> dict:
        return {"graded_list": [False]}

    def update(self, predictions):
        super().update(predictions)
        self._compute_pass_at_k(predictions=predictions)


class SciCodeMetrics(BaseMetrics):
    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        subtask_status_list = prediction['eval_status']
        correct_subtasks = sum(subtask['process_status'] == 'completed' for subtask in subtask_status_list)
        return {
            'problem_accuracy': correct_subtasks == len(subtask_status_list),
            'subtask_accuracy': correct_subtasks,
        }

    @classmethod
    def get_incorrect_sample(cls, prediction: dict) -> dict:
        prediction = prediction.copy()
        subtask_status_list = prediction['eval_status']
        for subtask in subtask_status_list:
            subtask['process_status'] = 'error'
        prediction['eval_status'] = subtask_status_list
        return prediction

    def update(self, predictions):
        super().update(predictions)
        self.subtasks_total += len(predictions[0]['eval_status'])
        self._compute_pass_at_k(predictions)

    def get_metrics(self):
        metrics_dict = super().get_metrics()
        for agg_mode in self.eval_dict.keys():
            metrics_dict[agg_mode]["num_problems"] = metrics_dict[agg_mode].pop("num_entries")
            metrics_dict[agg_mode]["num_subtasks"] = self.subtasks_total
            # correcting subtask normalization
            metrics_dict[agg_mode]["subtask_accuracy"] *= self.total / self.subtasks_total

        return metrics_dict

    def reset(self):
        super().reset()
        self.subtasks_total = 0


class LocAgentMetrics(BaseMetrics):
    def __init__(self):
        super().__init__()
        self.file_level_metrics = []
        self.chunk_containment_metrics = []
        self.total_gt_locations = 0
        self.total_pred_locations = 0
        self.successful_samples = 0
        self.failed_samples = 0
        self.no_gt_samples = 0  # Track samples with no ground truth
    
    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        """Extract metrics from the prediction eval_status."""
        eval_status = prediction.get('eval_status', [])
        
        if not eval_status or not isinstance(eval_status[0], dict):
            # Fallback for empty or malformed eval_status
            return {
                "file_precision": 0.0,
                "file_recall": 0.0,
                "file_f1": 0.0,
                "file_exact_match": 0.0,
                "file_accuracy": 0.0,
                "chunk_coverage_recall": 0.0,
                "chunk_avg_tightness": 0.0,
                "chunk_precision": 0.0
            }
        
        eval_result = eval_status[0]
        file_metrics = eval_result.get('file_level', {})
        chunk_metrics = eval_result.get('chunk_containment', {})
        
        return {
            "file_precision": file_metrics.get('precision', 0.0),
            "file_recall": file_metrics.get('recall', 0.0),
            "file_f1": file_metrics.get('f1', 0.0),
            "file_exact_match": file_metrics.get('exact_match', 0.0),
            "file_accuracy": file_metrics.get('accuracy', 0.0),
            "chunk_coverage_recall": chunk_metrics.get('coverage_recall', 0.0),
            "chunk_avg_tightness": chunk_metrics.get('avg_prediction_tightness', 0.0),
            "chunk_precision": chunk_metrics.get('precision', 0.0)
        }

    @classmethod
    def get_incorrect_sample(cls, prediction: dict) -> dict:
        """Return a sample representing completely incorrect predictions."""
        return {
            "eval_status": [{
                "file_level": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0, "accuracy": 0.0},
                "chunk_containment": {
                    "coverage_recall": 0.0,
                    "avg_prediction_tightness": 0.0,
                    "precision": 0.0,
                    "covered_chunks": 0,
                    "total_chunks": 0,
                    "useful_predictions": 0,
                    "total_predictions": 0
                },
                "ground_truth_count": 0,
                "predicted_count": 0,
                "is_failed_sample": True
            }]
        }

    def update(self, predictions):
        """Update metrics with new predictions."""
        # Always call parent update to maintain compatibility
        super().update(predictions)
        
        # Only use processing_stats if this is the first call to update
        # (to avoid overwriting counts from multiple batches)
        if predictions and 'processing_stats' in predictions[0] and self.successful_samples == 0 and self.failed_samples == 0:
            processing_stats = predictions[0]['processing_stats']
            self.successful_samples = processing_stats.get('successful_samples', 0)
            self.failed_samples = processing_stats.get('failed_samples', 0)
        
        # Collect detailed metrics for aggregate computation
        for prediction in predictions:
            eval_status = prediction.get('eval_status', [])
            if eval_status and isinstance(eval_status[0], dict):
                eval_result = eval_status[0]
                
                # Check if this sample has no ground truth
                gt_count = eval_result.get('ground_truth_count', 0)
                if gt_count == 0:
                    self.no_gt_samples += 1
                    # Skip this sample - don't include in metrics
                    continue
                
                # Count samples if we haven't gotten the full count from processing_stats
                # This handles cases where processing_stats is incomplete or missing
                if self.successful_samples + self.failed_samples < self.total:
                    # Check if this was a successful or failed sample
                    # Use explicit flag if available, otherwise infer from metrics
                    is_failed_sample = eval_result.get('is_failed_sample', False)
                    if not is_failed_sample:
                        # Also check if status field indicates failure
                        if 'status' in prediction and prediction['status'] != 'success':
                            is_failed_sample = True
                        else:
                            # Fallback detection for samples without explicit flag
                            predicted_count = eval_result.get('predicted_count', 0)
                            file_metrics = eval_result.get('file_level', {})
                            is_failed_sample = (predicted_count == 0 and 
                                              file_metrics.get('precision', 0) == 0 and 
                                              file_metrics.get('recall', 0) == 0 and 
                                              file_metrics.get('f1', 0) == 0)
                    
                    if is_failed_sample:
                        self.failed_samples += 1
                    else:
                        self.successful_samples += 1
                
                # Store individual metrics for later aggregation (only for samples with GT)
                if 'file_level' in eval_result:
                    self.file_level_metrics.append(eval_result['file_level'])
                if 'chunk_containment' in eval_result:
                    self.chunk_containment_metrics.append(eval_result['chunk_containment'])
                
                # Accumulate totals
                self.total_gt_locations += eval_result.get('ground_truth_count', 0)
                self.total_pred_locations += eval_result.get('predicted_count', 0)
        
        self._compute_pass_at_k(predictions=predictions)

    def _compute_aggregate_metrics(self, metrics_list):
        """Compute aggregate metrics from a list of individual metrics."""
        if not metrics_list:
            return {}
        
        aggregate = {}
        
        # For file-level metrics
        file_metric_names = ["precision", "recall", "f1", "exact_match", "accuracy"]
        
        # For chunk containment metrics
        chunk_metric_names = ["coverage_recall", "avg_prediction_tightness", "precision"]
        
        # Determine which metrics to aggregate based on what's in the first item
        if metrics_list and len(metrics_list) > 0:
            first_item = metrics_list[0]
            if "accuracy" in first_item:
                # File-level metrics
                for metric in file_metric_names:
                    values = [m.get(metric, 0.0) for m in metrics_list if metric in m]
                    aggregate[metric] = sum(values) / len(values) if values else 0.0
            elif "coverage_recall" in first_item:
                # Chunk containment metrics
                for metric in chunk_metric_names:
                    values = [m.get(metric, 0.0) for m in metrics_list if metric in m]
                    aggregate[metric] = sum(values) / len(values) if values else 0.0
            else:
                # Generic metrics
                for metric in ["precision", "recall", "f1", "exact_match"]:
                    values = [m.get(metric, 0.0) for m in metrics_list if metric in m]
                    aggregate[metric] = sum(values) / len(values) if values else 0.0
        
        # Add average_overlap for line-level metrics
        if any("average_overlap" in m for m in metrics_list):
            overlap_values = [m.get("average_overlap", 0.0) for m in metrics_list if "average_overlap" in m]
            aggregate["average_overlap"] = sum(overlap_values) / len(overlap_values) if overlap_values else 0.0
        
        return aggregate



    def reset(self):
        """Reset all metrics and aggregation data."""
        super().reset()
        self.file_level_metrics = []
        self.chunk_containment_metrics = []
        self.total_gt_locations = 0
        self.total_pred_locations = 0
        self.successful_samples = 0
        self.failed_samples = 0
        self.no_gt_samples = 0

    def metrics_to_print(self):
        """Define which metrics to print and their formatting for better readability."""
        from nemo_skills.evaluation.metrics.base import as_percentage
        
        def format_count(value):
            return str(int(value))
        
        def format_ratio(value):
            return f"{value:.2f}"
        
        # Return a well-organized, readable set of metrics
        return {
            # Summary
            "num_entries": lambda x: str(int(x)),
            "samples_with_gt": format_count,
            "samples_no_gt": format_count,
            "total_gt_locs": format_count,
            "total_pred_locs": format_count,
            
            # File-level results (aggregate across all test cases)
            "file_precision": as_percentage,
            "file_recall": as_percentage,
            "file_f1": as_percentage,
            "file_accuracy": as_percentage,
            
            # Chunk containment results
            "chunk_coverage_recall": as_percentage,
            "chunk_avg_tightness": as_percentage,
            "chunk_precision": as_percentage,
        }

    def print_detailed_metrics(self, benchmark_name, metrics_data, eval_mode="pass@1"):
        """Print a detailed, well-organized metrics table."""
        if eval_mode not in metrics_data:
            print(f"No data available for evaluation mode: {eval_mode}")
            return
        
        metrics = metrics_data[eval_mode]
        
        print(f"\n{'=' * 80}")
        print(f" {benchmark_name} - Detailed LocAgent Evaluation Results ".center(80, '='))
        print(f"{'=' * 80}")
        
        # Processing Statistics section
        print(f"\n🔄 PROCESSING STATISTICS")
        print(f"{'─' * 50}")
        print(f"{'Total Samples:':<25} {metrics.get('total_samples', 0):>8}")
        print(f"{'  With Ground Truth:':<25} {metrics.get('samples_with_gt', 0):>8}")
        print(f"{'  No Ground Truth:':<25} {metrics.get('samples_no_gt', 0):>8}")
        print(f"{'Successful Samples:':<25} {metrics.get('successful_samples', 0):>8}")
        print(f"{'Failed Samples:':<25} {metrics.get('failed_samples', 0):>8}")
        print(f"{'Success Rate:':<25} {metrics.get('success_rate', 0):>8.1f}%")
        
        # Summary section
        print(f"\n📊 SUMMARY STATISTICS")
        print(f"{'─' * 50}")
        print(f"{'Test Cases:':<25} {metrics.get('num_entries', 0):>8}")
        print(f"{'Ground Truth Locations:':<25} {metrics.get('total_gt_locations', 0):>8}")
        print(f"{'Predicted Locations:':<25} {metrics.get('total_pred_locations', 0):>8}")
        print(f"{'Avg GT per Case:':<25} {metrics.get('avg_gt_per_case', 0):>8.2f}")
        print(f"{'Avg Pred per Case:':<25} {metrics.get('avg_pred_per_case', 0):>8.2f}")
        
        # File-level section
        print(f"\n📁 FILE-LEVEL METRICS")
        print(f"{'─' * 50}")
        self._print_metric_group(metrics, "", [
            ("Precision", "file_precision"),
            ("Recall", "file_recall"),
            ("F1 Score", "file_f1"),
            ("Accuracy", "file_accuracy"),
            ("Exact Match", "file_exact_match")
        ])
        
        # Chunk containment section
        print(f"\n📦 CHUNK CONTAINMENT METRICS")
        print(f"{'─' * 50}")
        self._print_metric_group(metrics, "", [
            ("Coverage Recall", "chunk_coverage_recall"),
            ("Avg Prediction Tightness", "chunk_avg_tightness"),
            ("Precision", "chunk_precision")
        ])
        
        print(f"\n{'=' * 80}")
        print("💡 Key Insights:")
        file_f1 = metrics.get('file_f1', 0.0)
        chunk_coverage = metrics.get('chunk_coverage_recall', 0.0)
        chunk_tightness = metrics.get('chunk_avg_tightness', 0.0)
        
        print(f"   • File-level F1: {file_f1:.1f}% - How well the model identifies relevant files")
        print(f"   • Chunk Coverage: {chunk_coverage:.1f}% - How many ground truth chunks are fully covered")
        print(f"   • Prediction Tightness: {chunk_tightness:.1f}% - How precise the predictions are around ground truth")
        print(f"{'=' * 80}\n")

    def _print_metric_group(self, metrics, prefix, metric_list):
        """Helper to print a group of related metrics."""
        for display_name, metric_key in metric_list:
            value = metrics.get(metric_key, 0.0)
            print(f"{'  ' + display_name + ':':<25} {value:>8.1f}%")

    def get_metrics(self):
        """Get metrics including aggregate evaluation results with better naming."""
        metrics_dict = super().get_metrics()
        
        # Compute aggregate metrics for each evaluation type
        aggregate_file = self._compute_aggregate_metrics(self.file_level_metrics)
        aggregate_chunk = self._compute_aggregate_metrics(self.chunk_containment_metrics)
        
        # Add aggregate metrics to each aggregation mode with readable names
        for agg_mode in metrics_dict.keys():
            # Use self.total which is the actual number of entries processed
            total_samples = self.total
            # Effective total excludes samples with no ground truth
            effective_total = total_samples - self.no_gt_samples
            
            # If we have explicit success/failure counts, calculate success rate from those
            if self.successful_samples > 0 or self.failed_samples > 0:
                counted_samples = self.successful_samples + self.failed_samples
                # Scale up the success rate if we only counted a subset
                if counted_samples < effective_total:
                    success_rate = (self.successful_samples / counted_samples * 100) if counted_samples > 0 else 0.0
                else:
                    success_rate = (self.successful_samples / effective_total * 100) if effective_total > 0 else 0.0
            else:
                # Fallback: if no explicit counts, assume all samples were successful
                success_rate = 100.0 if effective_total > 0 else 0.0
            
            metrics_dict[agg_mode].update({
                # Processing statistics
                "total_samples": total_samples,
                "samples_with_gt": effective_total,
                "samples_no_gt": self.no_gt_samples,
                "successful_samples": self.successful_samples,
                "failed_samples": self.failed_samples,
                "success_rate": success_rate,
                
                # Summary statistics with shorter names for table
                "total_gt_locs": self.total_gt_locations,
                "total_pred_locs": self.total_pred_locations,
                "avg_gt_per_case": self.total_gt_locations / effective_total if effective_total > 0 else 0.0,
                "avg_pred_per_case": self.total_pred_locations / effective_total if effective_total > 0 else 0.0,
                
                # File-level aggregate metrics (displayed in table)
                "file_precision": aggregate_file.get('precision', 0.0) * 100,
                "file_recall": aggregate_file.get('recall', 0.0) * 100,
                "file_f1": aggregate_file.get('f1', 0.0) * 100,
                "file_exact_match": aggregate_file.get('exact_match', 0.0) * 100,
                "file_accuracy": aggregate_file.get('accuracy', 0.0) * 100,
                
                # Chunk containment aggregate metrics (displayed in table)
                "chunk_coverage_recall": aggregate_chunk.get('coverage_recall', 0.0) * 100,
                "chunk_avg_tightness": aggregate_chunk.get('avg_prediction_tightness', 0.0) * 100,
                "chunk_precision": aggregate_chunk.get('precision', 0.0) * 100,
            })
        
        return metrics_dict
