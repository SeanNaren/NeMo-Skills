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
        self.line_level_metrics_50 = []
        self.line_level_metrics_25 = []
        self.line_level_metrics_75 = []
        self.total_gt_locations = 0
        self.total_pred_locations = 0
    
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
                "line_precision_50": 0.0,
                "line_recall_50": 0.0,
                "line_f1_50": 0.0,
                "line_exact_match_50": 0.0,
                "line_avg_overlap_50": 0.0
            }
        
        eval_result = eval_status[0]
        file_metrics = eval_result.get('file_level', {})
        line_metrics_50 = eval_result.get('line_level_overlap_50', {})
        
        return {
            "file_precision": file_metrics.get('precision', 0.0),
            "file_recall": file_metrics.get('recall', 0.0),
            "file_f1": file_metrics.get('f1', 0.0),
            "file_exact_match": file_metrics.get('exact_match', 0.0),
            "line_precision_50": line_metrics_50.get('precision', 0.0),
            "line_recall_50": line_metrics_50.get('recall', 0.0),
            "line_f1_50": line_metrics_50.get('f1', 0.0),
            "line_exact_match_50": line_metrics_50.get('exact_match', 0.0),
            "line_avg_overlap_50": line_metrics_50.get('average_overlap', 0.0)
        }

    @classmethod
    def get_incorrect_sample(cls, prediction: dict) -> dict:
        """Return a sample representing completely incorrect predictions."""
        return {
            "eval_status": [{
                "file_level": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0},
                "line_level_overlap_50": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0, "average_overlap": 0.0},
                "line_level_overlap_25": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0, "average_overlap": 0.0},
                "line_level_overlap_75": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0, "average_overlap": 0.0},
                "ground_truth_count": 0,
                "predicted_count": 0
            }]
        }

    def update(self, predictions):
        """Update metrics with new predictions."""
        super().update(predictions)
        
        # Collect detailed metrics for aggregate computation
        for prediction in predictions:
            eval_status = prediction.get('eval_status', [])
            if eval_status and isinstance(eval_status[0], dict):
                eval_result = eval_status[0]
                
                # Store individual metrics for later aggregation
                if 'file_level' in eval_result:
                    self.file_level_metrics.append(eval_result['file_level'])
                if 'line_level_overlap_50' in eval_result:
                    self.line_level_metrics_50.append(eval_result['line_level_overlap_50'])
                if 'line_level_overlap_25' in eval_result:
                    self.line_level_metrics_25.append(eval_result['line_level_overlap_25'])
                if 'line_level_overlap_75' in eval_result:
                    self.line_level_metrics_75.append(eval_result['line_level_overlap_75'])
                
                # Accumulate totals
                self.total_gt_locations += eval_result.get('ground_truth_count', 0)
                self.total_pred_locations += eval_result.get('predicted_count', 0)
        
        self._compute_pass_at_k(predictions=predictions)

    def _compute_aggregate_metrics(self, metrics_list):
        """Compute aggregate metrics from a list of individual metrics."""
        if not metrics_list:
            return {}
        
        aggregate = {}
        metric_names = ["precision", "recall", "f1", "exact_match"]
        
        for metric in metric_names:
            values = [m.get(metric, 0.0) for m in metrics_list if metric in m]
            aggregate[metric] = sum(values) / len(values) if values else 0.0
        
        # Add average_overlap for line-level metrics
        if any("average_overlap" in m for m in metrics_list):
            overlap_values = [m.get("average_overlap", 0.0) for m in metrics_list if "average_overlap" in m]
            aggregate["average_overlap"] = sum(overlap_values) / len(overlap_values) if overlap_values else 0.0
        
        return aggregate

    def get_metrics(self):
        """Get metrics including aggregate evaluation results."""
        metrics_dict = super().get_metrics()
        
        # Compute aggregate metrics for each evaluation type
        aggregate_file = self._compute_aggregate_metrics(self.file_level_metrics)
        aggregate_line_50 = self._compute_aggregate_metrics(self.line_level_metrics_50)
        aggregate_line_25 = self._compute_aggregate_metrics(self.line_level_metrics_25)
        aggregate_line_75 = self._compute_aggregate_metrics(self.line_level_metrics_75)
        
        # Add aggregate metrics to each aggregation mode
        for agg_mode in metrics_dict.keys():
            metrics_dict[agg_mode].update({
                # File-level aggregate metrics
                "agg_file_precision": aggregate_file.get('precision', 0.0) * 100,
                "agg_file_recall": aggregate_file.get('recall', 0.0) * 100,
                "agg_file_f1": aggregate_file.get('f1', 0.0) * 100,
                "agg_file_exact_match": aggregate_file.get('exact_match', 0.0) * 100,
                
                # Line-level aggregate metrics (50% overlap)
                "agg_line_precision_50": aggregate_line_50.get('precision', 0.0) * 100,
                "agg_line_recall_50": aggregate_line_50.get('recall', 0.0) * 100,
                "agg_line_f1_50": aggregate_line_50.get('f1', 0.0) * 100,
                "agg_line_exact_match_50": aggregate_line_50.get('exact_match', 0.0) * 100,
                "agg_line_avg_overlap_50": aggregate_line_50.get('average_overlap', 0.0) * 100,
                
                # Line-level aggregate metrics (25% overlap)
                "agg_line_precision_25": aggregate_line_25.get('precision', 0.0) * 100,
                "agg_line_recall_25": aggregate_line_25.get('recall', 0.0) * 100,
                "agg_line_f1_25": aggregate_line_25.get('f1', 0.0) * 100,
                "agg_line_exact_match_25": aggregate_line_25.get('exact_match', 0.0) * 100,
                "agg_line_avg_overlap_25": aggregate_line_25.get('average_overlap', 0.0) * 100,
                
                # Line-level aggregate metrics (75% overlap)
                "agg_line_precision_75": aggregate_line_75.get('precision', 0.0) * 100,
                "agg_line_recall_75": aggregate_line_75.get('recall', 0.0) * 100,
                "agg_line_f1_75": aggregate_line_75.get('f1', 0.0) * 100,
                "agg_line_exact_match_75": aggregate_line_75.get('exact_match', 0.0) * 100,
                "agg_line_avg_overlap_75": aggregate_line_75.get('average_overlap', 0.0) * 100,
                
                # Summary statistics
                "total_ground_truth_locations": self.total_gt_locations,
                "total_predicted_locations": self.total_pred_locations,
                "avg_gt_locations_per_case": self.total_gt_locations / self.total if self.total > 0 else 0.0,
                "avg_pred_locations_per_case": self.total_pred_locations / self.total if self.total > 0 else 0.0
            })
        
        return metrics_dict

    def reset(self):
        """Reset all metrics and aggregation data."""
        super().reset()
        self.file_level_metrics = []
        self.line_level_metrics_50 = []
        self.line_level_metrics_25 = []
        self.line_level_metrics_75 = []
        self.total_gt_locations = 0
        self.total_pred_locations = 0

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
            "total_gt_locs": format_count,
            "total_pred_locs": format_count,
            
            # File-level results (aggregate across all test cases)
            "file_precision": as_percentage,
            "file_recall": as_percentage,
            "file_f1": as_percentage,
            
            # Line-level results (50% overlap threshold)
            "line_precision_50": as_percentage,
            "line_recall_50": as_percentage,
            "line_f1_50": as_percentage,
            "line_overlap_50": as_percentage,
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
        
        # Summary section
        print(f"\n📊 SUMMARY STATISTICS")
        print(f"{'─' * 50}")
        print(f"{'Test Cases:':<25} {metrics.get('num_entries', 0):>8}")
        print(f"{'Ground Truth Locations:':<25} {metrics.get('total_gt_locations', 0):>8}")
        print(f"{'Predicted Locations:':<25} {metrics.get('total_pred_locations', 0):>8}")
        print(f"{'Avg GT per Case:':<25} {metrics.get('avg_gt_per_case', 0):>8.2f}")
        print(f"{'Avg Pred per Case:':<25} {metrics.get('avg_pred_per_case', 0):>8.2f}")
        
        # File-level section
        print(f"\n📁 FILE-LEVEL METRICS (Individual Predictions)")
        print(f"{'─' * 50}")
        self._print_metric_group(metrics, "file", [
            ("Precision", "file_precision"),
            ("Recall", "file_recall"),
            ("F1 Score", "file_f1"),
            ("Exact Match", "file_exact_match")
        ])
        
        print(f"\n📁 FILE-LEVEL METRICS (Aggregate)")
        print(f"{'─' * 50}")
        self._print_metric_group(metrics, "agg_file", [
            ("Precision", "agg_file_precision"),
            ("Recall", "agg_file_recall"),
            ("F1 Score", "agg_file_f1"),
            ("Exact Match", "agg_file_exact_match")
        ])
        
        # Line-level section with different thresholds
        print(f"\n📏 LINE-LEVEL METRICS (Individual Predictions - 50% Overlap)")
        print(f"{'─' * 50}")
        self._print_metric_group(metrics, "line_50", [
            ("Precision", "line_precision_50"),
            ("Recall", "line_recall_50"),
            ("F1 Score", "line_f1_50"),
            ("Exact Match", "line_exact_match_50"),
            ("Avg Overlap", "line_avg_overlap_50")
        ])
        
        print(f"\n📏 LINE-LEVEL METRICS (Aggregate - Multiple Thresholds)")
        print(f"{'─' * 70}")
        
        # Multi-threshold comparison table
        thresholds = [25, 50, 75]
        metrics_names = ["Precision", "Recall", "F1 Score", "Exact Match", "Avg Overlap"]
        
        print(f"{'Metric':<15} {'25% Overlap':<12} {'50% Overlap':<12} {'75% Overlap':<12}")
        print(f"{'─' * 15} {'─' * 12} {'─' * 12} {'─' * 12}")
        
        for metric_name in metrics_names:
            row = f"{metric_name:<15}"
            for threshold in thresholds:
                metric_key = f"agg_line_{metric_name.lower().replace(' ', '_')}_{threshold}"
                if metric_name == "F1 Score":
                    metric_key = f"agg_line_f1_{threshold}"
                elif metric_name == "Exact Match":
                    metric_key = f"agg_line_exact_match_{threshold}"
                elif metric_name == "Avg Overlap":
                    metric_key = f"agg_line_avg_overlap_{threshold}"
                elif metric_name == "Precision":
                    metric_key = f"agg_line_precision_{threshold}"
                elif metric_name == "Recall":
                    metric_key = f"agg_line_recall_{threshold}"
                
                value = metrics.get(metric_key, 0.0)
                row += f" {value:>10.1f}%"
            print(row)
        
        print(f"\n{'=' * 80}")
        print("💡 Key Insights:")
        file_f1 = metrics.get('agg_file_f1', 0.0)
        line_f1_50 = metrics.get('agg_line_f1_50', 0.0)
        overlap_50 = metrics.get('agg_line_avg_overlap_50', 0.0)
        
        print(f"   • File-level F1: {file_f1:.1f}% - How well the model identifies relevant files")
        print(f"   • Line-level F1 (50%): {line_f1_50:.1f}% - How well the model identifies relevant line ranges")
        print(f"   • Average Overlap: {overlap_50:.1f}% - How closely predicted ranges match ground truth")
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
        aggregate_line_50 = self._compute_aggregate_metrics(self.line_level_metrics_50)
        aggregate_line_25 = self._compute_aggregate_metrics(self.line_level_metrics_25)
        aggregate_line_75 = self._compute_aggregate_metrics(self.line_level_metrics_75)
        
        # Add aggregate metrics to each aggregation mode with readable names
        for agg_mode in metrics_dict.keys():
            metrics_dict[agg_mode].update({
                # Summary statistics with shorter names for table
                "total_gt_locs": self.total_gt_locations,
                "total_pred_locs": self.total_pred_locations,
                "avg_gt_per_case": self.total_gt_locations / self.total if self.total > 0 else 0.0,
                "avg_pred_per_case": self.total_pred_locations / self.total if self.total > 0 else 0.0,
                
                # File-level aggregate metrics (displayed in table)
                "file_precision": aggregate_file.get('precision', 0.0) * 100,
                "file_recall": aggregate_file.get('recall', 0.0) * 100,
                "file_f1": aggregate_file.get('f1', 0.0) * 100,
                "file_exact_match": aggregate_file.get('exact_match', 0.0) * 100,
                
                # Line-level aggregate metrics (50% overlap - displayed in table)
                "line_precision_50": aggregate_line_50.get('precision', 0.0) * 100,
                "line_recall_50": aggregate_line_50.get('recall', 0.0) * 100,
                "line_f1_50": aggregate_line_50.get('f1', 0.0) * 100,
                "line_exact_match_50": aggregate_line_50.get('exact_match', 0.0) * 100,
                "line_overlap_50": aggregate_line_50.get('average_overlap', 0.0) * 100,
                
                # Additional metrics (stored but not displayed in main table)
                "agg_line_precision_25": aggregate_line_25.get('precision', 0.0) * 100,
                "agg_line_recall_25": aggregate_line_25.get('recall', 0.0) * 100,
                "agg_line_f1_25": aggregate_line_25.get('f1', 0.0) * 100,
                "agg_line_exact_match_25": aggregate_line_25.get('exact_match', 0.0) * 100,
                "agg_line_avg_overlap_25": aggregate_line_25.get('average_overlap', 0.0) * 100,
                
                "agg_line_precision_75": aggregate_line_75.get('precision', 0.0) * 100,
                "agg_line_recall_75": aggregate_line_75.get('recall', 0.0) * 100,
                "agg_line_f1_75": aggregate_line_75.get('f1', 0.0) * 100,
                "agg_line_exact_match_75": aggregate_line_75.get('exact_match', 0.0) * 100,
                "agg_line_avg_overlap_75": aggregate_line_75.get('average_overlap', 0.0) * 100,
            })
        
        return metrics_dict
