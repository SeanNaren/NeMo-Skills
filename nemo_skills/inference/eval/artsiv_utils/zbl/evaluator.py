from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class LocationMatch:
    """Represents a matched prediction-ground truth pair."""

    predicted: Dict
    ground_truth: Dict
    overlap: int
    extra_lines: int


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    popok: float

    # File-level metrics
    file_accuracy: float  # Percentage of ground truth files that were correctly identified
    file_matched: int  # Number of files correctly identified
    file_total_gt: int  # Total number of ground truth files

    # Chunk-level metrics
    chunk_accuracy: float  # Percentage of ground truth chunks that were correctly identified
    chunk_matched: int  # Number of chunks correctly identified
    chunk_total_gt: int  # Total number of ground truth chunks


@dataclass
class ParsedLocation:
    """Represents a parsed location with file path and line range."""

    raw: str
    file_path: str
    start_line: int
    end_line: int


class LocationParser:
    """Handles parsing of location strings into structured format."""

    @staticmethod
    def parse_locations(location_strings: List[str]) -> List[ParsedLocation]:
        """Parse location strings into structured format."""
        parsed = []

        for loc_str in location_strings:
            if ":" not in loc_str:
                continue

            file_part, line_part = loc_str.rsplit(":", 1)

            # Extract line numbers
            line_part = line_part.replace("L", "")

            try:
                if "-" in line_part:
                    start_str, end_str = line_part.split("-", 1)
                    start_line = int(start_str)
                    end_line = int(end_str)
                else:
                    start_line = end_line = int(line_part)
            except ValueError:
                continue

            # Normalize file path
            normalized_path = LocationParser._normalize_file_path(file_part)

            parsed.append(
                ParsedLocation(
                    raw=loc_str,
                    file_path=normalized_path,
                    start_line=start_line,
                    end_line=end_line,
                )
            )

        return parsed

    @staticmethod
    def _normalize_file_path(file_path: str) -> str:
        """Normalize file path to handle full paths vs relative paths."""
        # Remove common repository prefixes
        prefixes_to_remove = ["data/repos/", "repos/"]

        normalized = file_path
        for prefix in prefixes_to_remove:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]

        # Remove commit hash or instance ID (first part if it looks like a hash/ID)
        if "/" in normalized:
            parts = normalized.split("/", 1)
            if len(parts) > 1:
                # Check if first part looks like a hash/ID (alphanumeric, 8+ chars)
                first_part = parts[0]
                if (
                    len(first_part) >= 8 and first_part.replace("-", "").replace("_", "").isalnum() and not first_part.startswith(".")
                ):  # Not a hidden directory
                    normalized = parts[1]

        # Handle duplicate directory prefixes (e.g., repo/repo/file.py -> repo/file.py)
        path_parts = normalized.split("/")
        if len(path_parts) >= 2 and path_parts[0] == path_parts[1]:
            # Remove the first duplicate part
            normalized = "/".join(path_parts[1:])

        return normalized


class LocationMatcher:
    """Handles matching between predicted and ground truth locations."""

    @staticmethod
    def calculate_overlap(pred: ParsedLocation, gt: ParsedLocation) -> int:
        """Calculate line overlap between two locations."""
        overlap_start = max(pred.start_line, gt.start_line)
        overlap_end = min(pred.end_line, gt.end_line)
        return max(0, overlap_end - overlap_start + 1)

    @staticmethod
    def calculate_extra_lines(pred: ParsedLocation, gt: ParsedLocation) -> int:
        """Calculate extra lines in prediction beyond ground truth."""
        pred_lines = pred.end_line - pred.start_line + 1
        overlap = LocationMatcher.calculate_overlap(pred, gt)
        return max(0, pred_lines - overlap)

    @staticmethod
    def match_locations(
        gt_parsed: List[ParsedLocation], pred_parsed: List[ParsedLocation]
    ) -> Tuple[List[LocationMatch], List[ParsedLocation], List[ParsedLocation]]:
        """Match predictions with ground truth locations."""
        # Group ground truth by file
        gt_by_file = {}
        for gt in gt_parsed:
            if gt.file_path not in gt_by_file:
                gt_by_file[gt.file_path] = []
            gt_by_file[gt.file_path].append(gt)

        matched_pairs = []
        unmatched_gt = gt_parsed.copy()
        unmatched_pred = pred_parsed.copy()

        # First pass: match predictions to best ground truth
        for pred in pred_parsed:
            if pred.file_path not in gt_by_file:
                continue

            best_match = None
            best_overlap = 0

            for gt in gt_by_file[pred.file_path]:
                if gt in unmatched_gt:
                    overlap = LocationMatcher.calculate_overlap(pred, gt)
                    if overlap > 0:
                        best_overlap = overlap
                        best_match = gt

            if best_match and best_overlap > 0:
                matched_pairs.append(
                    LocationMatch(
                        predicted=pred,
                        ground_truth=best_match,
                        overlap=best_overlap,
                        extra_lines=LocationMatcher.calculate_extra_lines(pred, best_match),
                    )
                )
                unmatched_gt.remove(best_match)
                unmatched_pred.remove(pred)

        return matched_pairs, unmatched_gt, unmatched_pred


class MetricCalculator:
    """Calculates evaluation metrics."""

    @staticmethod
    def calculate_popok_metric(
        gt_parsed: List[ParsedLocation],
        pred_parsed: List[ParsedLocation],
        matched_pairs: List[LocationMatch],
    ) -> float:
        """Calculate the custom popok metric."""
        if not gt_parsed and not pred_parsed:
            return 1.0

        if not gt_parsed or not pred_parsed:
            return 0.0

        # File-level metrics
        gt_files = set(gt.file_path for gt in gt_parsed)
        pred_files = set(pred.file_path for pred in pred_parsed)

        correctly_predicted_files = set()
        for gt in gt_parsed:
            for pred in pred_parsed:
                if gt.file_path == pred.file_path:
                    correctly_predicted_files.add(gt.file_path)
                    break

        file_accuracy = len(correctly_predicted_files) / len(gt_files) if gt_files else 1.0

        # Line-level metrics
        total_gt_lines = sum(gt.end_line - gt.start_line + 1 for gt in gt_parsed)
        total_pred_lines = sum(pred.end_line - pred.start_line + 1 for pred in pred_parsed)
        correctly_predicted_lines = sum(pair.overlap for pair in matched_pairs)

        # Calculate line accuracy with better handling of partial matches
        # For cases where prediction is smaller but precise, use a more forgiving metric
        if len(matched_pairs) > 0:
            # Check if predictions are generally smaller than ground truth
            total_pred_lines = sum(pred.end_line - pred.start_line + 1 for pred in pred_parsed)
            if total_pred_lines <= total_gt_lines:
                # If predictions are smaller, use overlap-based accuracy instead of coverage
                line_accuracy = correctly_predicted_lines / total_gt_lines if total_gt_lines > 0 else 1.0
                # Boost accuracy for precise predictions
                line_accuracy = min(1.0, line_accuracy * 2.0)
            else:
                line_accuracy = correctly_predicted_lines / total_gt_lines if total_gt_lines > 0 else 1.0
        else:
            line_accuracy = correctly_predicted_lines / total_gt_lines if total_gt_lines > 0 else 1.0

        # Bonus for precise predictions (when prediction is smaller but within ground truth)
        precision_bonus = 0.0
        for pair in matched_pairs:
            gt_lines = pair.ground_truth.end_line - pair.ground_truth.start_line + 1
            pred_lines = pair.predicted.end_line - pair.predicted.start_line + 1
            if pred_lines <= gt_lines and pair.overlap > 0:
                # If prediction is smaller than ground truth but has overlap, give bonus
                # More generous bonus for precise predictions
                precision_bonus += 0.3 * (pair.overlap / gt_lines)

        # Tightness penalty (only for predictions larger than ground truth)
        total_extra_lines = sum(pair.extra_lines for pair in matched_pairs)
        tightness_penalty = min(0.2 * total_extra_lines / max(total_gt_lines, 1), 0.5)

        # Weighted combination
        line_score = 0.6 * line_accuracy
        file_score = 0.2 * file_accuracy
        tightness_score = 0.1 * (1.0 - tightness_penalty)
        precision_score = 0.1 * min(precision_bonus, 1.0)  # Cap precision bonus at 0.1

        popok_score = line_score + file_score + tightness_score + precision_score

        # For perfect matches, ensure we return 1.0
        if file_accuracy == 1.0 and line_accuracy >= 1.0 and len(matched_pairs) == len(gt_parsed) and len(matched_pairs) == len(pred_parsed):
            return 1.0

        return max(0.0, min(1.0, popok_score))

    @staticmethod
    def calculate_file_accuracy(
        gt_parsed: List[ParsedLocation],
        pred_parsed: List[ParsedLocation],
        matched_pairs: List[LocationMatch],
    ) -> Dict:
        """Calculate file-level accuracy metrics."""
        # Get unique files
        gt_files = set(gt.file_path for gt in gt_parsed)
        pred_files = set(pred.file_path for pred in pred_parsed)

        # Count correctly predicted files (files that exist in both sets)
        correctly_predicted_files = gt_files.intersection(pred_files)

        file_total_gt = len(gt_files)
        file_matched = len(correctly_predicted_files)
        file_accuracy = file_matched / file_total_gt if file_total_gt > 0 else 1.0

        return {
            "file_accuracy": file_accuracy,
            "file_matched": file_matched,
            "file_total_gt": file_total_gt,
        }

    @staticmethod
    def calculate_chunk_accuracy(gt_parsed: List[ParsedLocation], matched_pairs: List[LocationMatch]) -> Dict:
        """Calculate chunk-level accuracy metrics."""
        chunk_total_gt = len(gt_parsed)
        chunk_matched = len(matched_pairs)
        chunk_accuracy = chunk_matched / chunk_total_gt if chunk_total_gt > 0 else 1.0

        return {
            "chunk_accuracy": chunk_accuracy,
            "chunk_matched": chunk_matched,
            "chunk_total_gt": chunk_total_gt,
        }


class LocationEvaluator:
    """Evaluates predicted code locations against ground truth locations."""

    def evaluate(self, ground_truth_locations: List[str], predicted_locations: List[str]) -> Dict:
        """Evaluate predicted locations against ground truth."""
        # Parse locations
        gt_parsed = LocationParser.parse_locations(ground_truth_locations)
        pred_parsed = LocationParser.parse_locations(predicted_locations)

        # If no ground truth, something is wrong
        if not gt_parsed:
            import warnings

            warnings.warn(f"No ground truth locations found! This shouldn't happen. Raw ground truth: {ground_truth_locations}")
            return {
                "popok": 0.0,  # Bad result since this shouldn't happen
                "file_accuracy": 0.0,
                "file_matched": 0,
                "file_total_gt": 0,
                "chunk_accuracy": 0.0,
                "chunk_matched": 0,
                "chunk_total_gt": 0,
            }

        # If no predictions but have ground truth, nothing was found
        if not pred_parsed:
            return {
                "popok": 0.0,  # Found nothing
                "file_accuracy": 0.0,  # Found nothing
                "file_matched": 0,
                "file_total_gt": len(set(gt.file_path for gt in gt_parsed)),
                "chunk_accuracy": 0.0,  # Found nothing
                "chunk_matched": 0,
                "chunk_total_gt": len(gt_parsed),
            }

        # Match locations
        matched_pairs, _, _ = LocationMatcher.match_locations(gt_parsed, pred_parsed)

        # Calculate metrics
        popok_score = MetricCalculator.calculate_popok_metric(gt_parsed, pred_parsed, matched_pairs)
        file_metrics = MetricCalculator.calculate_file_accuracy(gt_parsed, pred_parsed, matched_pairs)
        chunk_metrics = MetricCalculator.calculate_chunk_accuracy(gt_parsed, matched_pairs)

        return {
            "popok": popok_score,
            **file_metrics,
            **chunk_metrics,
        }
