from pathlib import Path
from typing import Dict, List, Optional

from datasets import DatasetDict, load_dataset

from config import Config

from .logger import logger


class DatasetHandler:
    """Handles different dataset formats and normalizes them to a common structure."""

    SUPPORTED_DATASETS = {
        "princeton-nlp/SWE-bench_Lite": {
            "type": "swe_bench",
            "fields": {
                "instance_id": "instance_id",
                "repo": "repo",
                "base_commit": "base_commit",
                "problem_statement": "problem_statement",
                "patch": "patch",
            },
        },
        "princeton-nlp/SWE-bench_verified": {
            "type": "swe_bench",
            "fields": {
                "instance_id": "instance_id",
                "repo": "repo",
                "base_commit": "base_commit",
                "problem_statement": "problem_statement",
                "patch": "patch",
            },
        },
        "princeton-nlp/SWE-bench": {
            "type": "swe_bench",
            "fields": {
                "instance_id": "instance_id",
                "repo": "repo",
                "base_commit": "base_commit",
                "problem_statement": "problem_statement",
                "patch": "patch",
            },
        },
        # SWE-Gym variants
        "SWE-Gym/SWE-Gym": {
            "type": "swe_gym",
            "fields": {
                "instance_id": "id",
                "repo": "repo",
                "base_commit": "base_commit",
                "problem_statement": "problem_statement",
                "patch": "patch",
            },
        },
        "SWE-Gym/SWE-Gym-Raw": {
            "type": "swe_gym",
            "fields": {
                "instance_id": "id",
                "repo": "repo",
                "base_commit": "base_commit",
                "problem_statement": "problem_statement",
                "patch": "patch",
            },
        },
        # SWE-Fixer variants
        "internlm/SWE-Fixer-Train-110K": {
            "type": "swe_fixer",
            "fields": {
                "instance_id": "id",
                "repo": "repo",
                "base_commit": "base_commit",
                "problem_statement": "problem_statement",
                "patch": "patch",
            },
        },
        # SWE-Smith
        "SWE-bench/SWE-smith": {
            "type": "swe_smith",
            "fields": {
                "instance_id": "id",
                "repo": "repo",
                "base_commit": "base_commit",
                "problem_statement": "problem_statement",
                "patch": "patch",
            },
        },
    }

    def __init__(self, config: Config):
        self.config = config
        self.dataset_names, self.split_names = self._parse_dataset_names()
        self.dataset_configs = self._get_dataset_configs()

    def _parse_dataset_names(self) -> tuple[List[str], List[str]]:
        """Parse dataset names to extract base names and splits."""
        import re

        dataset_names_str = self.config.dataset_name

        # Regex to match dataset patterns:
        # 1. dataset[split1,split2,...] - dataset with splits
        # 2. dataset - dataset without splits
        dataset_pattern = r"([^,\[\]]+(?:\[[^\]]*\])?)"

        # Find all dataset matches
        matches = re.findall(dataset_pattern, dataset_names_str)

        if len(matches) == 1:
            # Single dataset
            dataset_name, splits = self._parse_single_dataset(matches[0])
            return dataset_name, [splits]  # Wrap in list for consistency
        elif len(matches) > 1:
            # Multiple datasets
            dataset_names = []
            split_names = []

            for match in matches:
                match = match.strip()
                if match:  # Skip empty matches
                    dataset_name, splits = self._parse_single_dataset(match)
                    dataset_names.extend(dataset_name)
                    split_names.append(splits)  # Keep splits as a list for each dataset

            return dataset_names, split_names
        else:
            # No valid matches
            raise ValueError(
                f"Invalid dataset specification: {dataset_names_str}. "
                f"Please use format 'dataset[split]' or 'dataset1,dataset2'. "
                f"Examples: 'princeton-nlp/SWE-bench_Lite[test]', 'princeton-nlp/SWE-bench[train,dev]'"
            )

    def _parse_single_dataset(self, dataset_str: str) -> tuple[List[str], List[str]]:
        """Parse a single dataset string to extract name and splits."""
        # Single dataset with split specification
        if "[" in dataset_str and "]" in dataset_str:
            base_name = dataset_str.split("[")[0]
            splits_str = dataset_str.split("[")[1].split("]")[0]
            # Handle comma-separated splits
            split_names = [s.strip() for s in splits_str.split(",")]
            return [base_name], split_names

        # Check for /split format (only for specific patterns like dataset/split)
        elif "/" in dataset_str and not any(org in dataset_str for org in ["princeton-nlp/", "SWE-Gym/", "SWE-bench/", "internlm/"]):
            # This is a split specification, not a dataset path
            parts = dataset_str.split("/")
            if len(parts) >= 2:
                base_name = "/".join(parts[:-1])
                split_name = parts[-1]
                return [base_name], [split_name]

        # No split specified - load all available splits
        return [dataset_str], []

    def _get_dataset_configs(self) -> List[Dict]:
        """Get configurations for all datasets."""
        configs = []
        for dataset_name in self.dataset_names:
            if dataset_name not in self.SUPPORTED_DATASETS:
                raise ValueError(f"Dataset '{dataset_name}' is not supported. " f"Supported datasets: {list(self.SUPPORTED_DATASETS.keys())}")
            configs.append(self.SUPPORTED_DATASETS[dataset_name])
        return configs

    def load_dataset(self) -> DatasetDict:
        """Load and normalize multiple datasets."""

        # Check if dataset is already saved
        existing_dataset = self._load_existing_dataset()
        if existing_dataset is not None:
            logger().success(f"📚 Loaded existing dataset from cache")
            return existing_dataset

        logger().highlight(f"📚 Loading {len(self.dataset_names)} datasets")

        all_merged_data = []

        # Process each dataset
        for i, (dataset_name, dataset_config) in enumerate(zip(self.dataset_names, self.dataset_configs)):
            logger().info(f"🔄 Loading dataset {i+1}/{len(self.dataset_names)}: {dataset_name}")

            # Load the raw dataset
            raw_dataset = load_dataset(dataset_name)

            # Normalize the dataset
            normalized_dataset = self._normalize_dataset(raw_dataset, dataset_config)

            # Get the split name for this dataset
            requested_splits = self.split_names[i] if i < len(self.split_names) else []

            # Track samples from this dataset for duplicate analysis
            dataset_samples = []

            # If no splits specified, load all available splits
            if not requested_splits:
                logger().debug(f"📂 No splits specified, loading all available splits: {list(normalized_dataset.keys())}")
                for split_name, split_data in normalized_dataset.items():
                    logger().success(f"➕ Adding {len(split_data)} samples from '{split_name}' split")
                    dataset_samples.extend(split_data)
            else:
                # Load only requested splits
                for split_name in requested_splits:
                    if split_name in normalized_dataset:
                        split_data = normalized_dataset[split_name]
                        logger().success(f"➕ Adding {len(split_data)} samples from '{split_name}' split")
                        dataset_samples.extend(split_data)
                    else:
                        logger().warning(f"⚠️  Warning: Split '{split_name}' not found in dataset, skipping")

            # Add samples from this dataset to the main list
            all_merged_data.extend(dataset_samples)

        # Create a new dataset with all merged data
        from datasets import Dataset

        # Deduplicate based on base_commit + patch_hash for true uniqueness
        logger().info(f"🔍 Starting duplicate removal process for {len(all_merged_data)} samples...")

        # Create a dictionary to track unique samples by base_commit + patch_hash
        unique_samples = {}
        duplicates_found = 0
        duplicate_details = {}

        import hashlib

        for sample in all_merged_data:
            base_commit = sample.get("base_commit", "")
            patch = sample.get("patch", "")
            patch_hash = hashlib.md5(patch.encode()).hexdigest()

            # Create unique key from base_commit + patch_hash
            unique_key = f"{base_commit}_{patch_hash}"

            if unique_key in unique_samples:
                duplicates_found += 1
                # Track which samples had this duplicate
                if unique_key not in duplicate_details:
                    duplicate_details[unique_key] = []
                duplicate_details[unique_key].append(sample)
                # Keep the first occurrence
                continue
            unique_samples[unique_key] = sample

        # Convert back to list
        deduplicated_data = list(unique_samples.values())

        # Store original count for statistics (before deduplication)
        self.total_original_samples = len(all_merged_data)
        self.total_final_samples = len(deduplicated_data)

        if duplicates_found > 0:
            logger().warning(f"🗑️  Removed {duplicates_found} duplicate samples based on base_commit + patch_hash")
            logger().success(f"✅ Final dataset: {len(deduplicated_data)} unique samples")

            # Show some examples of duplicates if in debug mode
            if self.config.debug and duplicate_details:
                logger().highlight(f"📋 Sample duplicate base_commit + patch_hash combinations:")
                for i, (unique_key, samples) in enumerate(list(duplicate_details.items())[:10]):
                    instance_ids = [s.get("instance_id", "unknown") for s in samples]
                    base_commit = samples[0].get("base_commit", "unknown")
                    patch_hash = hashlib.md5(samples[0].get("patch", "").encode()).hexdigest()[:8]
                    logger().verbose(f"    {base_commit}_{patch_hash}: {len(samples)} samples with instance_ids: {instance_ids}")
        else:
            logger().success(f"✅ No duplicates found - all {len(deduplicated_data)} samples are unique")

        # Clean the data to ensure it's serializable and keep only essential features
        logger().info(f"🧹 Cleaning data for serialization and keeping only essential features...")
        cleaned_data = self._clean_data_for_serialization(deduplicated_data)

        # Log feature reduction info
        if deduplicated_data:
            original_features = set(deduplicated_data[0].keys())
            cleaned_features = set(cleaned_data[0].keys())
            removed_features = original_features - cleaned_features
            logger().info(f"📊 Kept {len(cleaned_features)} essential features: {', '.join(sorted(cleaned_features))}")
            if removed_features:
                logger().debug(f"🗑️ Removed {len(removed_features)} unused features: {', '.join(sorted(removed_features))}")

        merged_dataset = Dataset.from_list(cleaned_data)
        merged_dataset = DatasetDict({"merged": merged_dataset})
        logger().success(f"🎉 Created merged dataset with {len(deduplicated_data)} unique samples from {len(self.dataset_names)} datasets")

        # Apply debug mode filtering if enabled
        if self.config.debug:
            logger().highlight(f"🔧 DEBUG MODE: Limiting dataset to 5 samples for faster testing")
            merged_dataset = DatasetDict({"merged": merged_dataset["merged"].select(range(min(5, len(merged_dataset["merged"]))))})

        # Save the final dataset
        self._save_dataset(merged_dataset)

        return merged_dataset

    def _clean_data_for_serialization(self, data_list: List[Dict]) -> List[Dict]:
        """Clean data to ensure it's serializable to Arrow format and keep only essential features."""
        import json
        from datetime import datetime

        # Only keep the features we actually use in our code localization system
        ESSENTIAL_FEATURES = {
            "instance_id",
            "repo",
            "base_commit",
            "patch",
            "problem_statement",
        }

        def clean_value(value):
            """Recursively clean a value to ensure it's serializable."""
            if isinstance(value, datetime):
                return value.isoformat()
            elif isinstance(value, (list, tuple)):
                # Convert lists to strings to avoid Arrow serialization issues
                if len(value) == 0:
                    return ""
                elif len(value) == 1:
                    return str(clean_value(value[0]))
                else:
                    # Join multiple items with a separator
                    cleaned_items = [clean_value(item) for item in value]
                    return " | ".join(str(item) for item in cleaned_items)
            elif isinstance(value, dict):
                # Convert dicts to strings to avoid Arrow serialization issues
                cleaned_dict = {k: clean_value(v) for k, v in value.items()}
                return json.dumps(cleaned_dict)
            elif hasattr(value, "__dict__"):
                # Handle objects with __dict__ attribute
                return str(value)
            elif hasattr(value, "isoformat"):
                # Handle date-like objects
                return value.isoformat()
            else:
                # Try to serialize to ensure it's JSON serializable
                try:
                    json.dumps(value)
                    return value
                except (TypeError, ValueError):
                    return str(value)

        cleaned_list = []
        total_samples = len(data_list)

        # Add progress bar for data cleaning
        from tqdm import tqdm

        logger().info(f"🔄 Cleaning {total_samples} samples...")

        for sample in tqdm(data_list, desc="Cleaning data", unit="samples", leave=False):
            cleaned_sample = {}
            # Only keep essential features
            for key, value in sample.items():
                if key in ESSENTIAL_FEATURES:
                    cleaned_sample[key] = clean_value(value)
            cleaned_list.append(cleaned_sample)

        logger().success(f"✅ Successfully cleaned {total_samples} samples")
        return cleaned_list

    def _load_existing_dataset(self) -> Optional[DatasetDict]:
        """Check if dataset is already saved and load it if found."""
        import hashlib
        from pathlib import Path

        # Create hash from datasets configuration
        datasets_str = self.config.dataset_name
        datasets_hash = hashlib.md5(datasets_str.encode()).hexdigest()[:8]

        # Check if dataset directory exists
        dataset_dir = Path(self.config.data_dir) / "hf_datasets" / f"dataset_{datasets_hash}"

        if dataset_dir.exists():
            try:
                # Try to load the dataset
                from datasets import DatasetDict

                dataset = DatasetDict.load_from_disk(str(dataset_dir))

                # Verify it has the expected structure
                if "merged" in dataset and len(dataset["merged"]) > 0:
                    logger().info(f"📚 Found existing dataset: {dataset_dir}")
                    logger().info(f"📊 Dataset hash: {datasets_hash} (from: {datasets_str})")
                    logger().info(f"📈 Loaded {len(dataset['merged'])} samples from cache")

                    # Load and display dataset info if available
                    info_file = dataset_dir / "dataset_info.json"
                    if info_file.exists():
                        import json

                        with open(info_file, "r") as f:
                            info = json.load(f)
                            stats = info.get("processing_statistics", {})
                            logger().info(
                                f"📋 Original processing: {stats.get('total_original_samples', 'unknown')} → {stats.get('total_final_samples', 'unknown')} samples ({stats.get('duplication_rate', 'unknown')}% duplicates removed)"
                            )

                    return dataset
                else:
                    logger().warning(f"⚠️ Found dataset directory but no valid 'merged' split: {dataset_dir}")
            except Exception as e:
                logger().warning(f"⚠️ Failed to load existing dataset: {e}")

        return None

    def _save_dataset_info(
        self,
        save_dir: Path,
        datasets_str: str,
        datasets_hash: str,
        dataset: DatasetDict,
    ) -> None:
        """Save comprehensive dataset information to a JSON file."""
        import json
        from datetime import datetime

        # Calculate statistics from the dataset processing
        total_original_samples = getattr(self, "total_original_samples", 0)
        total_final_samples = getattr(self, "total_final_samples", len(dataset["merged"]))
        duplicates_removed = total_original_samples - total_final_samples

        dataset_info = {
            "dataset_configuration": {
                "datasets_string": datasets_str,
                "datasets_hash": datasets_hash,
                "dataset_names": self.dataset_names,
                "split_names": self.split_names,
                "dataset_configs": [
                    {"name": name, "type": config["type"], "fields": config["fields"]}
                    for name, config in zip(self.dataset_names, self.dataset_configs)
                ],
                "dataset_save_location": str(save_dir),
            },
            "processing_statistics": {
                "total_original_samples": total_original_samples,
                "total_final_samples": total_final_samples,
                "duplicates_removed": duplicates_removed,
                "duplication_rate": (round(duplicates_removed / total_original_samples * 100, 2) if total_original_samples > 0 else 0),
                "retention_rate": (round(total_final_samples / total_original_samples * 100, 2) if total_original_samples > 0 else 0),
            },
            "deduplication_strategy": {
                "method": "base_commit + patch_hash",
                "description": "Samples are considered duplicates if they have the same base_commit and identical patch content (hash)",
            },
            "essential_features": [
                "instance_id",
                "repo",
                "base_commit",
                "patch",
                "problem_statement",
            ],
            "processing_timestamp": datetime.now().isoformat(),
            "processing_metadata": {
                "debug_mode": self.config.debug,
                "data_dir": str(self.config.data_dir),
            },
        }

        # Save to JSON file
        info_file = save_dir / "dataset_info.json"
        try:
            with open(info_file, "w") as f:
                json.dump(dataset_info, f, indent=2)
            logger().info(f"📋 Saved dataset info to: {info_file}")
        except Exception as e:
            logger().warning(f"⚠️ Failed to save dataset info: {e}")

    def _save_dataset(self, dataset: DatasetDict) -> None:
        """Save the final dataset to disk."""
        import hashlib
        from pathlib import Path

        # Create hash from datasets configuration
        datasets_str = self.config.dataset_name
        datasets_hash = hashlib.md5(datasets_str.encode()).hexdigest()[:8]

        # Create save directory
        save_dir = Path(self.config.data_dir) / "hf_datasets" / f"dataset_{datasets_hash}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save the dataset
        try:
            dataset.save_to_disk(str(save_dir))
            logger().success(f"💾 Saved dataset to: {save_dir}")
            logger().info(f"📊 Dataset hash: {datasets_hash} (from: {datasets_str})")

            # Save comprehensive dataset info
            self._save_dataset_info(save_dir, datasets_str, datasets_hash, dataset)

        except Exception as e:
            logger().warning(f"⚠️ Failed to save dataset: {e}")

    def _normalize_dataset(self, raw_dataset: DatasetDict, dataset_config: Dict) -> DatasetDict:
        """Normalize dataset to common format."""
        dataset_type = dataset_config["type"]
        field_mapping = dataset_config["fields"]

        normalized_splits = {}

        for split_name, split_data in raw_dataset.items():
            print(f"Normalizing split: {split_name} ({len(split_data)} samples)")

            # Apply dataset-specific normalization
            if dataset_type == "swe_bench":
                normalized_split = self._normalize_swe_bench(split_data, field_mapping)
            elif dataset_type == "swe_gym":
                normalized_split = self._normalize_swe_gym(split_data, field_mapping)
            elif dataset_type == "swe_fixer":
                normalized_split = self._normalize_swe_fixer(split_data, field_mapping)
            elif dataset_type == "swe_smith":
                normalized_split = self._normalize_swe_smith(split_data, field_mapping)
            else:
                normalized_split = self._normalize_generic(split_data, field_mapping)

            normalized_splits[split_name] = normalized_split

        return DatasetDict(normalized_splits)

    def _normalize_swe_bench(self, split_data, field_mapping: Dict) -> DatasetDict:
        """Normalize SWE-bench datasets."""
        # SWE-bench datasets are already in the correct format
        return split_data

    def _normalize_swe_gym(self, split_data, field_mapping: Dict) -> DatasetDict:
        """Normalize SWE-Gym datasets."""

        def normalize_example(example):
            normalized = {}

            # Map fields according to the mapping
            for target_field, source_field in field_mapping.items():
                if source_field in example:
                    normalized[target_field] = example[source_field]
                else:
                    # Handle missing fields
                    if target_field == "instance_id":
                        normalized[target_field] = f"swg_{example.get('id', 'unknown')}"
                    elif target_field == "problem_statement":
                        normalized[target_field] = example.get("problem_statement", "No problem statement provided")
                    elif target_field == "patch":
                        normalized[target_field] = example.get("patch", "")
                    else:
                        normalized[target_field] = example.get(source_field, "")

            return normalized

        return split_data.map(normalize_example)

    def _normalize_swe_fixer(self, split_data, field_mapping: Dict) -> DatasetDict:
        """Normalize SWE-Fixer datasets."""

        def normalize_example(example):
            normalized = {}

            # Map fields according to the mapping
            for target_field, source_field in field_mapping.items():
                if source_field in example:
                    normalized[target_field] = example[source_field]
                else:
                    # Handle missing fields
                    if target_field == "instance_id":
                        normalized[target_field] = f"swf_{example.get('id', 'unknown')}"
                    elif target_field == "problem_statement":
                        normalized[target_field] = example.get("problem_statement", "No problem statement provided")
                    elif target_field == "patch":
                        normalized[target_field] = example.get("patch", "")
                    else:
                        normalized[target_field] = example.get(source_field, "")

            return normalized

        return split_data.map(normalize_example)

    def _normalize_swe_smith(self, split_data, field_mapping: Dict) -> DatasetDict:
        """Normalize SWE-Smith datasets."""

        def normalize_example(example):
            normalized = {}

            # Map fields according to the mapping
            for target_field, source_field in field_mapping.items():
                if source_field in example:
                    normalized[target_field] = example[source_field]
                else:
                    # Handle missing fields
                    if target_field == "instance_id":
                        normalized[target_field] = f"sws_{example.get('id', 'unknown')}"
                    elif target_field == "problem_statement":
                        normalized[target_field] = example.get("problem_statement", "No problem statement provided")
                    elif target_field == "patch":
                        normalized[target_field] = example.get("patch", "")
                    else:
                        normalized[target_field] = example.get(source_field, "")

            return normalized

        return split_data.map(normalize_example)

    def _normalize_generic(self, split_data, field_mapping: Dict) -> DatasetDict:
        """Generic normalization for unknown dataset types."""

        def normalize_example(example):
            normalized = {}

            # Map fields according to the mapping
            for target_field, source_field in field_mapping.items():
                if source_field in example:
                    normalized[target_field] = example[source_field]
                else:
                    # Provide defaults for missing fields
                    if target_field == "instance_id":
                        normalized[target_field] = f"gen_{example.get('id', 'unknown')}"
                    elif target_field == "problem_statement":
                        normalized[target_field] = example.get("problem_statement", "No problem statement provided")
                    elif target_field == "patch":
                        normalized[target_field] = example.get("patch", "")
                    else:
                        normalized[target_field] = example.get(source_field, "")

            return normalized

        return split_data.map(normalize_example)

    def validate_dataset(self, dataset: DatasetDict) -> bool:
        """Validate that the dataset has the required fields."""
        required_fields = [
            "instance_id",
            "repo",
            "base_commit",
            "problem_statement",
            "patch",
        ]

        for split_name, split_data in dataset.items():
            if len(split_data) == 0:
                continue

            sample = split_data[0]
            missing_fields = [field for field in required_fields if field not in sample]

            if missing_fields:
                print(f"Warning: Split '{split_name}' is missing required fields: {missing_fields}")
                return False

        return True

    def get_dataset_info(self, dataset: DatasetDict) -> Dict:
        """Get information about the dataset."""
        # Since we now support multiple datasets, we'll show the first one's type or "mixed"
        dataset_type = "mixed" if len(self.dataset_configs) > 1 else self.dataset_configs[0]["type"]

        info = {
            "dataset_name": self.config.dataset_name,
            "dataset_type": dataset_type,
            "splits": {},
            "total_samples": 0,
        }

        for split_name, split_data in dataset.items():
            split_info = {
                "num_samples": len(split_data),
                "fields": list(split_data.column_names) if len(split_data) > 0 else [],
            }
            info["splits"][split_name] = split_info
            info["total_samples"] += len(split_data)

        return info

    @staticmethod
    def list_supported_datasets() -> List[str]:
        """List all supported datasets."""
        return list(DatasetHandler.SUPPORTED_DATASETS.keys())

    @staticmethod
    def get_dataset_description(dataset_name: str) -> Optional[str]:
        """Get description of a dataset."""
        descriptions = {
            "princeton-nlp/SWE-bench_Lite": "Lightweight version of SWE-bench with verified instances (splits: test)",
            "princeton-nlp/SWE-bench_verified": "SWE-bench dataset with verified instances only (splits: test)",
            "princeton-nlp/SWE-bench": "Full SWE-bench dataset with train/test splits (splits: train, test)",
            "SWE-Gym/SWE-Gym": "SWE-Gym dataset with software engineering tasks (splits: train, test, validation)",
            "SWE-Gym/SWE-Gym-Raw": "Raw SWE-Gym dataset with additional instances (splits: train, test, validation)",
            "internlm/SWE-Fixer-Train-110K": "SWE-Fixer training dataset with 110k instances from InternLM (splits: train, test)",
            "SWE-bench/SWE-smith": "SWE-Smith dataset with software engineering tasks (splits: train, test, validation)",
        }
        return descriptions.get(dataset_name, "No description available")
