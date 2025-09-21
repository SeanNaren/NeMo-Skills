import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union


class Logger:
    """Simple colorful logger with file output."""

    COLORS = {
        "BLUE": "\033[94m",  # Regular info - bright blue
        "GREEN": "\033[92m",  # Success - bright green
        "ORANGE": "\033[38;5;208m",  # Warnings - orange
        "RED": "\033[91m",  # Errors - bright red
        "PURPLE": "\033[95m",  # Metrics - bright purple
        "CYAN": "\033[96m",  # Additional info - cyan
        "YELLOW": "\033[93m",  # Highlights - yellow
        "GRAY": "\033[37m",  # Debug/verbose info - light gray
        "RESET": "\033[0m",  # Reset
    }

    # Class variable to store execution directory for this process
    _execution_dir = None

    # Global logger instance for centralized access
    _global_logger = None

    @classmethod
    def set_execution_dir(cls, execution_dir: Path):
        """Set the execution directory for all loggers in this process."""
        # Create logs directory if it doesn't exist
        execution_dir.mkdir(parents=True, exist_ok=True)

        # Store execution directory in class variable
        cls._execution_dir = execution_dir

    @classmethod
    def get_execution_dir(cls) -> Optional[Path]:
        """Get the current execution directory for this process."""
        return cls._execution_dir

    @classmethod
    def set_global_logger(cls, logger: "Logger") -> None:
        """Set the global logger instance for centralized access."""
        cls._global_logger = logger

    @classmethod
    def get_global_logger(cls) -> "Logger":
        """Get the global logger instance."""
        if cls._global_logger is None:
            raise RuntimeError("Global logger not initialized. Call Logger.set_global_logger() first.")
        return cls._global_logger

    def __init__(self, indent_level: int = 0, log_file: Optional[Union[Path, str]] = None):
        self.indent_level = indent_level

        # If log_file is a string, convert it to a Path relative to execution_dir
        if isinstance(log_file, str):
            exec_dir = self.get_execution_dir()
            if exec_dir:
                self.log_file = exec_dir / log_file
            else:
                # If no execution directory is set, use a default location
                default_logs_dir = Path(__file__).parent.parent / ".logs" / "default"
                default_logs_dir.mkdir(parents=True, exist_ok=True)
                self.log_file = default_logs_dir / log_file
        else:
            self.log_file = log_file

        # Create parent directory if needed
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _format(self, color: str, message: str) -> str:
        """Format message with color and indentation."""
        indent = "  " * self.indent_level
        return f"{indent}{color}{message}{self.COLORS['RESET']}"

    def _log_to_file(self, message: str):
        """Log message to file without colors."""
        if self.log_file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            indent = "  " * self.indent_level
            log_entry = f"[{timestamp}] {indent}{message}\n"

            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(log_entry)
            except Exception as e:
                # Fallback to stderr if file writing fails
                print(f"Failed to write to log file: {e}", file=sys.stderr)

    def _log(self, color: str, emoji: str, message: str):
        """Log message to both console and file."""
        # Format with emoji and text both indented
        indent = "  " * self.indent_level
        formatted_message = f"{indent}{emoji}{message}"

        # Print to console with colors
        print(f"{color}{formatted_message}{self.COLORS['RESET']}")

        # Write to file without colors
        self._log_to_file(formatted_message)

    def _log_no_emoji(self, color: str, message: str):
        """Log message to both console and file without emoji."""
        # Print to console with colors
        print(self._format(color, message))

        # Write to file without colors
        self._log_to_file(message)

    def info(self, message: str):
        """Log regular info message in blue."""
        self._log_no_emoji(self.COLORS["BLUE"], message)

    def success(self, message: str):
        """Log success message in green."""
        self._log_no_emoji(self.COLORS["GREEN"], message)

    def warning(self, message: str):
        """Log warning message in orange."""
        self._log_no_emoji(self.COLORS["ORANGE"], message)

    def error(self, message: str):
        """Log error message in red."""
        self._log_no_emoji(self.COLORS["RED"], message)

    def metrics(self, message: str):
        """Log metrics in bright purple to make them stand out."""
        self._log_no_emoji(self.COLORS["PURPLE"], message)

    def highlight(self, message: str):
        """Log highlighted information in yellow."""
        self._log_no_emoji(self.COLORS["YELLOW"], message)

    def debug(self, message: str):
        """Log debug information in cyan."""
        self._log_no_emoji(self.COLORS["CYAN"], message)

    def verbose(self, message: str):
        """Log verbose/debug information in gray."""
        self._log_no_emoji(self.COLORS["GRAY"], message)

    def indent(self):
        """Create a new logger with increased indentation."""
        return Logger(self.indent_level + 1, self.log_file)


class ExperimentMetadata:
    """Handles saving and loading experiment metadata for reproducibility."""

    def __init__(self, execution_dir: Path):
        self.execution_dir = execution_dir
        execution_dir.mkdir(parents=True, exist_ok=True)

    def save_config(self, config) -> bool:
        """Save configuration to a JSON file for experiment reproducibility."""
        try:
            from dataclasses import asdict

            # Convert dataclass to dict, handling sets by converting to lists
            config_dict = asdict(config)

            # Convert sets to lists for JSON serialization
            if "exclude_dirs" in config_dict and isinstance(config_dict["exclude_dirs"], set):
                config_dict["exclude_dirs"] = list(config_dict["exclude_dirs"])

            # Add metadata
            config_dict["_metadata"] = {
                "saved_at": datetime.now().isoformat(),
                "description": "Configuration for code localization experiment reproducibility",
                "model_name": config_dict.get("model_name", "unknown"),
            }

            config_file = self.execution_dir / "config.json"
            with open(config_file, "w") as f:
                json.dump(config_dict, f, indent=2)

            print(f"Configuration saved to: {config_file}")
            return True
        except Exception as e:
            print(f"Failed to save configuration: {e}")
            return False

    def save_environment_metadata(self, config=None) -> bool:
        """Save environment metadata for experiment reproducibility."""
        try:
            import platform
            import sys

            metadata = {
                "timestamp": datetime.now().isoformat(),
                "system_info": {
                    "platform": platform.platform(),
                    "python_version": sys.version,
                    "architecture": platform.architecture(),
                    "processor": platform.processor(),
                },
                "environment": {
                    "python_executable": sys.executable,
                    "working_directory": str(Path.cwd()),
                    "script_path": str(Path(__file__).resolve()),
                },
                "model_info": {
                    "model_name": (getattr(config, "model_name", "unknown") if config else "unknown"),
                    "tokenizer_name": (getattr(config, "tokenizer_model_name", "unknown") if config else "unknown"),
                },
            }

            # Try to get git information if available
            try:
                from git import Repo

                git_repo = Repo(Path(__file__).parent.parent, search_parent_directories=True)
                metadata["git_info"] = {
                    "commit_hash": git_repo.head.commit.hexsha,
                    "branch": git_repo.active_branch.name,
                    "is_dirty": git_repo.is_dirty(),
                    "remote_url": (next(git_repo.remotes.origin.urls) if git_repo.remotes else None),
                }
            except Exception as e:
                metadata["git_info"] = {"error": str(e)}

            # Try to get package versions for key dependencies
            try:
                import datasets
                import torch
                import transformers

                metadata["package_versions"] = {
                    "transformers": transformers.__version__,
                    "datasets": datasets.__version__,
                    "torch": torch.__version__,
                }
            except Exception as e:
                metadata["package_versions"] = {"error": str(e)}

            # Save metadata
            metadata_file = self.execution_dir / "environment_metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"Environment metadata saved to: {metadata_file}")
            return True

        except Exception as e:
            print(f"Warning: Failed to save environment metadata: {e}")
            return False

    def save_prompt_template(self, prompt_template_path: str) -> bool:
        """Save prompt template for experiment reproducibility."""
        try:
            # Save the prompt template file if it exists
            template_path = Path(prompt_template_path)
            if template_path.exists():
                # Copy the template file to the execution directory
                template_copy_path = self.execution_dir / "prompt_template.yaml"
                shutil.copy2(template_path, template_copy_path)
                print(f"Prompt template saved to: {template_copy_path}")

                # Also save template metadata
                template_metadata = {
                    "original_path": str(template_path.resolve()),
                    "copied_to": str(template_copy_path),
                    "file_size": template_path.stat().st_size,
                    "last_modified": datetime.fromtimestamp(template_path.stat().st_mtime).isoformat(),
                }

                template_metadata_file = self.execution_dir / "prompt_template_metadata.json"
                with open(template_metadata_file, "w") as f:
                    json.dump(template_metadata, f, indent=2)
                print(f"Prompt template metadata saved to: {template_metadata_file}")
                return True
            else:
                print(f"Warning: Prompt template file not found: {template_path}")
                return False

        except Exception as e:
            print(f"Warning: Failed to save prompt template: {e}")
            return False

    def create_experiment_readme(self, config) -> bool:
        """Create a README file explaining how to reproduce this experiment."""
        try:
            readme_content = f"""# Code Localization Experiment

        This directory contains the results and configuration for a code localization experiment run on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.

        ## Files in this directory:

        - `config.json` - Complete configuration used for this experiment
        - `environment_metadata.json` - System and environment information
        - `prompt_template.yaml` - Copy of the prompt template used
        - `prompt_template_metadata.json` - Metadata about the prompt template
        - `code_localizer.log` - Main execution log
        - `progress.json` - Progress tracking (updated during execution)
        - `aggregated_metrics.json` - Final aggregated results (created after completion)

        ## How to reproduce this experiment:

        ### Method 1: Using the saved configuration
        ```python
        from pathlib import Path
        from generate import GenerateCodeLocalizer

        # Load the exact configuration from this experiment
        experiment_dir = Path("path/to/this/experiment/directory")
        code_localizer = GenerateCodeLocalizer.from_experiment_dir(experiment_dir)
        code_localizer.run()
        ```

        ### Method 2: Manual configuration
        ```python
        from generate import GenerateCodeLocalizer, Config

        # Create config with the same parameters
        config = Config.load_from_file("config.json")
        code_localizer = GenerateCodeLocalizer(config)
        code_localizer.run()
        ```

        ## Key Configuration Parameters:

        - **Model**: {config.model_name or "unknown"}
        - **Tokenizer**: {config.tokenizer_model_name}
        - **Dataset**: {config.dataset_name}
        - **Max Context Length**: {config.max_context_length}
        - **Max Turns**: {config.max_num_turns}
        - **Remote Model**: {config.use_remote_model}

        ## Environment Information:

        - Python version and system details are stored in `environment_metadata.json`
        - Package versions for key dependencies are included
        - Git commit information (if available) is preserved

        ## Notes:

        - The experiment will create a new execution directory with its own timestamp
        - Progress is automatically saved and can be resumed if interrupted
        - All conversations and results are preserved for analysis
        """

            readme_file = self.execution_dir / "README.md"
            with open(readme_file, "w") as f:
                f.write(readme_content)

            print(f"Experiment README created: {readme_file}")
            return True

        except Exception as e:
            print(f"Warning: Failed to create experiment README: {e}")
            return False

    def save_all_metadata(self, config, prompt_template_path: str) -> bool:
        """Save all experiment metadata for reproducibility."""
        success = True

        # Save configuration
        if not self.save_config(config):
            success = False

        # Save environment metadata with model info
        if not self.save_environment_metadata(config):
            success = False

        # Save prompt template
        if not self.save_prompt_template(prompt_template_path):
            success = False

        # Create README
        if not self.create_experiment_readme(config):
            success = False

        return success


class ConversationTracker:
    """Tracks conversation history and saves to JSON."""

    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.conversation_data = {
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "dialog": [],
            "metadata": {
                "total_turns": 0,
                "tool_calls": 0,
                "locations_found": 0,
                "status": "in_progress",  # in_progress, success, failed
                "failure_reason": None,  # If status is failed, why it failed
            },
            "ground_truth_locations": [],  # Store ground truth locations
            "predicted_locations": [],  # Store predicted locations
        }

    def add_user_message(self, message: str, turn_number: int):
        """Add a user message to the conversation."""
        self.conversation_data["dialog"].append(
            {
                "turn": turn_number,
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self.conversation_data["metadata"]["total_turns"] = turn_number

    def add_assistant_message(self, message: str, turn_number: int):
        """Add an assistant message to the conversation."""
        self.conversation_data["dialog"].append(
            {
                "turn": turn_number,
                "role": "assistant",
                "content": message,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def add_tool_call(self, tool_name: str, arguments: Dict, result: str, turn_number: int):
        """Add a tool call to the conversation."""
        # Make sure arguments and result are serializable
        serializable_arguments = self._make_serializable(arguments)
        serializable_result = self._make_serializable(result)

        self.conversation_data["dialog"].append(
            {
                "turn": turn_number,
                "role": "tool_call",
                "tool": tool_name,
                "arguments": serializable_arguments,
                "result": serializable_result,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self.conversation_data["metadata"]["tool_calls"] += 1

    def update_locations_found(self, count: int):
        """Update the number of locations found."""
        self.conversation_data["metadata"]["locations_found"] = count
        # If we found any locations, mark as success
        if count > 0:
            self.conversation_data["metadata"]["status"] = "success"

    def update_total_turns(self, turn_number: int):
        """Update the total number of turns."""
        self.conversation_data["metadata"]["total_turns"] = turn_number

    def add_ground_truth_locations(self, locations: List[str]):
        """Add ground truth locations to the conversation."""
        self.conversation_data["ground_truth_locations"] = locations

    def add_predicted_locations(self, locations: List[str]):
        """Add predicted locations to the conversation."""
        self.conversation_data["predicted_locations"] = locations
        # If we got any predictions, mark as success
        if locations:
            self.conversation_data["metadata"]["status"] = "success"
        else:
            self.conversation_data["metadata"]["status"] = "failed"
            self.conversation_data["metadata"]["failure_reason"] = "No locations predicted"

    def mark_failed(self, reason: str):
        """Mark the conversation as failed with a reason."""
        self.conversation_data["metadata"]["status"] = "failed"
        self.conversation_data["metadata"]["failure_reason"] = reason

    def add_evaluation_metrics(self, metrics: Dict):
        """Add evaluation metrics to the conversation."""
        # Convert dataclass objects to dictionaries for JSON serialization
        serializable_metrics = self._make_serializable(metrics)
        self.conversation_data["evaluation"] = serializable_metrics
        # Update locations_found from total_predicted if not already set
        if "total_predicted" in metrics and self.conversation_data["metadata"]["locations_found"] == 0:
            self.conversation_data["metadata"]["locations_found"] = metrics.get("total_predicted", 0)

    def add_raw_outputs(self, raw_outputs: List[Dict]):
        """Add raw model outputs for debugging and investigation."""
        # Convert raw outputs to serializable format
        serializable_outputs = self._make_serializable(raw_outputs)
        self.conversation_data["raw_outputs"] = serializable_outputs

    def _make_serializable(self, obj):
        """Convert dataclass objects to dictionaries for JSON serialization."""
        # Handle Path objects specifically
        if hasattr(obj, "__class__") and obj.__class__.__name__ in [
            "PosixPath",
            "WindowsPath",
            "Path",
        ]:
            return str(obj)

        if hasattr(obj, "__dict__"):
            # Handle dataclass objects
            if hasattr(obj, "__dataclass_fields__"):
                return {k: self._make_serializable(v) for k, v in obj.__dict__.items()}
            # Handle regular objects
            return {k: self._make_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # For any other types, convert to string
            return str(obj)

    def save_conversation(self, task_dir: Path):
        """Save conversation to JSON file."""
        task_dir.mkdir(parents=True, exist_ok=True)
        filename = "conversation.json"
        filepath = task_dir / filename

        try:
            # Make sure all data is JSON serializable
            serializable_data = self._make_serializable(self.conversation_data)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            return filepath
        except Exception as e:
            print(f"Failed to save conversation: {e}", file=sys.stderr)
            return None


def logger() -> Logger:
    """Get the global logger instance."""
    return Logger.get_global_logger()
