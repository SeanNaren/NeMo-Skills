import os
from config import Config

import logging
from .repository_manager import RepositoryManager

LOG = logging.getLogger(get_logger_name(__file__))


class ToolExecutor:
    """Handles tool execution for the assistant."""

    def __init__(self, config: Config = None):
        self.repo_manager = repo_manager
        self.config = config

    def execute_tool(self, extracted_block: dict, repo_dir: str) -> str:
        """Execute a tool call and return the result."""
        LOG.info(f"Executing tool: {extracted_block.get('tool', '')}")

        if extracted_block.get("tool") == "view":
            return self._execute_view_tool(extracted_block, repo_dir)
        elif extracted_block.get("tool") == "repo_tree":
            return self._execute_repo_tree_tool(repo_dir)
        elif extracted_block.get("tool") == "codebase_search":
            return self._execute_codebase_search_tool(extracted_block, repo_dir)
        else:
            LOG.error(f"Unknown tool: {extracted_block.get('tool', '')}")
            return f"Error: Unknown tool '{extracted_block.get('tool', '')}'"

    def _execute_view_tool(self, extracted_block: dict, repo_dir: str) -> str:
        """Execute the view tool to show file contents."""
        try:
            file_path = extracted_block.get("path", "")
            view_range = extracted_block.get("view_range")

            LOG.info(f"Viewing file: {file_path}")
            if view_range:
                LOG.info(f"Range: lines {view_range[0]}-{view_range[1]}")

            # Resolve file path
            if os.path.isabs(file_path) or os.path.commonpath([os.path.abspath(file_path), os.path.abspath(repo_dir)]) == os.path.abspath(repo_dir):
                full_path = file_path
            else:
                full_path = os.path.join(repo_dir, file_path)

            with open(full_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Handle view range according to tool instructions:
            # - omit view_range to show the whole file
            # - [start, end] → show inclusive lines start-end
            # - [start, -1] → show from start to EOF
            if view_range is None:
                # Show whole file
                start_line, end_line = 1, len(lines)
                LOG.info(f"Showing entire file ({len(lines)} lines)")
            else:
                # Validate view_range format
                if not isinstance(view_range, list) or len(view_range) != 2:
                    LOG.error(f"Invalid view_range format: {view_range}. Expected [start, end] or [start, -1]")
                    return f"Error: Invalid view_range format. Expected [start, end] or [start, -1], got {view_range}"

                start_line, end_line = view_range[0], view_range[1]

                # Handle [start, -1] case (show from start to EOF)
                if end_line == -1:
                    end_line = len(lines)
                    LOG.info(f"Showing from line {start_line} to end of file ({len(lines)} lines)")
                else:
                    LOG.info(f"Showing lines {start_line}-{end_line}")

            # Validate line numbers
            if start_line < 1:
                LOG.info(f"Start line {start_line} is less than 1, adjusting to 1")
                start_line = 1

            if end_line > len(lines):
                LOG.info(f"End line {end_line} is greater than file length ({len(lines)}), adjusting to {len(lines)}")
                end_line = len(lines)

            if start_line > end_line:
                LOG.error(f"Start line {start_line} is greater than end line {end_line}")
                return f"Error: Start line {start_line} is greater than end line {end_line}"

            # Extract and format the requested lines
            result_lines = [f"{i+1:4d}: {lines[i].rstrip()}" for i in range(start_line - 1, end_line)]

            LOG.success(f"Successfully read {len(result_lines)} lines from {file_path}")

            # Create the full file content
            file_content = f"File: {file_path} (lines {start_line}-{end_line})\n" + "\n".join(result_lines)

            return file_content

        except Exception as e:
            LOG.error(f"Error executing view tool: {str(e)}")
            return f"Error executing view tool: {str(e)}"

    def _execute_repo_tree_tool(self, repo_dir: str) -> str:
        result = []
        repo_name = os.path.basename(repo_dir)
        result.append(f"{repo_name}/")

        # Get all allowed files with their line counts first (only if show_line_counts is enabled)
        file_line_counts = {}
        if getattr(self.config, "show_line_counts", True):
            for root, dirs, files in os.walk(repo_dir):
                for file in files:
                    # Check if file has any of the allowed extensions
                    file_ext = os.path.splitext(file)[1].lstrip(".")
                    if file_ext in self.config.file_extensions or file in self.config.file_extensions:
                        full_path = os.path.join(root, file)
                        try:
                            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                                line_count = sum(1 for _ in f)
                            file_line_counts[full_path] = line_count
                        except Exception:
                            file_line_counts[full_path] = 0

        def add_to_tree(path, prefix="", rel_path=""):
            try:
                # Get all items and sort them (directories first, then files)
                items = []
                for item in os.listdir(path):
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        items.append((item, True))  # (name, is_dir)
                    else:
                        # Check if file has any of the allowed extensions
                        file_ext = os.path.splitext(item)[1].lstrip(".")
                        if file_ext in self.config.file_extensions or item in self.config.file_extensions:
                            items.append((item, False))  # (name, is_file)

                # Sort: directories first, then files, both alphabetically
                items.sort(key=lambda x: (not x[1], x[0].lower()))

                for i, (item, is_dir) in enumerate(items):
                    # Skip unwanted directories and files
                    if item.startswith(".") or item in ["__pycache__", ".git"]:
                        continue

                    # Skip excluded directories
                    exclude_dirs = getattr(self.config, "exclude_dirs", set())
                    if is_dir and item.lower() in exclude_dirs:
                        continue

                    item_path = os.path.join(path, item)
                    is_last = i == len(items) - 1

                    if is_dir:
                        # Directory
                        result.append(f"{prefix}{'└── ' if is_last else '├── '}{item}/")
                        new_prefix = prefix + ("    " if is_last else "│   ")
                        new_rel_path = os.path.join(rel_path, item) if rel_path else item
                        add_to_tree(item_path, new_prefix, new_rel_path)
                    else:
                        # Allowed file
                        full_path = os.path.join(repo_dir, rel_path, item) if rel_path else os.path.join(repo_dir, item)
                        if getattr(self.config, "show_line_counts", True):
                            line_count = file_line_counts.get(full_path, 0)
                            result.append(f"{prefix}{'└── ' if is_last else '├── '}{item} ({line_count} lines)")
                        else:
                            result.append(f"{prefix}{'└── ' if is_last else '├── '}{item}")

            except Exception:
                pass

        add_to_tree(repo_dir)
        return "\n".join(result)

    def _execute_codebase_search_tool(self, extracted_block: dict, repo_dir: str) -> str:
        """Execute the codebase_search tool to search for code."""
        query = extracted_block.get("query", "")
        if not query:
            return "Error: No search query provided for codebase_search tool."

        LOG.info(f"Searching codebase for: {query}")

        try:
            import re
            from pathlib import Path

            results = []
            query_lower = query.lower()

            # Define file extensions to search (focus on code files)
            code_extensions = self.config.code_extensions if self.config and self.config.code_extensions else {'.py'}

            # Use exclude_dirs from config (DEFAULT_EXCLUDE_DIRS)
            exclude_dirs = self.config.exclude_dirs if self.config and self.config.exclude_dirs else set()

            repo_path = Path(repo_dir)

            for file_path in repo_path.rglob('*'):
                # Skip directories
                if file_path.is_dir():
                    continue

                # Skip excluded directories
                rel_path = file_path.relative_to(repo_path)
                if any(exclude_dir in rel_path.parts for exclude_dir in exclude_dirs):
                    continue

                # Skip files without code extensions
                if file_path.suffix not in code_extensions:
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    # Search for query in content
                    if query_lower in content.lower():
                        # Get relative path from repo root
                        rel_path = file_path.relative_to(repo_path)

                        # Find matching lines
                        lines = content.split('\n')
                        matching_lines = []

                        for i, line in enumerate(lines, 1):
                            if query_lower in line.lower():
                                # Truncate long lines for readability
                                line_preview = line.strip()[:100]
                                if len(line.strip()) > 100:
                                    line_preview += "..."
                                matching_lines.append(f"L{i}: {line_preview}")

                        # Create result entry
                        result = f"File: {rel_path}\n"
                        if matching_lines:
                            result += f"Matches found: {len(matching_lines)}\n"
                            result += "\n".join(matching_lines[:10])  # Limit to 10 lines
                            if len(matching_lines) > 10:
                                result += f"\n... and {len(matching_lines) - 10} more matches"
                        else:
                            result += "Query found in file content"

                        results.append(result)

                except Exception as e:
                    # Skip files that can't be read
                    continue

            # Sort results by relevance (files with more matches first)
            def sort_key(result):
                # Count the number of "L" lines (matching lines)
                match_count = result.count("L")
                return -match_count  # Negative for descending order

            results.sort(key=sort_key)

            return results

        except Exception as e:
            LOG.error(f"Error executing codebase_search tool: {str(e)}")
            return []