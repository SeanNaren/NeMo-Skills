import os
import logging
from nemo_skills.utils import get_logger_name
from nemo_skills.inference.eval.locagent_utils.utils import tree_repo_dict, connected_tree_repo_dict

LOG = logging.getLogger(get_logger_name(__file__))


class ToolExecutor:
    """Handles tool execution for the assistant against a nested dictionary representation of a repository."""

    def __init__(self, cfg=None):
        self.cfg = cfg

    def execute_tool(self, extracted_block: dict, repo_dict: dict) -> str:
        """Execute a tool call and return the result."""
        tool_name = extracted_block.get("tool", "")
        
        # Backward compatibility: infer tool from other fields if "tool" field is missing
        if not tool_name:
            if "path" in extracted_block:
                tool_name = "view_file"
            elif "query" in extracted_block:
                tool_name = "codebase_search"
            elif "file" in extracted_block:
                tool_name = "connected_tree"
            else:
                LOG.error(f"Cannot determine tool from extracted_block: {extracted_block}")
                return f"Error: Cannot determine which tool to execute. Please specify 'tool' field in your JSON."
        
        LOG.info(f"Executing tool: {tool_name}")

        if tool_name == "view" or tool_name == "view_file":
            return self._execute_view_tool(extracted_block, repo_dict)
        elif tool_name == "repo_tree":
            return self._execute_repo_tree_tool(repo_dict)
        elif tool_name == "connected_tree":
            return self._execute_connected_tree_tool(extracted_block, repo_dict)
        elif tool_name == "codebase_search":
            return self._execute_codebase_search_tool(extracted_block, repo_dict)
        else:
            LOG.error(f"Unknown tool: {tool_name}")
            return f"Error: Unknown tool '{tool_name}'"

    def _get_node_from_path(self, repo_dict: dict, path: str):
        """Helper function to navigate the nested dict using a file path."""
        # Normalize path to handle both empty and non-empty paths
        parts = [part for part in path.split('/') if part]
        current_level = repo_dict["structure"]
        try:
            for part in parts:
                current_level = current_level[part]
            return current_level
        except (KeyError, TypeError):
            return None

    def _is_file_node(self, node: dict) -> bool:
        """Checks if a node in the dictionary represents a file."""
        return isinstance(node, dict) and "text" in node and isinstance(node["text"], list)

    def _is_dir_node(self, node: dict) -> bool:
        """Checks if a node in the dictionary represents a directory."""
        return isinstance(node, dict) and "text" not in node

    def _find_similar_files(self, repo_dict: dict, target_filename: str, max_suggestions: int = 3) -> list:
        """Find files with similar names to the target filename."""
        similar_files = []
        target_name = target_filename.lower()
        
        def search_recursive(d, current_path=""):
            for key, value in d.items():
                current_file_path = f"{current_path}/{key}" if current_path else key
                
                if self._is_file_node(value):
                    # Check if filename matches (case-insensitive)
                    if key.lower() == target_name:
                        similar_files.append(current_file_path)
                    # Check if filename contains the target or target contains filename
                    elif target_name in key.lower() or key.lower() in target_name:
                        similar_files.append(current_file_path)
                elif self._is_dir_node(value):
                    search_recursive(value, current_file_path)
        
        search_recursive(repo_dict["structure"])
        
        # Sort by similarity (exact matches first, then by length difference)
        def similarity_score(filepath):
            filename = filepath.split('/')[-1].lower()
            if filename == target_name:
                return 0  # Exact match gets highest priority
            elif target_name in filename:
                return 1  # Target contained in filename
            elif filename in target_name:
                return 2  # Filename contained in target
            else:
                return 3  # Other matches
        
        similar_files.sort(key=similarity_score)
        return similar_files[:max_suggestions]

    def _execute_view_tool(self, extracted_block: dict, repo_dict: dict) -> str:
        """Execute the view tool to show file contents from the dictionary."""
        try:
            file_path = extracted_block.get("path", "")
            if not file_path:
                return "Error: No file path provided for the 'view' tool."
            view_range = extracted_block.get("view_range")

            LOG.info(f"Viewing file: {file_path}")
            if view_range:
                LOG.info(f"Range: lines {view_range[0]}-{view_range[1]}")

            file_node = self._get_node_from_path(repo_dict, file_path)

            if not self._is_file_node(file_node):
                LOG.error(f"File not found or is a directory: {file_path}")
                
                # Extract just the filename for searching similar files
                filename = file_path.split('/')[-1] if '/' in file_path else file_path
                similar_files = self._find_similar_files(repo_dict, filename)
                
                error_msg = f"Error: File not found at path '{file_path}'"
                
                if similar_files:
                    error_msg += f"\n\n💡 Did you mean one of these files?"
                    for i, similar_file in enumerate(similar_files, 1):
                        error_msg += f"\n   {i}. {similar_file}"
                    error_msg += f"\n\nTry using the exact path from the suggestions above."
                else:
                    error_msg += f"\n\n💡 No similar files found with name '{filename}'. Use the repo_tree tool to browse available files."
                
                return error_msg

            lines = file_node["text"]  # The lines are already clean, without trailing '\n'

            if view_range is None:
                start_line, end_line = 1, len(lines)
                LOG.info(f"Showing entire file ({len(lines)} lines)")
            else:
                if not isinstance(view_range, list) or len(view_range) != 2:
                    LOG.error(f"Invalid view_range format: {view_range}. Expected [start, end] or [start, -1]")
                    return f"Error: Invalid view_range format. Expected [start, end] or [start, -1], got {view_range}"
                start_line, end_line = view_range[0], view_range[1]
                if end_line == -1:
                    end_line = len(lines)
                    LOG.info(f"Showing from line {start_line} to end of file ({len(lines)} lines)")
                else:
                    LOG.info(f"Showing lines {start_line}-{end_line}")

            if start_line < 1:
                start_line = 1
            if end_line > len(lines):
                end_line = len(lines)
            if start_line > end_line:
                LOG.error(f"Start line {start_line} is greater than end line {end_line}")
                return f"Error: Start line {start_line} is greater than end line {end_line}"

            # Check if file content needs truncation
            max_lines = self.cfg.max_view_lines if self.cfg and hasattr(self.cfg, 'max_view_lines') else 1000
            total_lines_to_show = end_line - start_line + 1
            truncated = False
            truncation_message = ""
            
            if max_lines > 0 and total_lines_to_show > max_lines:
                # Truncate to max_lines, but keep the requested start_line
                original_end_line = end_line
                end_line = start_line + max_lines - 1
                truncated = True
                truncation_message = f"\nWARNING: File content truncated. Showing {max_lines} lines out of {total_lines_to_show} requested lines (original range: {start_line}-{original_end_line}). Use smaller ranges to view specific sections.\n"
                LOG.warning(f"File {file_path} truncated from {total_lines_to_show} to {max_lines} lines")

            result_lines = [f"{i+1:4d}: {lines[i]}" for i in range(start_line - 1, end_line)]
            LOG.info(f"Successfully read {len(result_lines)} lines from {file_path}")
            
            file_header = f"File: {file_path} (lines {start_line}-{end_line})"
            if truncated:
                file_header += f" [TRUNCATED - Original file has {len(lines)} total lines]"
            
            file_content = file_header + truncation_message + "\n" + "\n".join(result_lines)
            return file_content

        except Exception as e:
            LOG.error(f"Error executing view tool: {str(e)}")
            return f"Error executing view tool: {str(e)}"

    def _execute_repo_tree_tool(self, repo_dict: dict) -> str:
        """Generates a tree view of the repository from the nested dictionary."""
        try:
            show_line_counts = self.cfg.show_line_counts if self.cfg else True
            return tree_repo_dict(repo_dict, show_line_counts)
        except Exception as e:
            LOG.error(f"Error executing repo_tree tool: {str(e)}")
            return f"Error executing repo_tree tool: {str(e)}"

    def _execute_connected_tree_tool(self, extracted_block: dict, repo_dict: dict) -> str:
        """Generates a connected tree view showing import dependencies."""
        try:
            file_path = extracted_block.get("file", None)
            show_line_counts = self.cfg.show_line_counts if self.cfg else True
            
            LOG.info(f"Generating connected tree for file: {file_path or 'entire repository'}")
            return connected_tree_repo_dict(repo_dict, file_path, show_line_counts)
        except Exception as e:
            LOG.error(f"Error executing connected_tree tool: {str(e)}")
            return f"Error executing connected_tree tool: {str(e)}"

    def _execute_codebase_search_tool(self, extracted_block: dict, repo_dict: dict) -> str:
        """Executes a codebase search on the nested dictionary, returning occurrence statistics."""
        query = extracted_block.get("query", "")
        if not query:
            return "Error: No search query provided for codebase_search tool."

        LOG.info(f"Searching codebase for: {query}")

        try:
            results = []
            query_lower = query.lower()

            def search_recursive(current_dict: dict, current_path: str):
                for name, node in current_dict.items():
                    new_path = f"{current_path}/{name}" if current_path else name

                    if self._is_dir_node(node):
                        search_recursive(node, new_path)

                    elif self._is_file_node(node):
                        lines = node["text"]
                        # Count all occurrences of the query in this file
                        match_count = sum(1 for line in lines if query_lower in line.lower())

                        # If no matches were found in this file, skip it
                        if match_count == 0:
                            continue

                        # Store the result as a tuple (negative count for sorting, path, count)
                        results.append((-match_count, new_path, match_count))

            search_recursive(repo_dict, "")

            # Sort results by match count (descending), then by file path (ascending)
            results.sort()

            if not results:
                return f"No results found for query: '{query}'"

            # Format the results as a list of file paths with occurrence counts
            output_lines = [f"Search results for query: '{query}'", ""]
            output_lines.append(f"{'Occurrences':<12} File Path")
            output_lines.append("-" * 60)
            
            for _, file_path, count in results:
                output_lines.append(f"{count:<12} {file_path}")
            
            output_lines.append("")
            output_lines.append(f"Total files with matches: {len(results)}")
            total_occurrences = sum(count for _, _, count in results)
            output_lines.append(f"Total occurrences: {total_occurrences}")

            return "\n".join(output_lines)

        except Exception as e:
            LOG.error(f"Error executing codebase_search tool: {str(e)}")
            return f"Error executing codebase_search tool: {str(e)}"
