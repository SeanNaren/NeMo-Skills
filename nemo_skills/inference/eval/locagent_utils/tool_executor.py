import os
import logging
from nemo_skills.utils import get_logger_name
from nemo_skills.inference.eval.locagent_utils.utils import tree_repo_dict

LOG = logging.getLogger(get_logger_name(__file__))


class ToolExecutor:
    """Handles tool execution for the assistant against a nested dictionary representation of a repository."""

    def __init__(self):
        pass

    def execute_tool(self, extracted_block: dict, repo_dict: dict) -> str:
        """Execute a tool call and return the result."""
        tool_name = extracted_block.get("tool", "")
        LOG.info(f"Executing tool: {tool_name}")

        if tool_name == "view":
            return self._execute_view_tool(extracted_block, repo_dict)
        elif tool_name == "repo_tree":
            return self._execute_repo_tree_tool(repo_dict)
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
                return f"Error: File not found at path '{file_path}'"

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

            result_lines = [f"{i+1:4d}: {lines[i]}" for i in range(start_line - 1, end_line)]
            LOG.info(f"Successfully read {len(result_lines)} lines from {file_path}")
            file_content = f"File: {file_path} (lines {start_line}-{end_line})\n" + "\n".join(result_lines)
            return file_content

        except Exception as e:
            LOG.error(f"Error executing view tool: {str(e)}")
            return f"Error executing view tool: {str(e)}"

    def _execute_repo_tree_tool(self, repo_dict: dict) -> str:
        """Generates a tree view of the repository from the nested dictionary."""
        try:
            return tree_repo_dict(repo_dict)
        except Exception as e:
            LOG.error(f"Error executing repo_tree tool: {str(e)}")
            return f"Error executing repo_tree tool: {str(e)}"

    def _execute_codebase_search_tool(self, extracted_block: dict, repo_dict: dict) -> str:
        """Executes a codebase search on the nested dictionary, showing context around matches."""
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
                        # Find all lines containing the query (using 0-based indexing)
                        match_indices = [i for i, line in enumerate(lines) if query_lower in line.lower()]

                        # If no matches were found in this file, skip it
                        if not match_indices:
                            continue

                        file_result_parts = [f"File: {new_path}"]
                        total_matches_in_file = len(match_indices)

                        # Use a set to track lines already shown in a snippet to avoid overlaps
                        # for matches that are close to each other.
                        shown_indices = set()

                        MAX_SNIPPETS_PER_FILE = 3  # Limit the number of context blocks per file
                        snippets_created = 0

                        for match_idx in match_indices:
                            if snippets_created >= MAX_SNIPPETS_PER_FILE:
                                break

                            # If this match was already included in a previous snippet's context, skip it.
                            if match_idx in shown_indices:
                                continue

                            snippets_created += 1

                            # Define the context window: 20 lines before, the match, 20 lines after
                            start_idx = max(0, match_idx - 20)
                            end_idx = min(len(lines), match_idx + 21)

                            file_result_parts.append(f"\n--- Snippet {snippets_created} (match on line {match_idx + 1}) ---")

                            for i in range(start_idx, end_idx):
                                line_num = i + 1
                                line_content = lines[i].rstrip()

                                # Highlight the specific matching line with a '>'
                                if i == match_idx:
                                    prefix = f"> {line_num:4d}"
                                else:
                                    prefix = f"  {line_num:4d}"

                                file_result_parts.append(f"{prefix}: {line_content}")
                                shown_indices.add(i)

                        # Add a summary if some matches were not shown in detail
                        if total_matches_in_file > snippets_created:
                            remaining_matches = total_matches_in_file - snippets_created
                            file_result_parts.append(f"\n... and {remaining_matches} more match(es) in this file.")

                        # The -total_matches_in_file is used to sort results by relevance (most matches first)
                        results.append((-total_matches_in_file, "\n".join(file_result_parts)))

            search_recursive(repo_dict, "")

            # Sort results by match count (descending), then by file path (ascending)
            results.sort()
            results = results[:5]

            # Extract just the formatted string part for the final output
            final_results = [res[1] for res in results]

            if not final_results:
                return f"No results found for query: '{query}'"

            # Join the results from different files with a clear separator
            return "\n\n==================================================\n\n".join(final_results)

        except Exception as e:
            LOG.error(f"Error executing codebase_search tool: {str(e)}")
            return f"Error executing codebase_search tool: {str(e)}"
