import json
import re
from typing import Dict
import logging
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


class DialogProcessor:
    """Processes dialog output to extract tool calls and locations."""

    @staticmethod
    def extract_response(text: str, remove_thinking: bool):
        if remove_thinking and "</think>" in text:
            dialog_text = text.split("</think>")[1].lstrip().rstrip()
        else:
            dialog_text = text

        if "###Tool" in dialog_text:
            LOG.info("Found ###Tool block in output")
            return DialogProcessor._extract_tool_calls(dialog_text)
        elif "###Locations" in dialog_text:
            LOG.info("Found ###Locations block in output")
            return DialogProcessor._extract_locations(dialog_text)
        else:
            LOG.warning("No ###Tool or ###Locations found, checking for implicit tool calls")
            return DialogProcessor._extract_implicit_tool_calls(dialog_text)

    # USED
    @staticmethod
    def _extract_tool_calls(dialog_text: str) -> Dict:
        """Extract tool calls from dialog text."""
        tool_match = re.search(r"###Tool\s*\n(.*?)(?=\n###|$)", dialog_text, re.DOTALL)

        if tool_match:
            try:
                tool_data = json.loads(tool_match.group(1).strip())

                # Determine tool name
                if "path" in tool_data:
                    tool_data["tool"] = "view"
                elif "query" in tool_data:
                    tool_data["tool"] = "codebase_search"
                elif "repo_tree" in tool_data or len(tool_data) == 0:
                    tool_data["tool"] = "repo_tree"
                else:
                    tool_data["tool"] = "unknown"

                return {"type": "tool_calls", "tool_call": tool_data}
            except json.JSONDecodeError as e:
                LOG.warning(f"Failed to parse tool call JSON: {e}")
                return None
        return None

    # USED
    @staticmethod
    def _extract_locations(dialog_text: str) -> Dict:
        """Extract location predictions from dialog text."""
        locations = []
        locations_match = re.search(r"###Locations\s*\n(.*?)(?=\n###|$)", dialog_text, re.DOTALL)

        if locations_match:
            locations_text = locations_match.group(1).strip()

            for line in locations_text.split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    location_match = re.match(r"([^:]+):L(\d+)-L(\d+)", line)
                    if location_match:
                        file_path, start_line, end_line = location_match.groups()
                        locations.append(
                            {
                                "file_path": file_path,
                                "start_line": int(start_line),
                                "end_line": int(end_line),
                                "raw": line,
                            }
                        )
                    else:
                        locations.append({"raw": line})

        return {"type": "locations", "locations": locations}

    # USED
    @staticmethod
    def _extract_implicit_tool_calls(dialog_text: str) -> Dict:
        """Extract tool calls that appear as JSON without ###Tool wrapper."""
        # Check for JSON after </think> tag
        if "</think>" in dialog_text:
            # Split and clean up the text after </think>
            after_think = dialog_text.split("</think>")[1].strip()
            # Replace escaped newlines and multiple newlines with a single space
            after_think = re.sub(r"\\n|\n+", " ", after_think)

            # First try: Look for any JSON-like structure with a path field
            json_pattern = r"\{(?:[^{}]|\"[^\"]*\")*\"path\"(?:[^{}]|\"[^\"]*\")*\}"
            json_match = re.search(json_pattern, after_think)

            if json_match:
                try:
                    json_str = json_match.group(0)
                    # Clean up any remaining escapes or whitespace
                    json_str = json_str.replace("\\", "")
                    tool_data = json.loads(json_str)

                    # Determine tool type based on content
                    if "path" in tool_data:
                        tool_data["tool"] = "view"
                        # Add view_range if not present
                        if "view_range" not in tool_data:
                            tool_data["view_range"] = None
                    elif "query" in tool_data:
                        tool_data["tool"] = "codebase_search"
                        # Check if query is empty or whitespace
                        if not tool_data.get("query", "").strip():
                            LOG.warning(f"Empty query in codebase_search tool call")
                            return None
                    elif "repo_tree" in tool_data or len(tool_data) == 0:
                        tool_data["tool"] = "repo_tree"
                    else:
                        tool_data["tool"] = "unknown"
                    LOG.info(f"Found JSON tool call after </think>: {tool_data}")
                    return {"type": "tool_calls", "tool_call": tool_data}
                except json.JSONDecodeError as e:
                    LOG.warning(f"Failed to parse JSON after </think>: {e}")

            # Second try: Simple file path request
            # Look for quoted or unquoted file paths after </think>
            file_path_match = re.search(
                r'(?:\'|")?([^\'"\s]+?\.(?:py|cpp|h|hpp|java|js|ts|rb|go|rs|cs|php))(?:\'|")?',
                after_think,
            )
            if file_path_match:
                tool_data = {
                    "tool": "view",
                    "path": file_path_match.group(1),
                    "view_range": None,
                }
                LOG.info(f"Found simple file path request: {tool_data['path']}")
                return {"type": "tool_calls", "tool_call": tool_data}

            # Third try: Simple search query request
            # Look for quoted or unquoted search terms after </think>
            # Common patterns: "search for X", "find X", "look for X", "X function", "X class"
            search_patterns = [
                r'search\s+for\s+(?:\'|")?([A-Za-z_][A-Za-z0-9_]*)(?:\'|")?',
                r'find\s+(?:\'|")?([A-Za-z_][A-Za-z0-9_]*)(?:\'|")?',
                r'look\s+for\s+(?:\'|")?([A-Za-z_][A-Za-z0-9_]*)(?:\'|")?',
                r'(?:\'|")?([A-Za-z_][A-Za-z0-9_]*)\s+function(?:\'|")?',
                r'(?:\'|")?([A-Za-z_][A-Za-z0-9_]*)\s+class(?:\'|")?',
                r'(?:\'|")?([A-Za-z_][A-Za-z0-9_]*)\s+method(?:\'|")?',
                r'(?:\'|")?([A-Za-z_][A-Za-z0-9_]*)\s+variable(?:\'|")?',
            ]

            for pattern in search_patterns:
                search_match = re.search(pattern, after_think, re.IGNORECASE)
                if search_match:
                    search_term = search_match.group(1).strip()
                    # Clean up the search term
                    search_term = re.sub(r'[\'"]', "", search_term)
                    if search_term and len(search_term) > 1:  # Ensure it's not just whitespace
                        tool_data = {
                            "tool": "codebase_search",
                            "query": search_term,
                        }
                        LOG.info(f"Found simple search request: {tool_data['query']}")
                        return {"type": "tool_calls", "tool_call": tool_data}

        # Look for specific tool call patterns
        # View tool: {"path": "...", "view_range": [...]}
        view_pattern = r'\{[^{}]*"path"[^{}]*(?:"view_range"[^{}]*)?}'
        # Repo tree tool: {} or {"repo_tree": true} or similar
        repo_tree_pattern = r'\{[^{}]*"repo_tree"[^{}]*\}'
        # Codebase search tool: {"query": "..."}
        codebase_search_pattern = r'\{[^{}]*"query"[^{}]*\}'

        # Try view tool first
        json_match = re.search(view_pattern, dialog_text)
        if json_match:
            try:
                tool_data = json.loads(json_match.group())
                tool_data["tool"] = "view"
                LOG.warning(f"Found implicit view tool call: {tool_data}")
                return {"type": "tool_calls", "tool_call": tool_data}
            except json.JSONDecodeError as e:
                LOG.warning(f"Failed to parse implicit view tool call JSON: {e}")

        # Try codebase search tool
        json_match = re.search(codebase_search_pattern, dialog_text)
        if json_match:
            try:
                tool_data = json.loads(json_match.group())
                tool_data["tool"] = "codebase_search"
                # Check if query is empty or whitespace
                if not tool_data.get("query", "").strip():
                    LOG.warning(f"Empty query in codebase_search tool call")
                else:
                    LOG.info(f"Found implicit codebase_search tool call: {tool_data}")
                    return {"type": "tool_calls", "tool_call": tool_data}
            except json.JSONDecodeError as e:
                LOG.warning(f"Failed to parse implicit codebase_search tool call JSON: {e}")

        # Try repo tree tool
        json_match = re.search(repo_tree_pattern, dialog_text)
        if json_match:
            try:
                tool_data = json.loads(json_match.group())
                tool_data["tool"] = "repo_tree"
                LOG.info(f"Found implicit repo_tree tool call: {tool_data}")
                return {"type": "tool_calls", "tool_call": tool_data}
            except json.JSONDecodeError as e:
                LOG.warning(f"Failed to parse implicit repo_tree tool call JSON: {e}")

        # Try simple search query detection in general dialog text
        # Look for common search patterns throughout the dialog
        search_patterns_general = [
            r'search\s+for\s+(?:\'|")?([A-Za-z_][A-Za-z0-9_]*)(?:\'|")?',
            r'find\s+(?:\'|")?([A-Za-z_][A-Za-z0-9_]*)(?:\'|")?',
            r'look\s+for\s+(?:\'|")?([A-Za-z_][A-Za-z0-9_]*)(?:\'|")?',
            r'(?:\'|")?([A-Za-z_][A-Za-z0-9_]*)\s+function(?:\'|")?',
            r'(?:\'|")?([A-Za-z_][A-Za-z0-9_]*)\s+class(?:\'|")?',
            r'(?:\'|")?([A-Za-z_][A-Za-z0-9_]*)\s+method(?:\'|")?',
            r'(?:\'|")?([A-Za-z_][A-Za-z0-9_]*)\s+variable(?:\'|")?',
            # Also look for standalone terms that might be function/class names
            r'(?:\'|")?([A-Za-z_][A-Za-z0-9_]*[A-Z][A-Za-z0-9_]*)(?:\'|")?',  # CamelCase
            r'(?:\'|")?([a-z_][a-z0-9_]*)(?:\'|")?',  # snake_case
            # Specific pattern for quoted identifiers
            r'["\']([A-Za-z_][A-Za-z0-9_]*[A-Za-z0-9_]*)["\']',  # Quoted identifiers
        ]

        for pattern in search_patterns_general:
            search_match = re.search(pattern, dialog_text, re.IGNORECASE)
            if search_match:
                search_term = search_match.group(1).strip()
                # Clean up the search term
                search_term = re.sub(r'[\'"]', "", search_term)
                # Filter out common words that shouldn't be searched
                common_words = {
                    "the",
                    "and",
                    "or",
                    "but",
                    "in",
                    "on",
                    "at",
                    "to",
                    "for",
                    "of",
                    "with",
                    "by",
                    "is",
                    "are",
                    "was",
                    "were",
                    "be",
                    "been",
                    "have",
                    "has",
                    "had",
                    "do",
                    "does",
                    "did",
                    "will",
                    "would",
                    "could",
                    "should",
                    "may",
                    "might",
                    "can",
                    "this",
                    "that",
                    "these",
                    "those",
                    "a",
                    "an",
                    "as",
                    "if",
                    "then",
                    "else",
                    "when",
                    "where",
                    "why",
                    "how",
                    "what",
                    "which",
                    "who",
                    "whom",
                    "whose",
                    "need",
                    "find",
                    "search",
                    "look",
                    "function",
                    "class",
                    "method",
                    "variable",
                    "query",
                }
                if (
                    search_term
                    and len(search_term) > 2  # Ensure it's not just a short word
                    and search_term.lower() not in common_words
                    and not search_term.isdigit()
                ):  # Don't search for pure numbers
                    tool_data = {
                        "tool": "codebase_search",
                        "query": search_term,
                    }
                    LOG.info(f"Found implicit search request: {tool_data['query']}")
                    return {"type": "tool_calls", 'tool_call': tool_data}

        # Check for empty JSON object (repo_tree tool)
        empty_json_match = re.search(r"^\s*\{\s*\}\s*$", dialog_text.strip())
        if empty_json_match:
            tool_data = {"tool": "repo_tree"}
            LOG.info(f"Found implicit empty repo_tree tool call: {tool_data}")
            return {"type": "tool_calls", "tool_call": tool_data}

        LOG.warning("No ###Tool, ###Locations, or implicit tool calls found in dialog output")
        return None
