import json
import re
import ast
from typing import Dict
import logging
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


class DialogProcessor:
    """Processes dialog output to extract tool calls and locations."""

    @staticmethod
    def extract_response(dialog_text: str, config=None):
        if "<tool_call>" in dialog_text:
            LOG.info("Found <tool_call> block in output")
            return DialogProcessor._extract_tool_calls(dialog_text)
        elif "<locations>" in dialog_text:
            LOG.info("Found <locations> block in output")
            return DialogProcessor._extract_locations(dialog_text)
        else:
            # Check if implicit tool detection is enabled
            enable_implicit = getattr(config, 'enable_implicit_tool_detection', True) if config else True
            if enable_implicit:
                LOG.warning("No <tool_call> or <locations> found, checking for implicit tool calls")
                return DialogProcessor._extract_implicit_tool_calls(dialog_text, config)
            else:
                LOG.warning("No <tool_call> or <locations> found, and implicit tool detection is disabled")
                return None

    # USED
    @staticmethod
    def _extract_tool_calls(dialog_text: str) -> Dict:
        """Extract tool calls from dialog text."""
        # Look for <tool_call>content</tool_call> pattern
        tool_match = re.search(r"<tool_call>(.*?)</tool_call>", dialog_text, re.DOTALL)

        if tool_match:
            tool_content = tool_match.group(1).strip()
            
            # Try to parse as function call first (new format)
            function_result = DialogProcessor._parse_function_call(tool_content)
            if function_result:
                return function_result
            
            # Fall back to JSON format (old format)
            try:
                tool_data = json.loads(tool_content)
                
                # The old format expects {"tool_name": {arguments}} structure
                # Extract the tool name and arguments
                if len(tool_data) == 1:
                    tool_name = list(tool_data.keys())[0]
                    tool_args = tool_data[tool_name]
                    
                    # Create the format expected by the tool executor
                    result_data = {"tool": tool_name}
                    result_data.update(tool_args)
                    
                    return {"type": "tool_calls", "tool_call": result_data}
                else:
                    LOG.warning(f"Unexpected tool call format: {tool_data}")
                    return None
                    
            except json.JSONDecodeError as e:
                LOG.warning(f"Failed to parse tool call as JSON or function: {e}")
                return None
        return None

    @staticmethod
    def _parse_function_call(function_text: str) -> Dict:
        """Parse function call syntax like 'view_file(path="file.py", view_range=[1,10])'"""
        try:
            # Clean up the function text
            function_text = function_text.strip()
            
            # Use regex to extract function name and arguments
            # Pattern: function_name(arguments...)
            function_pattern = r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)$'
            match = re.match(function_pattern, function_text, re.DOTALL)
            
            if not match:
                return None
                
            function_name = match.group(1)
            args_text = match.group(2).strip()
            
            # Create result with tool name
            result_data = {"tool": function_name}
            
            # Parse arguments if any
            if args_text:
                # Handle the case where we have arguments
                try:
                    # Try to parse as a function call using ast.parse
                    # We need to create a valid function call for ast to parse
                    fake_call = f"dummy({args_text})"
                    tree = ast.parse(fake_call, mode='eval')
                    
                    if isinstance(tree.body, ast.Call):
                        call_node = tree.body
                        
                        # Process positional arguments
                        for i, arg in enumerate(call_node.args):
                            if function_name == "view_file" and i == 0:
                                result_data["path"] = ast.literal_eval(arg)
                            elif function_name == "view_file" and i == 1:
                                result_data["view_range"] = ast.literal_eval(arg)
                            elif function_name == "codebase_search" and i == 0:
                                result_data["query"] = ast.literal_eval(arg)
                        
                        # Process keyword arguments
                        for keyword in call_node.keywords:
                            result_data[keyword.arg] = ast.literal_eval(keyword.value)
                            
                except (SyntaxError, ValueError) as e:
                    LOG.warning(f"Failed to parse function arguments: {e}")
                    return None
            
            # Validate the function call
            if function_name in ["view_file", "codebase_search", "repo_tree"]:
                LOG.info(f"Parsed function call: {function_name} with args: {result_data}")
                return {"type": "tool_calls", "tool_call": result_data}
            else:
                LOG.warning(f"Unknown function name: {function_name}")
                return None
                
        except Exception as e:
            LOG.warning(f"Error parsing function call: {e}")
            return None

    # USED
    @staticmethod
    def _extract_locations(dialog_text: str) -> Dict:
        """Extract location predictions from dialog text."""
        locations = []
        # Look for <locations>content</locations> pattern
        locations_match = re.search(r"<locations>(.*?)</locations>", dialog_text, re.DOTALL)

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
    def _extract_implicit_tool_calls(dialog_text: str, config=None) -> Dict:
        """Extract tool calls that appear as JSON without <tool_call> wrapper."""
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
                        tool_data["tool"] = "view_file"
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
            file_extensions = getattr(config, 'file_extensions', None) if config else None
            if file_extensions:
                # Create pattern from configured file extensions
                extensions_pattern = "|".join(re.escape(ext) for ext in file_extensions)
                file_path_pattern = rf'(?:\'|")?([^\'"\s]+?\.(?:{extensions_pattern}))(?:\'|")?'
            else:
                # Fallback to default extensions if none provided
                file_path_pattern = r'(?:\'|")?([^\'"\s]+?\.(?:py|cpp|h|hpp|java|js|ts|rb|go|rs|cs|php))(?:\'|")?'
            
            file_path_match = re.search(file_path_pattern, after_think)
            if file_path_match:
                tool_data = {
                    "tool": "view_file",
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
                tool_data["tool"] = "view_file"
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
                common_words_list = getattr(config, 'common_words_filter', None) if config else None
                common_words = set(common_words_list)
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

        LOG.warning("No <tool_call>, <locations>, or implicit tool calls found in dialog output")
        return None
