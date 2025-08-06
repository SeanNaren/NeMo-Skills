import re
from typing import List, Dict
import os


def get_version():
    """Get the current version from VERSION file."""
    version_file = os.path.join(os.path.dirname(__file__), "VERSION")
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            return f.read().strip()
    return "0.0.0"

def filter_repo_dict(repo_dict: dict, exclude_dirs: list, file_extensions: list) -> dict:
    """Filter repo_dict by removing excluded directories and files with unwanted extensions.
    
    Returns a new repo_dict with filtered structure, removing empty directories.
    """
    def filter_level(d):
        filtered = {}
        for key, value in d.items():
            # Skip excluded directories
            if key in exclude_dirs:
                continue
                
            # Check if it's a file (has extension) and if extension is allowed
            if "." in key and key.split(".")[-1] not in file_extensions:
                continue
                
            # Determine if this is a folder
            is_folder = isinstance(value, dict) and set(value.keys()) != {'classes', 'functions', 'text'}
            
            if is_folder:
                # Recursively filter the folder
                filtered_subfolder = filter_level(value)
                # Only include the folder if it has content after filtering
                if filtered_subfolder:
                    filtered[key] = filtered_subfolder
            else:
                # Include files and leaf nodes
                filtered[key] = value
                
        return filtered
    
    # Create a new repo_dict with filtered structure
    filtered_repo_dict = repo_dict.copy()
    if 'structure' in repo_dict:
        filtered_repo_dict['structure'] = filter_level(repo_dict['structure'])
    
    return filtered_repo_dict


def tree_repo_dict(repo_dict: dict):
    def build_level(d, prefix=""):
        lines = []
        items = list(d.keys())
        
        for i, key in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└── " if is_last else "|-- "
            lines.append(f"{prefix}{connector}{key}")

            node = d[key]
            is_folder = isinstance(node, dict) and set(node.keys()) != {'classes', 'functions', 'text'}

            if is_folder:
                new_prefix = prefix + ("    " if is_last else "|   ")
                lines.extend(build_level(node, new_prefix))
        return lines

    all_lines = build_level(repo_dict['structure'])
    return ".\n" + "\n".join(all_lines)

def extract_locations_from_patch(patch: str) -> List[Dict[str, any]]:
    """Extract file locations from git patch and return as list of dictionaries with file_path, start_line, end_line, and raw format."""
    if not patch:
        return []

    locations = []
    current_file = None
    current_line = 0
    start_line = None
    end_line = None
    in_hunk = False

    for line in patch.split("\n"):
        # File path parsing
        if line.startswith(("--- ", "+++ ")):
            file_path = line[4:]
            if file_path.startswith(("a/", "b/")):
                file_path = file_path[2:]
            current_file = file_path

        # Hunk header parsing
        elif line.startswith("@@ "):
            # Finalize previous hunk
            if current_file and start_line is not None and end_line is not None:
                raw_format = f"{current_file}:L{start_line}-L{end_line}"
                locations.append({
                    'file_path': current_file,
                    'start_line': start_line,
                    'end_line': end_line,
                    'raw': raw_format
                })

            hunk_match = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
            if hunk_match:
                new_start = int(hunk_match.group(1))
                new_count = int(hunk_match.group(2)) if hunk_match.group(2) else 1
                current_line = new_start
                start_line = None
                end_line = None
                in_hunk = True

        elif in_hunk:
            if line.startswith("+") and not line.startswith("+++"):
                if start_line is None:
                    start_line = current_line
                end_line = current_line
                current_line += 1
            elif line.startswith("-") and not line.startswith("---"):
                continue  # Removed line
            else:
                current_line += 1  # Context line

    # Final hunk
    if current_file and start_line is not None and end_line is not None:
        raw_format = f"{current_file}:L{start_line}-L{end_line}"
        locations.append({
            'file_path': current_file,
            'start_line': start_line,
            'end_line': end_line,
            'raw': raw_format
        })

    return locations
