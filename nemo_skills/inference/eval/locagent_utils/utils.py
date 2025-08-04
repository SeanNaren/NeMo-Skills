import re
from typing import List
import os


def get_version():
    """Get the current version from VERSION file."""
    version_file = os.path.join(os.path.dirname(__file__), "VERSION")
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            return f.read().strip()
    return "0.0.0"

def tree_structure_from_pickle(data: dict, exclude_dirs: set):
    def build_level(d, prefix=""):
        lines = []
        items = list(d.keys())
        filtered_items = [key for key in items if key not in exclude_dirs]
        for i, key in enumerate(filtered_items):
            is_last = i == len(filtered_items) - 1
            connector = "└── " if is_last else "|-- "
            lines.append(f"{prefix}{connector}{key}")

            node = d[key]
            is_folder = isinstance(node, dict) and set(node.keys()) != {'classes', 'functions', 'text'}

            if is_folder:
                new_prefix = prefix + ("    " if is_last else "|   ")
                lines.extend(build_level(node, new_prefix))
        return lines

    all_lines = build_level(data['structure'])
    return ".\n" + "\n".join(all_lines)

def extract_locations_from_patch(patch: str) -> List[str]:
    """Extract file locations from git patch."""
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
                locations.append(f"{current_file}:L{start_line}-L{end_line}")

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
        locations.append(f"{current_file}:L{start_line}-L{end_line}")

    return locations
