"""
Patch processing utilities for Artsiv.

This module provides functions for extracting information from git patches:
- Extracting file paths from patches
- Extracting changed line locations from patches
"""

import re
from typing import Any, Dict, List


class PatchProcessor:
    """Git patch processing utilities."""

    @staticmethod
    def extract_files_from_patch(patch: str) -> List[str]:
        """Extract unique file paths that are modified in a git patch.

        Returns list of file paths, excluding /dev/null for new files.
        """
        if not patch:
            return []

        files = set()

        for line in patch.splitlines():
            if line.startswith("--- "):
                file_path = line[4:]
                if file_path != "/dev/null":  # Skip /dev/null for new files
                    if file_path.startswith(("a/", "b/")):
                        file_path = file_path[2:]
                    files.add(file_path)
            elif line.startswith("+++ "):
                file_path = line[4:]
                if file_path != "/dev/null":  # Skip /dev/null (shouldn't happen for +++)
                    if file_path.startswith(("a/", "b/")):
                        file_path = file_path[2:]
                    files.add(file_path)

        return sorted(list(files))

    @staticmethod
    def extract_locations_from_patch(patch: str, exclude_new_files: bool = True) -> List[Dict[str, Any]]:
        """Extract changed line ranges from a git patch using ORIGINAL file line numbers.

        Args:
            patch: The git patch string to parse
            exclude_new_files: If True, excludes locations from newly created files (default: True)

        Returns list of dicts: file_path, start_line, end_line, raw.
        Tracks where changes occur in the original file.
        """
        if not patch:
            return []

        locations = []
        current_file = None
        is_new_file = False
        original_line = 0
        new_line = 0

        for line in patch.splitlines():
            # File path from --- line
            if line.startswith("--- "):
                file_path = line[4:]
                if file_path == "/dev/null":
                    # This is a new file, wait for +++ line
                    is_new_file = True
                    current_file = None
                else:
                    if file_path.startswith(("a/", "b/")):
                        file_path = file_path[2:]
                    current_file = file_path
                    is_new_file = False

            # File path from +++ line (for new files)
            elif line.startswith("+++ "):
                if is_new_file:
                    file_path = line[4:]
                    if file_path.startswith(("a/", "b/")):
                        file_path = file_path[2:]
                    current_file = file_path

            # Hunk header
            elif line.startswith("@@ "):
                # Parse: @@ -original_start[,original_count] +new_start[,new_count] @@
                m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
                if m:
                    original_line = int(m.group(1))
                    new_line = int(m.group(3))

            elif current_file and line:
                # Skip processing if current_file is /dev/null (shouldn't happen with fix above)
                if current_file == "/dev/null":
                    continue

                # Track changes in original file
                if line.startswith("-") and not line.startswith("---"):
                    # Line removed from original - this is a change location
                    locations.append(
                        {
                            'file_path': current_file,
                            'start_line': original_line,
                            'end_line': original_line,
                            'raw': f"{current_file}:L{original_line}-L{original_line}",
                        }
                    )
                    original_line += 1
                elif line.startswith("+") and not line.startswith("+++"):
                    # Line added
                    if is_new_file:
                        # For new files, track line 1 as the change location
                        # We only add this once per new file
                        if not exclude_new_files and not any(loc['file_path'] == current_file for loc in locations):
                            locations.append(
                                {
                                    'file_path': current_file,
                                    'start_line': 1,
                                    'end_line': 1,
                                    'raw': f"{current_file}:L1-L1",
                                }
                            )
                    else:
                        # For existing files, track where the addition would be inserted
                        if not locations or locations[-1]['end_line'] != original_line - 1:
                            # Pure addition at current position in original
                            locations.append(
                                {
                                    'file_path': current_file,
                                    'start_line': original_line,
                                    'end_line': original_line,
                                    'raw': f"{current_file}:L{original_line}-L{original_line}",
                                }
                            )
                    new_line += 1
                else:
                    # Context line - advances both counters
                    if not is_new_file:  # Only advance for existing files
                        original_line += 1
                    new_line += 1

        # Merge adjacent locations
        merged = []
        for loc in locations:
            if (
                merged
                and loc['file_path'] == merged[-1]['file_path']
                and loc['start_line'] <= merged[-1]['end_line'] + 1
            ):
                # Extend the previous location
                merged[-1]['end_line'] = max(merged[-1]['end_line'], loc['end_line'])
                merged[-1]['raw'] = f"{merged[-1]['file_path']}:L{merged[-1]['start_line']}-L{merged[-1]['end_line']}"
            else:
                merged.append(loc)

        return merged
