import re
from typing import List, Dict, Any
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


def tree_repo_dict(repo_dict: dict, show_line_counts: bool = True):
    def build_level(d, prefix=""):
        lines = []
        items = list(d.keys())

        for i, key in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└── " if is_last else "|-- "

            node = d[key]
            is_folder = isinstance(node, dict) and set(node.keys()) != {'classes', 'functions', 'text'}

            # Add line count for files if enabled
            if show_line_counts and not is_folder and isinstance(node, dict) and "text" in node and isinstance(node["text"], list):
                line_count = len(node["text"])
                lines.append(f"{prefix}{connector}{key} ({line_count} lines)")
            else:
                lines.append(f"{prefix}{connector}{key}")

            if is_folder:
                new_prefix = prefix + ("    " if is_last else "|   ")
                lines.extend(build_level(node, new_prefix))
        return lines

    all_lines = build_level(repo_dict['structure'])
    return ".\n" + "\n".join(all_lines)


def calculate_ground_truth_percentage(repo_dict: dict, locations: List[Dict[str, Any]], 
                                    exclude_dirs: list, file_extensions: list) -> tuple:
    """Calculate percentage of ground truth files that exist in the filtered repository.
    
    Args:
        repo_dict: The filtered repository dictionary with 'structure' key
        locations: List of location dicts from extract_locations_from_patch
        exclude_dirs: List of excluded directory names
        file_extensions: List of allowed file extensions
        
    Returns:
        Tuple of (percentage, debug_info_dict)
    """
    import logging
    LOG = logging.getLogger(__name__)
    
    if not locations or 'structure' not in repo_dict:
        return 0.0, {}
    
    # Get all files in the repo tree
    all_files = set()
    def collect_files(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                # Check if it's a file based on the filter_repo_dict logic
                # From filter_repo_dict: is_folder = isinstance(value, dict) and set(value.keys()) != {'classes', 'functions', 'text'}
                # So a file is NOT a folder
                is_file = False
                if value is None:
                    # Some files might be represented as None
                    is_file = True
                elif isinstance(value, dict):
                    # A file has exactly the keys {'classes', 'functions', 'text'}
                    # A folder is any other dict
                    is_folder = set(value.keys()) != {'classes', 'functions', 'text'}
                    is_file = not is_folder
                
                if is_file:
                    file_path = f"{path}/{key}" if path else key
                    all_files.add(file_path)
                    LOG.debug(f"Found file in repo: {file_path}")
                elif isinstance(value, dict):  # It's a directory
                    new_path = f"{path}/{key}" if path else key
                    collect_files(value, new_path)
    
    collect_files(repo_dict['structure'])
    LOG.info(f"Total files found in filtered repo: {len(all_files)}")
    
    # Check how many ground truth files exist
    ground_truth_files = {loc['file_path'] for loc in locations}
    LOG.info(f"Ground truth files from patch: {ground_truth_files}")
    
    # Check if ground truth files were filtered out by extension
    filtered_out_by_extension = []
    for gt_file in ground_truth_files:
        if '.' in gt_file:
            ext = gt_file.split('.')[-1]
            if ext not in file_extensions:
                filtered_out_by_extension.append((gt_file, ext))
    
    if filtered_out_by_extension:
        LOG.warning(f"Ground truth files filtered out by extension: {filtered_out_by_extension}")
        LOG.info(f"Allowed extensions: {file_extensions}")
    
    # Check if ground truth files are in excluded directories
    filtered_out_by_dir = []
    for gt_file in ground_truth_files:
        path_parts = gt_file.split('/')
        for part in path_parts[:-1]:  # Check all directory parts except filename
            if part in exclude_dirs:
                filtered_out_by_dir.append((gt_file, part))
                break
    
    if filtered_out_by_dir:
        LOG.warning(f"Ground truth files in excluded directories: {filtered_out_by_dir}")
        LOG.info(f"Excluded directories: {exclude_dirs[:10]}...")  # Show first 10
    
    # Check for exact matches
    existing_files = ground_truth_files.intersection(all_files)
    
    # Also check for potential path mismatches (e.g., leading slashes, different separators)
    normalized_existing = []
    if not existing_files and ground_truth_files:
        LOG.debug("No exact matches found, checking for path variations...")
        # Normalize paths for comparison
        normalized_repo_files = {f.strip('/').replace('//', '/') for f in all_files}
        normalized_gt_files = {f.strip('/').replace('//', '/') for f in ground_truth_files}
        normalized_existing = normalized_gt_files.intersection(normalized_repo_files)
        if normalized_existing:
            LOG.warning(f"Found {len(normalized_existing)} files with normalized paths, but paths don't match exactly")
            LOG.debug(f"Sample repo files: {list(all_files)[:5]}")
            LOG.debug(f"Sample GT files: {list(ground_truth_files)[:5]}")
    
    percentage = 0.0
    if ground_truth_files:
        percentage = (len(existing_files) / len(ground_truth_files)) * 100
        
    LOG.info(f"Ground truth files: {len(ground_truth_files)}, Existing in repo: {len(existing_files)} ({percentage:.1f}%)")
    
    # Provide explanation if percentage is 0
    if ground_truth_files and percentage == 0:
        LOG.warning("0% ground truth files found in filtered repo structure!")
        if filtered_out_by_extension:
            LOG.warning(f"  - {len(filtered_out_by_extension)} files filtered by extension")
        if filtered_out_by_dir:
            LOG.warning(f"  - {len(filtered_out_by_dir)} files in excluded directories")
        if not filtered_out_by_extension and not filtered_out_by_dir:
            LOG.warning("  - Files may have been excluded during repo loading or have path mismatches")
    
    # Collect missing files for detailed reporting
    missing_files = ground_truth_files - existing_files
    missing_files_details = []
    
    for missing_file in missing_files:
        reason = "unknown"
        details = {"file": missing_file}
        
        # Check if filtered by extension
        if '.' in missing_file:
            ext = missing_file.split('.')[-1]
            if ext not in file_extensions:
                reason = "filtered_by_extension"
                details["extension"] = ext
        
        # Check if filtered by directory
        path_parts = missing_file.split('/')
        for part in path_parts[:-1]:  # Check all directory parts except filename
            if part in exclude_dirs:
                reason = "filtered_by_directory"
                details["excluded_dir"] = part
                break
        
        if reason == "unknown":
            # File wasn't filtered by our rules, might be missing from repo
            reason = "not_in_repository"
        
        details["reason"] = reason
        missing_files_details.append(details)
    
    debug_info = {
        'total_repo_files': len(all_files),
        'total_ground_truth_files': len(ground_truth_files),
        'existing_files': len(existing_files),
        'filtered_by_extension': len(filtered_out_by_extension),
        'filtered_by_dir': len(filtered_out_by_dir),
        'normalized_matches': len(normalized_existing) if normalized_existing else 0,
        'missing_files_details': missing_files_details
    }
    
    return percentage, debug_info


def connected_tree_repo_dict(repo_dict: dict, target_file: str = None, show_line_counts: bool = False):
    """
    Generate a concise connected tree showing ONLY internal project dependencies.
    
    Key features:
    - Shows ONLY internal imports (nemo_skills.*, relative imports)
    - Excludes ALL external libraries (no numpy, torch, hydra, etc.)
    - Skips test/docs/build/cache directories
    - Only analyzes Python files (.py)
    - Limits display to 10 imports/dependents per file
    - Summary mode shows top 10 most connected files

    Args:
        repo_dict: Repository dictionary structure
        target_file: Optional file path to focus on (if None, shows summary)
        show_line_counts: Whether to show line counts for files (default: False)
    
    Returns:
        Concise string showing internal project structure and dependencies
    
    Example output for specific file:
        ═══ nemo_skills/inference/eval/locagent.py ═══
        
        → IMPORTS (3):
           • nemo_skills.inference.generate
           • nemo_skills.utils
           • .locagent_utils.dialog_processor
        
        ← IMPORTED BY (1):
           • nemo_skills/cli/eval.py
    """
    import re
    
    def detect_internal_modules(structure):
        """Detect the base module names from the repository structure."""
        internal_modules = set()
        
        # Look at top-level directories that contain Python files
        for key, value in structure.items():
            if isinstance(value, dict):
                # Check if this directory contains Python files
                has_python_files = False
                
                def check_for_python(d):
                    for k, v in d.items():
                        if isinstance(v, dict):
                            if "text" in v and k.endswith('.py'):
                                return True
                            if check_for_python(v):
                                return True
                    return False
                
                if check_for_python(value) or (key + '.py' in structure):
                    internal_modules.add(key)
        
        # Also check for common patterns
        if 'nemo_skills' in internal_modules or any('nemo' in m for m in internal_modules):
            internal_modules.update(['nemo_skills', 'nemo'])
        
        return tuple(internal_modules) if internal_modules else ('nemo_skills', 'nemo')

    def extract_imports_from_file(file_node, internal_modules=None):
        """Extract ONLY internal import statements from a file node."""
        if not isinstance(file_node, dict) or "text" not in file_node:
            return []

        imports = set()
        lines = file_node["text"]
        
        # Use provided internal modules or default patterns
        if internal_modules is None:
            internal_modules = ('nemo_skills', 'nemo')

        for line in lines:
            line = line.strip()
            # Match various import patterns
            if line.startswith('import ') or line.startswith('from '):
                # Handle "from module import ..." and "import module"
                if line.startswith('from '):
                    # Handle relative imports
                    if line.startswith('from .'):
                        # Relative import - always internal
                        match = re.match(r'from\s+(\.[^\s]+)\s+import', line)
                        if match:
                            imports.add(match.group(1))
                    else:
                        # Absolute import
                        match = re.match(r'from\s+([^\s]+)\s+import', line)
                        if match:
                            module = match.group(1)
                            # Only keep if it's an internal module
                            if any(module.startswith(internal) for internal in internal_modules):
                                imports.add(module)
                elif line.startswith('import '):
                    # import module, module2
                    match = re.match(r'import\s+(.+)', line)
                    if match:
                        modules = match.group(1).split(',')
                        for module in modules:
                            module = module.strip().split(' as ')[0]  # Remove 'as alias'
                            # Only keep if it's an internal module
                            if any(module.startswith(internal) for internal in internal_modules):
                                imports.add(module)

        return list(imports)

    def normalize_module_to_file(module, all_files):
        """Convert internal module name to actual file paths in the repo."""
        matches = []
        
        # Handle relative imports
        if module.startswith('.'):
            # Relative imports are harder to resolve without context
            # For now, skip them as they're already internal
            return matches
        
        # Convert module path to file path patterns
        # e.g., 'nemo_skills.inference.eval' -> 'nemo_skills/inference/eval'
        module_path = module.replace('.', '/')
        
        for file_path in all_files:
            # Check for exact module match with .py extension
            if file_path == module_path + '.py':
                matches.append(file_path)
            # Check for __init__.py in package
            elif file_path == module_path + '/__init__.py':
                matches.append(file_path)
            # Check if the module path is part of the file path
            elif module_path in file_path:
                # Make sure it's a proper path component match
                # e.g., 'utils' shouldn't match 'myutils.py'
                path_parts = file_path.split('/')
                module_parts = module_path.split('/')
                for i in range(len(path_parts) - len(module_parts) + 1):
                    if path_parts[i:i+len(module_parts)] == module_parts:
                        matches.append(file_path)
                        break

        return list(set(matches))  # Remove duplicates

    def collect_file_dependencies(structure, internal_modules):
        """Collect all files with their dependencies."""
        files_data = {}
        
        # Directories to skip for faster processing
        skip_dirs = {'__pycache__', '.git', 'node_modules', 'venv', '.env', 'dist', 'build', 
                    'tests', 'test', 'testing', 'docs', 'documentation', 'migrations',
                    '.pytest_cache', '.mypy_cache', '.tox', 'htmlcov', 'coverage'}

        def traverse(d, path=""):
            for key, value in d.items():
                # Skip certain directories
                if key in skip_dirs:
                    continue
                    
                current_file_path = f"{path}/{key}" if path else key

                if isinstance(value, dict) and "text" in value:
                    # It's a file - only process Python files
                    if key.endswith('.py'):
                        imports = extract_imports_from_file(value, internal_modules)
                        if imports:  # Only include files that have internal imports
                            files_data[current_file_path] = {
                                'imports': imports,
                                'line_count': len(value["text"]) if show_line_counts else None,
                                'dependencies': [],  # Will be filled later
                                'dependents': [],  # Will be filled later
                            }
                elif isinstance(value, dict):
                    # It's a directory
                    traverse(value, current_file_path)

        traverse(structure)

        # Now resolve imports to actual files
        all_files = list(files_data.keys())
        for file_path, file_info in files_data.items():
            for imported_module in file_info['imports']:
                matching_files = normalize_module_to_file(imported_module, all_files)
                for match in matching_files:
                    if match != file_path:  # Don't self-reference
                        file_info['dependencies'].append(match)
                        files_data[match]['dependents'].append(file_path)

        # Remove duplicates
        for file_info in files_data.values():
            file_info['dependencies'] = list(set(file_info['dependencies']))
            file_info['dependents'] = list(set(file_info['dependents']))

        return files_data

    def build_dependency_tree(files_data, target_file=None):
        """Build a concise dependency tree representation."""
        if target_file:
            # Show dependency info for specific file
            if target_file not in files_data:
                return f"ERROR: File '{target_file}' not found in repository."

            file_info = files_data[target_file]
            line_count = f" [{file_info['line_count']}L]" if show_line_counts and file_info['line_count'] is not None else ""
            
            lines = [f"═══ {target_file}{line_count} ═══"]
            
            # Show imports (what this file depends on)
            if file_info['dependencies']:
                lines.append(f"\n→ IMPORTS ({len(file_info['dependencies'])}):")
                for dep in sorted(file_info['dependencies'])[:10]:  # Limit to 10 most relevant
                    lines.append(f"   • {dep}")
                if len(file_info['dependencies']) > 10:
                    lines.append(f"   ... and {len(file_info['dependencies']) - 10} more")
            else:
                lines.append("\n→ IMPORTS: None")
            
            # Show dependents (what depends on this file)
            if file_info['dependents']:
                lines.append(f"\n← IMPORTED BY ({len(file_info['dependents'])}):")
                for dep in sorted(file_info['dependents'])[:10]:  # Limit to 10 most relevant
                    lines.append(f"   • {dep}")
                if len(file_info['dependents']) > 10:
                    lines.append(f"   ... and {len(file_info['dependents']) - 10} more")
            else:
                lines.append("\n← IMPORTED BY: None")
            
            return "\n".join(lines)
        
        else:
            # Show summary of most connected files only
            lines = ["═══ REPOSITORY CONNECTION SUMMARY ═══\n"]
            
            # Find most connected files (by total connections)
            connection_scores = []
            for file_path, file_info in files_data.items():
                total_connections = len(file_info['dependencies']) + len(file_info['dependents'])
                if total_connections > 0:  # Only show files with connections
                    connection_scores.append((total_connections, file_path, file_info))
            
            # Sort by connection count
            connection_scores.sort(reverse=True)
            
            # Show top 10 most connected files
            lines.append("TOP CONNECTED FILES:")
            for i, (score, file_path, file_info) in enumerate(connection_scores[:10]):
                imports = len(file_info['dependencies'])
                imported_by = len(file_info['dependents'])
                lines.append(f"{i+1:2d}. {file_path}")
                lines.append(f"    → imports: {imports}, ← imported by: {imported_by}")
            
            if len(connection_scores) > 10:
                lines.append(f"\n... and {len(connection_scores) - 10} more files with connections")
            
            # Summary statistics
            total_files = len(files_data)
            connected_files = len(connection_scores)
            lines.append(f"\nTOTAL: {connected_files}/{total_files} files have import connections")
            
            return "\n".join(lines)

    # Main logic
    # Auto-detect internal modules from the repository structure
    internal_modules = detect_internal_modules(repo_dict['structure'])
    
    # Log detected modules for debugging
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"Detected internal modules: {internal_modules}")
    
    # Collect dependencies using only internal modules
    files_data = collect_file_dependencies(repo_dict['structure'], internal_modules)
    
    return build_dependency_tree(files_data, target_file)


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
                locations.append({
                    'file_path': current_file, 
                    'start_line': original_line, 
                    'end_line': original_line, 
                    'raw': f"{current_file}:L{original_line}-L{original_line}"
                })
                original_line += 1
            elif line.startswith("+") and not line.startswith("+++"):
                # Line added
                if is_new_file:
                    # For new files, track line 1 as the change location
                    # We only add this once per new file
                    if not exclude_new_files and not any(loc['file_path'] == current_file for loc in locations):
                        locations.append({
                            'file_path': current_file,
                            'start_line': 1,
                            'end_line': 1,
                            'raw': f"{current_file}:L1-L1"
                        })
                else:
                    # For existing files, track where the addition would be inserted
                    if not locations or locations[-1]['end_line'] != original_line - 1:
                        # Pure addition at current position in original
                        locations.append({
                            'file_path': current_file, 
                            'start_line': original_line, 
                            'end_line': original_line, 
                            'raw': f"{current_file}:L{original_line}-L{original_line}"
                        })
                new_line += 1
            else:
                # Context line - advances both counters
                if not is_new_file:  # Only advance for existing files
                    original_line += 1
                new_line += 1

    # Merge adjacent locations
    merged = []
    for loc in locations:
        if merged and loc['file_path'] == merged[-1]['file_path'] and loc['start_line'] <= merged[-1]['end_line'] + 1:
            # Extend the previous location
            merged[-1]['end_line'] = max(merged[-1]['end_line'], loc['end_line'])
            merged[-1]['raw'] = f"{merged[-1]['file_path']}:L{merged[-1]['start_line']}-L{merged[-1]['end_line']}"
        else:
            merged.append(loc)
    
    return merged
