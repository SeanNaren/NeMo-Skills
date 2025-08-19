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


def connected_tree_repo_dict(repo_dict: dict, target_file: str = None, show_line_counts: bool = True):
    """
    Generate a connected tree/graph representation showing import dependencies.

    Args:
        repo_dict: Repository dictionary structure
        target_file: Optional file path to focus on (if None, shows whole repo dependency graph)
        show_line_counts: Whether to show line counts for files
    """
    import re

    def extract_imports_from_file(file_node):
        """Extract import statements from a file node."""
        if not isinstance(file_node, dict) or "text" not in file_node:
            return []

        imports = set()
        lines = file_node["text"]

        for line in lines:
            line = line.strip()
            # Match various import patterns
            if line.startswith('import ') or line.startswith('from '):
                # Handle "from module import ..." and "import module"
                if line.startswith('from '):
                    # from module.submodule import something
                    match = re.match(r'from\s+([^\s]+)\s+import', line)
                    if match:
                        module = match.group(1)
                        imports.add(module)
                elif line.startswith('import '):
                    # import module, module2
                    match = re.match(r'import\s+(.+)', line)
                    if match:
                        modules = match.group(1).split(',')
                        for module in modules:
                            module = module.strip().split(' as ')[0]  # Remove 'as alias'
                            imports.add(module)

        return list(imports)

    def normalize_module_to_file(module, all_files):
        """Convert module name to actual file paths in the repo."""
        matches = []

        # Direct file name match (e.g., 'utils' -> 'utils.py' or 'path/utils.py')
        module_base = module.split('.')[0]
        for file_path in all_files:
            file_name = file_path.split('/')[-1]
            file_base = file_name.replace('.py', '')

            # Exact match
            if file_base == module_base:
                matches.append(file_path)
            # Module path match (e.g., 'src.utils' -> 'src/utils.py')
            elif module.replace('.', '/') + '.py' == file_path:
                matches.append(file_path)
            # Partial path match
            elif module.replace('.', '/') in file_path:
                matches.append(file_path)

        return matches

    def collect_file_dependencies(structure):
        """Collect all files with their dependencies."""
        files_data = {}

        def traverse(d, path=""):
            for key, value in d.items():
                current_file_path = f"{path}/{key}" if path else key

                if isinstance(value, dict) and "text" in value:
                    # It's a file
                    imports = extract_imports_from_file(value)
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
        """Build the dependency tree representation."""
        if target_file:
            # Show dependency tree for specific file
            if target_file not in files_data:
                return f"ERROR: File '{target_file}' not found in repository."

            file_info = files_data[target_file]
            line_count_str = f" ({file_info['line_count']} lines)" if file_info['line_count'] is not None else ""

            lines = [
                f"DEPENDENCY TREE FOR: {target_file}{line_count_str}",
                "",
                f"{target_file}{line_count_str}",
            ]

            # Show what this file imports (dependencies)
            if file_info['dependencies']:
                lines.append("├── IMPORTS:")
                for i, dep in enumerate(sorted(file_info['dependencies'])):
                    dep_info = files_data[dep]
                    dep_line_count = f" ({dep_info['line_count']} lines)" if dep_info['line_count'] is not None else ""
                    connector = "│   ├──" if i < len(file_info['dependencies']) - 1 else "│   └──"
                    lines.append(f"{connector} {dep}{dep_line_count}")

                    # Show second-level dependencies (what the dependencies import)
                    if dep_info['dependencies']:
                        sub_deps = [d for d in dep_info['dependencies'] if d != target_file][:3]  # Limit to 3, avoid cycles
                        for j, sub_dep in enumerate(sub_deps):
                            sub_dep_info = files_data[sub_dep]
                            sub_dep_line_count = f" ({sub_dep_info['line_count']} lines)" if sub_dep_info['line_count'] is not None else ""
                            sub_connector = "│   │   ├──" if j < len(sub_deps) - 1 else "│   │   └──"
                            if i == len(file_info['dependencies']) - 1:  # Last dependency
                                sub_connector = "    │   ├──" if j < len(sub_deps) - 1 else "    │   └──"
                            lines.append(f"{sub_connector} {sub_dep}{sub_dep_line_count}")
            else:
                lines.append("├── IMPORTS: (none)")

            # Show what imports this file (dependents)
            if file_info['dependents']:
                lines.append("└── IMPORTED BY:")
                for i, dep in enumerate(sorted(file_info['dependents'])):
                    dep_info = files_data[dep]
                    dep_line_count = f" ({dep_info['line_count']} lines)" if dep_info['line_count'] is not None else ""
                    connector = "    ├──" if i < len(file_info['dependents']) - 1 else "    └──"
                    lines.append(f"{connector} {dep}{dep_line_count}")
            else:
                lines.append("└── IMPORTED BY: (none)")

            return "\n".join(lines)

        else:
            # Show full repository dependency graph
            lines = ["REPOSITORY DEPENDENCY GRAPH", "", "FILES BY DEPENDENCY LEVEL:", ""]

            # Calculate dependency levels (0 = no dependencies, higher = more dependencies)
            levels = {}
            for file_path, file_info in files_data.items():
                dep_count = len(file_info['dependencies'])
                if dep_count not in levels:
                    levels[dep_count] = []
                levels[dep_count].append(file_path)

            for level in sorted(levels.keys()):
                lines.append(f"LEVEL {level} ({level} dependencies):")
                for file_path in sorted(levels[level]):
                    file_info = files_data[file_path]
                    line_count_str = f" ({file_info['line_count']} lines)" if file_info['line_count'] is not None else ""
                    dependent_count = len(file_info['dependents'])
                    lines.append(f"  {file_path}{line_count_str} → used by {dependent_count} files")

                    # Show what this file imports (concise)
                    if file_info['dependencies']:
                        dep_names = [dep.split('/')[-1] for dep in file_info['dependencies'][:3]]
                        if len(file_info['dependencies']) > 3:
                            dep_names.append(f"...+{len(file_info['dependencies']) - 3} more")
                        lines.append(f"    imports: {', '.join(dep_names)}")
                lines.append("")

            # Show most connected files (files with highest total connections)
            lines.append("MOST CONNECTED FILES:")
            connections = []
            for file_path, file_info in files_data.items():
                total_connections = len(file_info['dependencies']) + len(file_info['dependents'])
                if total_connections > 0:
                    connections.append((total_connections, file_path, file_info))

            connections.sort(reverse=True)
            for i, (conn_count, file_path, file_info) in enumerate(connections[:5]):
                line_count_str = f" ({file_info['line_count']} lines)" if file_info['line_count'] is not None else ""
                lines.append(f"  {i+1}. {file_path}{line_count_str}")
                lines.append(f"     imports {len(file_info['dependencies'])}, imported by {len(file_info['dependents'])}")

            return "\n".join(lines)

    # Main logic
    files_data = collect_file_dependencies(repo_dict['structure'])
    return build_dependency_tree(files_data, target_file)


def extract_locations_from_patch(patch: str) -> List[Dict[str, Any]]:
    """Extract changed line ranges from a git patch using ORIGINAL file line numbers.

    Returns list of dicts: file_path, start_line, end_line, raw.
    Tracks where changes occur in the original file.
    """
    if not patch:
        return []

    locations = []
    current_file = None
    original_line = 0
    new_line = 0
    
    for line in patch.splitlines():
        # File path
        if line.startswith("--- "):
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
                # Line added - if previous line wasn't a removal, this is a pure addition
                # The location in original file is where it would be inserted
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
