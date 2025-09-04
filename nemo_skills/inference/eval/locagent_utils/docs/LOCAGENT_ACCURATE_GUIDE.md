# LocAgent Accurate Technical Guide

Based on actual code implementation in `locagent.py`, `dialog_processor.py`, and `tool_executor.py`.

## Table of Contents
1. [Configuration - Actual Values](#configuration---actual-values)
2. [Available Tools](#available-tools)
3. [File Filtering](#file-filtering)
4. [Token Management](#token-management)
5. [Truncation Strategies](#truncation-strategies)
6. [Advanced Features](#advanced-features)
7. [Tool Call Formats](#tool-call-formats)
8. [Implementation Details](#implementation-details)

## Configuration - Actual Values

These are the ACTUAL default values from `LocalAgentGenerationConfig`:

```python
# Core settings
total_steps: int = 20                        # Maximum tool calls per problem
remove_thinking: bool = True                 # Strip <think> tags from output
mount_directory: str = "/repos/"             # Where repos are mounted

# File filtering - CRITICAL: Only Python and config files!
file_extensions: list = ["py", "cfg"]        # ONLY these extensions are included!

# Context management (recently updated)
max_seq_length: int = 262144                 # Maximum context length
tokens_to_generate: int = 81920              # Reserved for response

# Truncation
truncation_strategy: str = "bookend"         # Default strategy
enable_enhanced_context: bool = True         # Use better token counting
context_safety_margin: float = 0.9           # Use 90% of max context
use_tiktoken: bool = True                    # Use tiktoken if available

# Loop detection
enable_loop_detection: bool = True           # Detect repetitive behavior
loop_detection_threshold: int = 3            # Trigger after 3 identical calls

# Final turn prompt
enable_final_turn_prompt: bool = True        # Force predictions on last turn
final_turn_instruction_type: str = "aligned" # Match system prompt format
final_turn_threshold: float = 1.0            # Only on very last turn

# Summarization (currently disabled)
enable_turn_summarization: bool = False      # NOT ACTIVE

# Display
show_line_counts: bool = False               # Don't show line counts in tree
max_view_lines: int = 1000                   # Max lines per view_file call

# Implicit tool detection
enable_implicit_tool_detection: bool = True  # Detect tools without <tool_call>
```

## Available Tools

The agent has exactly 4 tools available (from `tool_executor.py`):

### 1. view_file (alias: "view")
```json
{
  "tool": "view_file",
  "path": "src/main.py",
  "view_range": [10, 50]  // Optional: [start, end] or [start, -1] for to-end
}
```
- Shows file contents with line numbers
- `view_range` is optional (shows entire file if omitted)
- Use `[start, -1]` to view from line to end of file
- Files are truncated to `max_view_lines` (1000) if too long

### 2. repo_tree
```json
{
  "tool": "repo_tree"
}
```
- Shows filtered repository structure
- Only includes files with extensions in `file_extensions` (py, cfg)
- Excludes directories in `exclude_dirs` list

### 3. codebase_search
```json
{
  "tool": "codebase_search",
  "query": "search_term"
}
```
- Case-insensitive search across all included files
- Shows context (20 lines before/after) around matches
- Limited to 3 snippets per file, top 5 files by match count

### 4. connected_tree
```json
{
  "tool": "connected_tree",
  "file": "src/main.py"  // Optional: specific file to analyze
}
```
- Shows import dependencies
- If no file specified, shows entire repository connections

## File Filtering

### Only These Extensions Are Included!
```python
file_extensions = ["py", "cfg"]  # ONLY Python and config files!
```

### Excluded Directories
```python
exclude_dirs = [
    # Testing
    "test", "tests", "testing", "test_", "_test",
    
    # Build/Cache
    "__pycache__", ".pytest_cache", ".tox", ".mypy_cache",
    "dist", "build", "target", "bin", "obj", "coverage",
    
    # Version Control & CI
    ".git", ".github", "ci", "cd", "github", "gitlab", "bitbucket",
    
    # Documentation
    "docs", "examples", "readme", "license", "changelog", "contributing",
    
    # Development
    "scripts", "tools", "deploy", "deployment", "docker", "kubernetes",
    
    # Environments
    "venv", "env", "node_modules",
    
    # Static/Data
    "static", "assets", "media", "uploads", "logs", "tmp", "temp", "cache",
    "data", "datasets", "notebooks", "jupyter", "ipynb_checkpoints",
    
    # Localization
    "locale", "translations", "i18n", "l10n",
    
    # Dependencies
    "vendor", "libs", "dependencies",
    
    # Settings
    "settings", "local_settings", "fixtures",
    
    # IMPORTANT - These were removed for good reasons:
    # "utils" - Too many legitimate utility files
    # "migrations" - Django migrations contain bug fixes
    # "lib" - matplotlib's main source directory!
    # "config", "conf" - Configuration files often have bugs
]
```

## Token Management

### Token Estimation (from `dialog_processor.py`)
```python
# Code: ~3.5 characters per token
# Text: ~4.5 characters per token
# Mixed: Weighted average based on code indicators
# Add 5% buffer for safety
```

### Context Length Calculation
```python
# Total available: max_seq_length (262144)
# Reserved for response: tokens_to_generate (81920)
# Usable for context: 262144 - 81920 = 180224 tokens
# With safety margin (0.9): ~162201 tokens
```

## Truncation Strategies

### 1. Sequential (Default in dialog_processor)
- Removes oldest assistant+tool output pairs
- Preserves initial problem statement (turn 0)
- Keeps conversation continuity

### 2. Bookend (Default in config)
- Keeps first turn (problem) + last 1-2 turns
- Removes all middle turns
- Best for large codebases

### 3. Smart Bookend
- Tries bookend first
- Falls back to keeping only last turn if still too long

### 4. Enhanced (Multi-level)
When enabled, tries in order:
1. Smart bookend
2. Aggressive bookend (minimal context)
3. Emergency (problem summary only)

## Advanced Features

### 1. Loop Detection
Detects when agent repeats identical tool calls:
- Threshold: 3 consecutive identical calls
- Injects intervention message to break loop
- Analyzes patterns for debugging

### 2. Implicit Tool Detection
When no explicit `<tool_call>` found:
- Checks after `</think>` tag for JSON
- Detects simple file paths: `"main.py"`
- Detects search patterns: `"search for X"`
- Filters common words before creating search

### 3. Final Turn Prompt
On the very last turn (step 20):
- Injects instruction to force location predictions
- Type "aligned" matches system prompt format
- Ensures agent provides its best guess

### 4. Enhanced Context Management
- Uses `tiktoken` for accurate token counting
- Proactive checking before LLM calls
- Multiple truncation fallbacks
- Safety margin prevents overflow

## Tool Call Formats

The dialog processor supports multiple formats:

### 1. Standard XML Format
```xml
<tool_call>
{
  "tool": "view_file",
  "path": "main.py"
}
</tool_call>
```

### 2. Unclosed Tag (Error Recovery)
```xml
<tool_call>
{
  "tool": "view_file",
  "path": "main.py"
```

### 3. Python Dict Format
```python
{'type': 'tool_calls', 'tool_call': {'tool': 'view_file', 'path': 'main.py'}}
```

### 4. Implicit After Think
```xml
</think>
{"path": "main.py", "view_range": [1, 50]}
```

### 5. Simple Requests
```xml
</think>
main.py
```
or
```xml
</think>
search for calculate_total
```

## Implementation Details

### Turn Structure
Each turn contains:
```python
{
    "inputs": str,         # User input or tool output
    "assistant": str,      # Model's response
    "tool_call": dict,     # Tool invocation (optional)
    "tool_output": str,    # Tool result (optional)
}
```

### Error Handling
- Missing files: Suggests similar filenames
- Invalid ranges: Shows entire file with explanation
- Empty searches: Returns "No results found"
- Context overflow: Progressive truncation

### Key Files and Their Roles
```
locagent.py                      # Main orchestration, config
dialog_processor.py              # LLM interaction, truncation
tool_executor.py                 # Tool implementation
utils.py                         # Repository filtering, tree generation
bookend_truncation.py            # Alternative truncation strategies
loop_detection.py                # Repetition prevention
enhanced_context_management.py   # Token counting, safety checks
final_turn_prompt.py             # Last turn instruction injection
```

## Common Pitfalls

1. **File Extensions**: Only `.py` and `.cfg` files are visible!
2. **Token Limits**: 262144 total, but only ~162k usable with safety
3. **View Limits**: Files truncated to 1000 lines
4. **Search Limits**: 3 snippets per file, top 5 files
5. **Truncation**: Middle turns are lost with bookend strategy

## Performance Tips

1. **Start with `repo_tree`**: Understand structure first
2. **Use specific paths**: Avoid searching if you know the file
3. **Search strategically**: Use unique identifiers
4. **View ranges**: Don't request entire large files
5. **Monitor tokens**: Watch for truncation warnings

## Debugging

Enable debug logging to see:
- Token counts: "Context check: X tokens, target: Y"
- Truncation: "Enhanced truncation stats: {...}"
- Tool execution: "Executing tool: X"
- Loop detection: "Loop detected! Agent has repeated..."

## Current Limitations

1. **File Types**: ONLY Python (.py) and config (.cfg) files
2. **No Summarization**: Feature disabled (enable_turn_summarization = False)
3. **Context Loss**: Bookend truncation loses middle turns
4. **Search Context**: Limited to 20 lines before/after matches
5. **View Size**: Maximum 1000 lines per view_file call
