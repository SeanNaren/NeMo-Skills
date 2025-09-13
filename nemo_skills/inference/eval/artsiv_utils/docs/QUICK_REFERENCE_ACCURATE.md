# Artsiv Quick Reference (Accurate)

## Real Configuration Values

```python
# Core
total_steps: int = 20                    # Max tool calls
file_extensions: list = ["py", "cfg"]    # ONLY Python & config files!

# Context (recently updated)
max_seq_length: int = 262144            # Max context tokens
tokens_to_generate: int = 81920         # Reserved for response
# Usable context: ~162k tokens with safety margin

# Truncation
truncation_strategy: str = "bookend"    # Default (not sequential!)

# Features (all enabled by default)
enable_loop_detection: bool = True
enable_enhanced_context: bool = True
enable_final_turn_prompt: bool = True
enable_turn_summarization: bool = False  # DISABLED
```

## Available Tools (Only 4!)

```python
# 1. View file
{"tool": "view_file", "path": "file.py", "view_range": [1, 50]}

# 2. Repository tree
{"tool": "repo_tree"}

# 3. Search code
{"tool": "codebase_search", "query": "search_term"}

# 4. Import connections
{"tool": "connected_tree", "file": "optional_file.py"}
```

## Token Calculation

```python
# Total: 262,144 tokens
# Reserved: 81,920 tokens  
# Available: 180,224 tokens
# With 90% safety: ~162,201 tokens usable
```

## File Filtering Reality

**ONLY THESE EXTENSIONS ARE INCLUDED:**
- `.py` - Python files
- `.cfg` - Config files

**That's it! No .js, .java, .cpp, etc.**

## Excluded Directories
```
test*, __pycache__, .git, docs, examples, scripts, venv, 
node_modules, dist, build, static, vendor, etc.

NOT excluded: utils, lib, migrations, config, conf
```

## Truncation Strategies

| Strategy | Reality |
|----------|---------|
| bookend | DEFAULT - Keeps first + last turns only |
| sequential | Removes oldest turns sequentially |
| smart_bookend | Bookend with fallback |
| enhanced | Multi-level with aggressive fallback |

## Tool Call Formats Supported

```xml
<!-- Standard -->
<tool_call>{"tool": "view_file", "path": "main.py"}</tool_call>

<!-- Implicit after think -->
</think>
{"path": "main.py"}

<!-- Simple -->
</think>
main.py
```

## Common Mistakes to Avoid

❌ Searching for .js files (not included!)
❌ Expecting summarization (it's disabled!)
❌ Viewing files > 1000 lines (truncated!)
❌ Using wrong tool names
❌ Forgetting only .py and .cfg exist

## Quick Debugging

```python
# Check logs for:
"Context check: X tokens, target: Y"
"Using truncation strategy: bookend"
"File extensions filter: ['py', 'cfg']"
"Loop detected! Agent has repeated..."
```

## Actual Limits

- Max file view: 1000 lines
- Search results: Top 5 files, 3 snippets each
- Context: ~162k tokens usable
- Tool calls: 20 maximum
- File types: 2 (py, cfg)

## Emergency Commands

```python
# When stuck in loop
# → Loop detection auto-intervenes after 3 repeats

# When context exceeded  
# → Auto-truncates with bookend strategy

# On last turn
# → Auto-injects location prediction prompt
```
