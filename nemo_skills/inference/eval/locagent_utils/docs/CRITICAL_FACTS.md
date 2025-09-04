# Critical Facts About LocAgent

## 🚨 Most Important Facts

### 1. File Extensions - ONLY 2 TYPES!
```python
file_extensions = ["py", "cfg"]  # That's it!
```
- ❌ NO JavaScript (.js)
- ❌ NO Java (.java)  
- ❌ NO C++ (.cpp, .h)
- ❌ NO TypeScript (.ts)
- ❌ NO Go, Rust, Ruby, etc.
- ✅ ONLY Python (.py)
- ✅ ONLY Config (.cfg)

### 2. Actual Context Limits
```python
max_seq_length = 262,144      # NOT 32,768!
tokens_to_generate = 81,920   # NOT 8,192!
```

### 3. Default Truncation Strategy
```python
truncation_strategy = "bookend"  # NOT "sequential"!
```
This means by default, middle turns are REMOVED when context is exceeded.

### 4. Summarization is DISABLED
```python
enable_turn_summarization = False  # NOT active!
```
All the summarization code exists but is NOT being used.

### 5. Only 4 Tools Available
1. `view_file` - View file contents
2. `repo_tree` - Show repository structure  
3. `codebase_search` - Search for text
4. `connected_tree` - Show import dependencies

That's it! No other tools exist.

### 6. View File Limits
- Maximum 1000 lines shown per call
- Files are truncated with warning if longer

### 7. Search Limits
- Shows top 5 files by match count
- Maximum 3 snippets per file
- 20 lines of context before/after match

### 8. Repository Filtering
These directories are EXCLUDED:
- test, tests, __pycache__, .git, docs, examples, scripts, venv, etc.

These are NOT excluded (important!):
- `utils` - Contains legitimate utility files
- `lib` - Main source directories
- `migrations` - Often contain bug fixes
- `config`, `conf` - Configuration files with bugs

## Why This Matters

1. **Performance**: Agent can only see Python and config files
2. **Context**: Much larger context window than you might expect
3. **Truncation**: You lose middle conversation with bookend
4. **No Summarization**: Don't expect conversation compression
5. **Limited Tools**: Only 4 tools, not dozens

## Common Misconceptions

❌ "Agent can analyze JavaScript codebases" - NO, only Python!
❌ "Summarization reduces token usage" - NO, it's disabled!
❌ "Sequential truncation is default" - NO, bookend is!
❌ "Small context window" - NO, it's 262k tokens!
❌ "Many file types supported" - NO, only .py and .cfg!

## Quick Test

Ask the agent to:
1. Search for a .js file → Should find nothing
2. View a file > 1000 lines → Should see truncation
3. Use repo_tree → Should only see .py and .cfg files
4. Check turns after truncation → Middle turns gone with bookend
