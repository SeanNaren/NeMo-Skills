# LocAgent Utilities

This directory contains all utility modules and documentation for the LocAgent code generation system.

## Structure

### Core Modules
- `utils.py` - General utility functions (repo filtering, tree generation, patch parsing)
- `bookend_truncation.py` - Alternative truncation strategies for dialogue history
- `locagent_summarization.py` - Summarization code (currently disabled, preserved for future use)

### Version-Specific Modules
- `v4/` - Current production version
  - `dialog_processor.py` - Handles LLM response parsing and dialogue management
  - `tool_executor.py` - Executes tool calls (view_file, codebase_search, etc.)

### Documentation
- `TESTING_GUIDE_BACKWARDS_COMPATIBLE.md` - How to test different truncation strategies
- `BOOKEND_TRUNCATION_TESTING.md` - Specific guide for bookend truncation testing
- `FAILED_SUMMARIZATION_ANALYSIS.md` - Analysis of why failed summarization led to better scores
- `SUMMARIZATION_ISSUE_ANALYSIS.md` - Detailed analysis of summarization problems
- `SUMMARIZATION_TODO.md` - TODO list for fixing summarization
- `DEV_NULL_BUG_FIX.md` - Documentation of the /dev/null bug fix
- `QUICK_TEST_BOOKEND.md` - Quick test instructions for bookend truncation

## Import Paths

All modules should be imported with the full path:
```python
from nemo_skills.inference.eval.locagent_utils.utils import filter_repo_dict, tree_repo_dict
from nemo_skills.inference.eval.locagent_utils.bookend_truncation import bookend_truncate_dialogue_history
from nemo_skills.inference.eval.locagent_utils.v4.dialog_processor import DialogProcessor
from nemo_skills.inference.eval.locagent_utils.v4.tool_executor import ToolExecutor
```

## Version Management

The main `locagent.py` uses the `PROMPT_TEMPLATE_VERSION` variable to select which version to use:
```python
PROMPT_TEMPLATE_VERSION: str = "v4"
```

This allows easy switching between different implementations while maintaining backwards compatibility.
