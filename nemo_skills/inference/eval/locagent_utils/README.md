# LocAgent Utilities

This directory contains all utility modules and documentation for the LocAgent code generation system.

## Structure

### Core Modules
- `utils.py` - General utility functions (repo filtering, tree generation, patch parsing)
- `bookend_truncation.py` - Alternative truncation strategies for dialogue history
- `locagent_summarization.py` - Summarization code (currently disabled, preserved for future use)
- `loop_detection.py` - Detection and prevention of repetitive agent behavior
- `enhanced_context_management.py` - Advanced context length management with accurate token counting
- `final_turn_prompt.py` - Final turn prompt injection to ensure location predictions

### Version-Specific Modules
- `v4/` - Current production version
  - `dialog_processor.py` - Handles LLM response parsing and dialogue management
  - `tool_executor.py` - Executes tool calls (view_file, codebase_search, etc.)

### Documentation
All functionality-related documentation has been organized in the `docs/` subdirectory:
- `docs/README.md` - Overview of all documentation files
- `docs/SUMMARIZATION_*.md` - Summarization-related documentation
- `docs/BOOKEND_*.md` - Bookend truncation documentation
- `docs/LOOP_DETECTION_*.md` - Loop detection system documentation
- `docs/ENHANCED_CONTEXT_*.md` - Context management documentation
- `docs/FINAL_TURN_PROMPT_*.md` - Final turn prompt feature documentation
- `docs/DEV_NULL_BUG_FIX.md` - Bug fix documentation
- `docs/TESTING_GUIDE_BACKWARDS_COMPATIBLE.md` - Testing guide

## Import Paths

All modules should be imported with the full path:
```python
from nemo_skills.inference.eval.locagent_utils.utils import filter_repo_dict, tree_repo_dict
from nemo_skills.inference.eval.locagent_utils.bookend_truncation import bookend_truncate_dialogue_history
from nemo_skills.inference.eval.locagent_utils.loop_detection import detect_repetitive_tool_calls
from nemo_skills.inference.eval.locagent_utils.enhanced_context_management import TokenCounter
from nemo_skills.inference.eval.locagent_utils.final_turn_prompt import inject_final_turn_instruction
from nemo_skills.inference.eval.locagent_utils.v4.dialog_processor import DialogProcessor
from nemo_skills.inference.eval.locagent_utils.v4.tool_executor import ToolExecutor
```

## Version Management

The main `locagent.py` uses the `PROMPT_TEMPLATE_VERSION` variable to select which version to use:
```python
PROMPT_TEMPLATE_VERSION: str = "v4"
```

This allows easy switching between different implementations while maintaining backwards compatibility.
