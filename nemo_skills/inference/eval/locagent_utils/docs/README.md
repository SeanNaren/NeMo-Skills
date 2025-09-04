# LocAgent Utilities Documentation

This directory contains all functionality-related documentation for the LocAgent utilities.

## 📖 Start Here
- **LOCAGENT_ACCURATE_GUIDE.md** - Accurate technical guide based on actual code implementation
- **QUICK_REFERENCE_ACCURATE.md** - Quick reference with real configuration values and limits

## Core Functionality Documents

### Summarization
- **SUMMARIZATION_ISSUE_ANALYSIS.md** - Analysis of the summarization loop issue that caused agent to get stuck
- **SUMMARIZATION_TODO.md** - Future plans for implementing proper summarization
- **SUMMARIZATION_CURRENT_STATE.md** - Current state of summarization implementation
- **FAILED_SUMMARIZATION_ANALYSIS.md** - Analysis of why summarization model loading failed

### Truncation Strategies
- **BOOKEND_TRUNCATION_TESTING.md** - Testing documentation for bookend truncation strategy
- **QUICK_TEST_BOOKEND.md** - Quick testing guide for bookend truncation

### Loop Detection & Prevention
- **LOOP_DETECTION_ANALYSIS.md** - Analysis of repetitive agent behavior patterns
- **LOOP_DETECTION_IMPLEMENTATION.md** - Implementation details of loop detection system
- **LOOP_DETECTION_SOLUTION_SUMMARY.md** - Summary of the loop detection solution

### Context Length Management
- **ENHANCED_CONTEXT_MANAGEMENT.md** - Enhanced context management with accurate token counting
- **CONTEXT_LENGTH_SOLUTION_SUMMARY.md** - Summary of context length management solution

### Final Turn Prompt Injection
- **FINAL_TURN_PROMPT_FEATURE.md** - Feature documentation for final turn prompt injection
- **FINAL_TURN_PROMPT_SUMMARY.md** - Summary of final turn prompt functionality
- **FINAL_TURN_PROMPT_ALIGNMENT.md** - Alignment with system prompt format

### Bug Fixes & Improvements
- **DEV_NULL_BUG_FIX.md** - Documentation of the /dev/null bug fix in patch parsing
- **AGGRESSIVE_TRUNCATION_FIX.md** - Fix for KeyError when aggressive truncation is applied

### Testing & Compatibility
- **TESTING_GUIDE_BACKWARDS_COMPATIBLE.md** - Guide for testing with backwards compatibility

## Feature Status

| Feature | Status | Implementation File | Documentation |
|---------|--------|-------------------|---------------|
| Summarization | DISABLED | `locagent_summarization.py` | SUMMARIZATION_*.md |
| Bookend Truncation | Active (DEFAULT) | `bookend_truncation.py` | BOOKEND_*.md |
| Loop Detection | Active | `loop_detection.py` | LOOP_DETECTION_*.md |
| Enhanced Context Management | Active | `enhanced_context_management.py` | ENHANCED_CONTEXT_*.md |
| Final Turn Prompt | Active | `final_turn_prompt.py` | FINAL_TURN_PROMPT_*.md |
| Utility Functions | Active | `utils.py` | DEV_NULL_BUG_FIX.md |

## ⚠️ Critical Information

- **File Extensions**: ONLY `.py` and `.cfg` files are included in the repository view!
- **Context Limits**: 262,144 total tokens, ~162k usable with safety margin
- **Default Truncation**: `bookend` strategy (keeps first + last turns only)
- **Summarization**: Currently DISABLED (`enable_turn_summarization = False`)
