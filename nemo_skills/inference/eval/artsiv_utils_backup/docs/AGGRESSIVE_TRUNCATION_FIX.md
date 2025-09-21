# Aggressive Truncation KeyError Fix

## Issue Description

When the enhanced context management system applied aggressive truncation due to context length exceeding, it was causing a `KeyError: 'assistant'` during prompt filling. This error occurred specifically when:

1. A tool output exceeded the context length limit (e.g., viewing a large file)
2. The system applied aggressive_bookend truncation to reduce tokens
3. The truncated turns were missing required fields that the prompt system expected

### Error Pattern

```
Context check: 28052 tokens, target: 21299 (safe max: 29491, reserve: 8192)
WARNING  Context length exceeded: 28052 > 21299
INFO  Trying smart_bookend truncation strategy
WARNING  smart_bookend still too long: 28052 tokens
INFO  Trying aggressive_bookend truncation strategy
INFO  aggressive_bookend succeeded: 490 tokens
...
ERROR  Unexpected error in process_single_datapoint: 'assistant'
KeyError: 'assistant'
```

## Root Cause

The `_aggressive_bookend_truncate` function in `enhanced_context_management.py` was creating minimal turns to save tokens, but it was not preserving the required structure that the prompt filling system expected. Specifically:

1. It was creating turns with only the 'inputs' field
2. If the original turn had an 'assistant' field, it was being dropped
3. The prompt filler (`prompt/utils.py` line 293) expected the 'assistant' field to exist when present in the original

## Solution

Modified the `_aggressive_bookend_truncate` function to:

1. Preserve the turn structure by keeping all essential fields that exist in the original
2. Always include the 'assistant' field if it existed in the original turn (even if truncated or empty)
3. Also preserve tool-related fields ('tool_call', 'tool_output') when they exist

### Code Changes

```python
# Before (causing KeyError)
first_turn = {'inputs': first_turn.get('inputs', '')}

# After (preserving structure)
minimal_first_turn = {'inputs': first_turn.get('inputs', '')}
if 'assistant' in first_turn:
    minimal_first_turn['assistant'] = first_turn.get('assistant', '')[:200]
if 'tool_call' in first_turn:
    minimal_first_turn['tool_call'] = first_turn['tool_call']
if 'tool_output' in first_turn:
    minimal_first_turn['tool_output'] = first_turn['tool_output'][:200] + "... [output truncated]"
```

## Testing

Created `test_truncation_fix.py` to verify:
1. The truncation preserves required fields
2. No KeyError occurs during prompt filling
3. Various turn structures are handled correctly

Test results show the fix is working correctly:
- ✓ 'assistant' field preserved when it exists
- ✓ No KeyError would occur in prompt filling
- ✓ Tool-related fields are preserved

## Impact

This fix ensures that even under aggressive truncation scenarios:
- The agent can continue processing without crashing
- Context is preserved as much as possible while staying within limits
- The prompt system receives the expected data structure

## Related Files

- `/mnt/ssd/htamoyan/NeMo-Skills/nemo_skills/inference/eval/artsiv_utils/enhanced_context_management.py` - Contains the fix
- `/mnt/ssd/htamoyan/NeMo-Skills/nemo_skills/inference/eval/artsiv_utils/test_truncation_fix.py` - Test script
- `/mnt/ssd/htamoyan/NeMo-Skills/nemo_skills/prompt/utils.py` - Where the error was occurring
