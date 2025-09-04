# Loop Detection Implementation Guide

## Overview

We've implemented a comprehensive loop detection and prevention system to stop the agent from getting stuck in repetitive patterns (like generating the same tool call 18 times in a row).

## Features

### 1. Loop Detection
- Monitors generated tool calls for repetitive patterns
- Triggers when the same tool call appears 3+ times (configurable)
- Analyzes recent vs total repetitions to detect active loops

### 2. Loop Prevention
- **Pre-generation**: Detects potential loops before LLM call
- **Post-generation**: Identifies loops after generation
- Injects intervention messages to guide the agent

### 3. Intervention System
- Adds system messages explaining the loop
- Provides specific guidance based on the repeated tool
- Forces the agent to try alternative approaches

## Configuration

```yaml
# In your locagent config:
enable_loop_detection: true  # Enable/disable loop detection (default: true)
loop_detection_threshold: 3  # Number of repetitions to trigger detection (default: 3)
```

## How It Works

### Detection Algorithm
1. Extracts tool calls from generation history
2. Normalizes JSON to handle formatting differences
3. Counts occurrences of each unique tool call
4. Checks if recent calls (last 10) are >70% identical

### Prevention Flow
```
Before LLM Call:
1. Check if we have enough history (threshold - 1 calls)
2. Detect if previous calls show a loop pattern
3. If loop detected, inject intervention message

After LLM Call:
1. Add generation to history
2. Check for loops (threshold reached)
3. If loop detected:
   - Log warning with details
   - Inject intervention for next turn
   - Add metadata to generation
```

### Intervention Messages

For file viewing loops:
```
SYSTEM INTERVENTION: Loop detected! You have attempted to view 'file.py' 5 times with the same parameters.

The file appears to be too large or the output is being truncated. Please try a different approach:
1. View a specific section using line numbers
2. Search for specific content using grep or find
3. Look at the file structure first
4. Check if there's a more specific file

DO NOT repeat the same view_file command.
```

## Example Output

When a loop is detected, you'll see:
```json
{
  "_loop_detected": true,
  "_loop_info": {
    "repeated_call": "{\"view_file\": {\"path\": \"file.py\", \"view_range\": [1, -1]}}",
    "total_repetitions": 5,
    "recent_repetitions": 5,
    "loop_percentage": 100.0
  },
  "_intervention_added": true
}
```

## Benefits

1. **Prevents Infinite Loops**: Stops agent from wasting tokens
2. **Improves Success Rate**: Forces exploration of alternatives
3. **Better Debugging**: Clear visibility into what went wrong
4. **Graceful Recovery**: Guides agent to productive paths

## Testing

To test loop detection:
1. Set a low threshold: `loop_detection_threshold: 2`
2. Give the agent a task with a large file
3. Watch for intervention messages in the output

## Backwards Compatibility

- Feature is optional (can be disabled)
- Uses try/except imports
- Defaults to no-op if not available
- No impact on existing functionality

## Future Improvements

1. **Smart Context Preservation**: Keep diverse attempts in context
2. **Tool-Specific Interventions**: Custom guidance per tool type
3. **Learning from Loops**: Track common loop patterns
4. **Automatic Parameter Adjustment**: Suggest specific ranges/parameters
