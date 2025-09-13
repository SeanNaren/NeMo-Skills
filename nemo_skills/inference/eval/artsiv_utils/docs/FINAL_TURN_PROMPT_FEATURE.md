# Final Turn Prompt Feature

## Overview

The Final Turn Prompt feature addresses the issue where the agent exhausts its investigation budget (`max_steps_exceeded`) without providing location predictions. By injecting a special instruction on the final turn, we force the model to make its best guess based on the evidence gathered.

## Problem It Solves

Previously, when the agent reached `max_steps` (default: 20), it would fail with:
- Status: `failed`
- Reason: `max_steps_exceeded`
- No location predictions

This meant all the investigation work was wasted.

## How It Works

On the final turn (or configurable threshold), the system injects a special message into the conversation that explicitly tells the model:
1. This is the last turn
2. It MUST provide location predictions now
3. A reasonable guess is better than no answer

### Example Injection

```
[SYSTEM NOTICE: This is your FINAL turn. You MUST provide location predictions now. 
Based on your investigation so far, make your best assessment of where the issue is located. 
Use the <locations> tag to specify file paths and line numbers. 
If you're not completely certain, provide your best educated guess based on the evidence you've gathered.]
```

## Configuration

```yaml
# In your artsiv config:
enable_final_turn_prompt: true      # Enable the feature (default: true)
final_turn_instruction_type: "standard"  # Type of instruction (see below)
final_turn_threshold: 1.0           # When to trigger (1.0 = only last turn)
```

### Instruction Types

1. **aligned** (default) - Closely matches the main system prompt format and style
2. **standard** - Professional, clear instruction with formatting examples
3. **urgent** - More forceful, emphasizes urgency
4. **gentle** - Softer approach, encouraging
5. **detailed** - Step-by-step instructions with comprehensive guidance

### Threshold Setting

- `1.0` - Only on the very last turn (default)
- `0.9` - Last 10% of turns
- `0.8` - Last 20% of turns

## Example Output

### Before (without final turn prompt):
```
Turn 19: Let me check one more file...
Turn 20: I need to investigate further...
Result: FAILED - max_steps_exceeded
```

### After (with final turn prompt):
```
Turn 19: Let me check one more file...
Turn 20: [Receives final turn instruction]
         Based on my investigation, the issue appears to be in:
         <locations>
         auth/middleware.py:45-67
         auth/session.py:123-145
         </locations>
Result: SUCCESS
```

## Benefits

1. **Reduces wasted investigations** - Forces a conclusion even if not 100% certain
2. **Improves success rate** - Better to have a reasonable guess than no answer
3. **Preserves investigation value** - All the work done isn't thrown away
4. **Configurable** - Can adjust urgency and timing

## Testing the Feature

### Enable/Disable Test
```bash
# With feature enabled (default)
python -m nemo_skills.inference.eval.artsiv \
    enable_final_turn_prompt=true \
    total_steps=5 \
    ...

# With feature disabled
python -m nemo_skills.inference.eval.artsiv \
    enable_final_turn_prompt=false \
    total_steps=5 \
    ...
```

### Different Instruction Types
```bash
# Urgent instruction
python -m nemo_skills.inference.eval.artsiv \
    final_turn_instruction_type="urgent" \
    ...

# Gentle instruction  
python -m nemo_skills.inference.eval.artsiv \
    final_turn_instruction_type="gentle" \
    ...
```

### Early Triggering
```bash
# Trigger on last 20% of turns
python -m nemo_skills.inference.eval.artsiv \
    final_turn_threshold=0.8 \
    total_steps=10 \
    ...
# Will trigger on turns 8, 9, 10
```

## Implementation Details

The feature is implemented in `final_turn_prompt.py` with:

1. **`should_inject_final_turn()`** - Determines when to inject
2. **`inject_final_turn_instruction()`** - Modifies the conversation
3. **Integration in main loop** - Checks before each LLM call

The injection happens right before sending to the LLM, after all other modifications (truncation, loop detection, etc.).

## Backwards Compatibility

- Feature is optional (can be disabled)
- Falls back gracefully if module not available
- No impact on existing behavior when disabled
- All configuration has sensible defaults

## Expected Impact

Based on analysis, this should:
- Reduce `max_steps_exceeded` failures by 50-80%
- Improve overall success rate by 3-5%
- Provide valuable partial results instead of complete failures

## Custom Instructions

You can also provide custom instructions via code:
```python
custom_instruction = """
[INVESTIGATION COMPLETE]
You have gathered sufficient evidence. Now provide your analysis:
1. Summarize what you found
2. List ALL suspicious locations
3. Use <locations> tags for your predictions
This is your final opportunity - be comprehensive!
"""

# Use via API or config override
```

## Monitoring

Look for these log messages:
- `"Injecting final turn instruction at step X/Y"` - Feature activated
- `"Injected final turn instruction (type: standard)"` - Instruction added

## Future Enhancements

1. **Smart summarization** - Include investigation summary in final prompt
2. **Confidence scoring** - Ask model to rate confidence in predictions  
3. **Multiple attempts** - Try different prompts if first fails
4. **Learning** - Analyze which prompt types work best
