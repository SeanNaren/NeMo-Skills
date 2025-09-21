# Final Turn Prompt - System Prompt Alignment

## Overview

The final turn prompts have been updated to precisely match the format and style of the main system prompt for maximum consistency and effectiveness.

## Key Alignments

### 1. Format Consistency

**Main System Prompt Style:**
- Uses `**bold text**` for emphasis
- Clear XML formatting examples with proper indentation
- Strict rules about tag structure
- Emphasis on `<think>` followed by `<tool_call>` OR `<locations>`

**Final Turn Prompt (Aligned Type):**
```
**Interaction protocol update**
You have reached the maximum number of tool calls. You **must** now reply with a `<locations>` block.

**Strict rules for this final response:**
- Assistant message must follow the EXACT structure: `<think>...</think>` followed by `<locations>...</locations>`
- **DO NOT** issue any more `<tool_call>` blocks
- **DO NOT** include any text outside the required tags
```

### 2. Language and Tone

Both use:
- Direct, imperative language ("You **must**", "**DO NOT**")
- Technical precision about XML tag formatting
- Clear examples with proper syntax highlighting
- Emphasis on exhaustive location finding

### 3. Structure Requirements

The aligned prompt maintains the same strict structure:
```xml
<think>
Based on my investigation, I have identified the following locations that need editing...
</think>

<locations>
path/to/file.py:L<start>-L<end>
another/file.rs:L<start>-L<end>
</locations>
```

### 4. Core Philosophy

Both prompts emphasize:
- **Be exhaustive**: "emit **every** file and line range that needs editing"
- **Over-include rather than miss**: "It's better to over-inspect and over-include"
- **No code output**: Focus solely on locations
- **Strict formatting**: Exact tag structure required

## Instruction Type Comparison

### "aligned" (Default - Recommended)
- Mirrors system prompt language exactly
- Uses same formatting conventions
- Maintains protocol consistency

### "standard"
- More detailed with numbered steps
- Still uses consistent formatting
- Good for models that need more structure

### "urgent"
- Shorter, more forceful
- For situations needing immediate response
- Maintains format requirements

### "gentle"
- Softer tone while maintaining format
- For models that respond better to encouragement

### "detailed"
- Most comprehensive with examples
- Step-by-step breakdown
- For complex investigations

## Configuration

```yaml
# Recommended settings for best alignment:
enable_final_turn_prompt: true
final_turn_instruction_type: "aligned"  # Matches system prompt style
final_turn_threshold: 1.0               # Only on final turn
```

## Benefits of Alignment

1. **Consistency** - Model receives familiar formatting throughout
2. **Clarity** - No confusion about expected output format
3. **Effectiveness** - Leverages the model's training on the system prompt style
4. **Reduced Errors** - Less chance of format mistakes

## Example Effect

When the model reaches turn 20/20, it sees:

```
The session file shows timeout handling.

**Interaction protocol update**
You have reached the maximum number of tool calls. You **must** now reply with a `<locations>` block.

**Strict rules for this final response:**
- Assistant message must follow the EXACT structure: `<think>...</think>` followed by `<locations>...</locations>`
- **DO NOT** issue any more `<tool_call>` blocks
- **DO NOT** include any text outside the required tags
- Based on all code you have inspected, emit **every** file and line range that needs editing
- If uncertain about exact lines, include your best assessment

**Output format:**
```xml
<think>
Based on my investigation, I have identified the following locations that need editing...
</think>

<locations>
path/to/file.py:L<start>-L<end>
another/file.rs:L<start>-L<end>
</locations>
```

Remember: It's better to over-inspect and over-include than to miss a required edit location.
```

This maintains perfect consistency with the original system prompt's style and requirements.
