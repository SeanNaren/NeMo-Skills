# Enhanced Context Management for LocAgent

## Overview

The enhanced context management system addresses the 6% failure rate due to `context_length_exceeded` errors by providing:

1. **Accurate token counting** using tiktoken (when available)
2. **Proactive context checking** before LLM calls
3. **Safety margins** to prevent edge cases
4. **Aggressive truncation strategies** when needed

## Key Features

### 1. Accurate Token Counting

Instead of character-based estimation (e.g., 4 chars = 1 token), we use:
- **tiktoken** library for exact tokenization matching OpenAI models
- Improved heuristics that account for code, special characters, and formatting
- Proper handling of message structure overhead

### 2. Safety Margins

- Default 90% usage of max context (configurable)
- Prevents edge cases where estimation is slightly off
- Leaves room for unexpected token usage

### 3. Multi-Strategy Truncation

When context is too long, tries strategies in order:
1. **Smart Bookend** - Keep first and last turns with summaries
2. **Aggressive Bookend** - Minimal context preservation
3. **Emergency** - Absolute minimum (problem statement only)

### 4. Proactive Checks

- Checks context length BEFORE making LLM call
- Fails fast with clear error messages
- Logs detailed statistics for debugging

## Configuration

```yaml
# In your locagent config:
enable_enhanced_context: true  # Enable enhanced context management
context_safety_margin: 0.9     # Use 90% of max context
use_tiktoken: true            # Use tiktoken if available
truncation_strategy: "enhanced"  # Or use with other strategies
```

## How It Works

### Token Counting Flow

```python
1. Check if tiktoken is available
2. If yes: Use exact tokenization for the model
3. If no: Use improved character-based estimation
   - Code: ~3.0 chars/token
   - Text with special chars: ~3.2 chars/token  
   - Regular text: ~3.8 chars/token
4. Add message structure overhead (~15 tokens per turn)
```

### Truncation Flow

```python
1. Count current tokens accurately
2. Calculate safe target (max * safety_margin - tokens_to_generate)
3. If over limit:
   a. Try smart bookend truncation
   b. If still too long, try aggressive bookend
   c. If still too long, use emergency truncation
4. Log statistics for debugging
```

### Example Statistics

```
Context check: 28543 tokens, target: 24576 (safe max: 29491, reserve: 8192)
Using enhanced context management
Enhanced truncation stats: {
  'original_turns': 42,
  'original_tokens': 28543,
  'final_turns': 4,
  'final_tokens': 23456,
  'truncation_applied': True,
  'truncation_strategy': 'smart_bookend'
}
```

## Installation

### Optional: Install tiktoken for best accuracy

```bash
pip install tiktoken
```

The system works without tiktoken but is more accurate with it.

## Backwards Compatibility

The enhanced context management is:
- **Optional** - Can be disabled via config
- **Backwards compatible** - Falls back gracefully if dependencies missing
- **Non-breaking** - Existing truncation strategies still work

## Debugging Context Issues

### 1. Check Token Counts

Enable debug logging to see exact token counts:
```python
import logging
logging.getLogger('locagent').setLevel(logging.DEBUG)
```

### 2. Analyze Failures

Look for these patterns in logs:
- `Context length check failed` - Proactive check caught issue
- `Enhanced truncation stats` - Shows what happened during truncation
- `tiktoken not available` - Using estimation instead of exact counting

### 3. Tune Parameters

If still getting failures:
- Reduce `context_safety_margin` (e.g., 0.85 for 85%)
- Increase `tokens_to_generate` if responses are being cut off
- Use more aggressive `truncation_strategy`

## Performance Impact

- **With tiktoken**: ~10-20ms overhead per token count
- **Without tiktoken**: Negligible overhead (<1ms)
- **Truncation**: Only when needed, typically <100ms

The overhead is minimal compared to LLM inference time.

## Future Improvements

1. **Dynamic safety margins** based on content type
2. **Smarter truncation** that preserves key information
3. **Token count caching** for repeated content
4. **Model-specific optimizations**

## Known Issues and Fixes

### Aggressive Truncation KeyError (Fixed)

**Issue**: When aggressive truncation was applied, it could cause `KeyError: 'assistant'` during prompt filling.

**Fix**: Modified `_aggressive_bookend_truncate` to preserve the required turn structure, including the 'assistant' field when it exists in the original turn.

**Details**: See `AGGRESSIVE_TRUNCATION_FIX.md` for full documentation of this fix.
