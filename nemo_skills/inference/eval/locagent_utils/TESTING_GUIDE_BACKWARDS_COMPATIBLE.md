# Testing Guide: Backwards Compatible Truncation & Summarization

## Overview

The locagent now supports multiple truncation strategies while maintaining full backwards compatibility. By default, it uses the existing sequential truncation.

## Configuration Options

### Truncation Strategies

```yaml
truncation_strategy: "sequential"  # Default - existing behavior
# Other options:
# - "bookend" - Keep only first and last turns
# - "smart_bookend" - Bookend with fallback logic
```

### Summarization Settings (Future Use)

```yaml
enable_turn_summarization: false  # Default - disabled
max_summary_sentences: 5
min_turns_for_summarization: 10
summarization_model: false
```

## Testing Commands

### 1. Baseline Test (Current Default Behavior)

This uses the existing sequential truncation - exactly as before:

```bash
python -m nemo_skills.inference.eval.locagent \
    inference.max_tokens=8192 \
    total_steps=20 \
    max_seq_length=32768 \
    tokens_to_generate=8192
```

### 2. Test Bookend Truncation

To test if bookend truncation reproduces the good results from failed summarization:

```bash
python -m nemo_skills.inference.eval.locagent \
    inference.max_tokens=8192 \
    total_steps=20 \
    max_seq_length=32768 \
    tokens_to_generate=8192 \
    truncation_strategy="bookend"
```

### 3. Test Smart Bookend (Adaptive)

This tries bookend first, then falls back if needed:

```bash
python -m nemo_skills.inference.eval.locagent \
    inference.max_tokens=8192 \
    total_steps=20 \
    max_seq_length=32768 \
    tokens_to_generate=8192 \
    truncation_strategy="smart_bookend"
```

### 4. Compare Different Context Lengths

Test how strategies perform with different context limits:

```bash
# Small context (forces more truncation)
python -m nemo_skills.inference.eval.locagent \
    inference.max_tokens=4096 \
    total_steps=20 \
    max_seq_length=16384 \
    tokens_to_generate=4096 \
    truncation_strategy="bookend"

# Large context (less truncation needed)
python -m nemo_skills.inference.eval.locagent \
    inference.max_tokens=8192 \
    total_steps=20 \
    max_seq_length=65536 \
    tokens_to_generate=8192 \
    truncation_strategy="sequential"
```

## What to Monitor

### Log Messages

You'll see informative messages about the truncation strategy:

```
INFO: Using truncation strategy: bookend
DEBUG: Using bookend truncation strategy
INFO: Truncated dialogue from 15 to 3 turns using bookend strategy
```

If bookend module is missing:
```
WARNING: Bookend truncation module not available. Falling back to sequential truncation.
```

### Performance Metrics

Compare these across strategies:
1. **Success rate** - Does bookend improve completion?
2. **Average turns to solution** - Are solutions found faster?
3. **Token efficiency** - How much context is used?
4. **Failure reasons** - What causes failures?

## Expected Behavior by Strategy

### Sequential (Default)
- Removes oldest turns first
- Keeps partial assistant responses (200-300 chars)
- Gradual context reduction
- May keep confusing partial context

### Bookend
- Keeps turn 0 (problem statement)
- Keeps last 1-2 turns only
- Removes ALL middle turns
- Clean "problem → current state" view

### Smart Bookend
- Tries bookend first
- Falls back to keeping only recent turns if needed
- More adaptive to edge cases

## Migration Path

1. **Phase 1**: Test with current code (sequential)
2. **Phase 2**: A/B test bookend vs sequential
3. **Phase 3**: If bookend performs better, make it default
4. **Phase 4**: Re-enable and fix summarization later

## Rollback Plan

If any issues arise, simply remove or don't specify `truncation_strategy`:

```bash
# This reverts to exact previous behavior
python -m nemo_skills.inference.eval.locagent \
    inference.max_tokens=8192 \
    total_steps=20
```

## Advanced Debugging

Enable detailed logging to see truncation decisions:

```python
import logging

# In your script before running
logging.getLogger('nemo_skills.inference.eval.locagent').setLevel(logging.DEBUG)
logging.getLogger('nemo_skills.inference.eval.locagent_utils.bookend_truncation').setLevel(logging.DEBUG)
```

## Future: Re-enabling Summarization

When ready to test summarization (after fixing the loop issues):

```bash
python -m nemo_skills.inference.eval.locagent \
    inference.max_tokens=8192 \
    total_steps=20 \
    enable_turn_summarization=true \
    min_turns_for_summarization=10 \
    summarization_model=true
```

Note: This is currently disabled to prevent the loop issues discovered earlier.
