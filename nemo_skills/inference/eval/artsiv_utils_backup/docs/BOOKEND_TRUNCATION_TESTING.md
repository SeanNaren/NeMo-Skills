# Testing Bookend Truncation Strategy

## Quick Test

To test if the bookend truncation strategy reproduces the good results from failed summarization:

### 1. Import the New Strategy

In `artsiv.py`, add:
```python
from nemo_skills.inference.eval.artsiv_utils.bookend_truncation import bookend_truncate_dialogue_history
```

### 2. Replace Truncation Method

Replace line ~394:
```python
# OLD:
data_point['turns'] = truncate_dialogue_history(
    data_point['turns'], self.cfg.max_seq_length, self.cfg.tokens_to_generate
)

# NEW:
data_point['turns'] = bookend_truncate_dialogue_history(
    data_point['turns'], self.cfg.max_seq_length, self.cfg.tokens_to_generate
)
```

### 3. Run Evaluation

Test with the same parameters that previously gave good results:
```bash
python -m nemo_skills.inference.eval.artsiv \
    inference.max_tokens=8192 \
    total_steps=20 \
    max_seq_length=32768 \
    tokens_to_generate=8192
```

## What to Expect

The bookend truncation will:
1. Keep the problem statement (turn 0)
2. Keep only the last 1-2 turns
3. Remove ALL middle turns when context is exceeded

This should replicate the behavior when summarization failed, which gave the best scores.

## Comparison Test

For a proper comparison, run three versions:

### Version A: Current Sequential Truncation (baseline)
```python
# Use existing truncate_dialogue_history
```

### Version B: Bookend Truncation
```python
# Use bookend_truncate_dialogue_history
```

### Version C: No Truncation (if context allows)
```python
# Comment out truncation entirely for short conversations
```

## Metrics to Track

1. **Success Rate**: Does bookend truncation improve task completion?
2. **Turn Efficiency**: Do models need fewer turns to solve problems?
3. **Token Usage**: How much context is actually being used?
4. **Failure Patterns**: What types of failures occur with each strategy?

## Expected Results

Based on the analysis, bookend truncation should:
- ✅ Improve success rate (like the failed summarization case)
- ✅ Reduce confusion from partial middle context
- ✅ Provide cleaner problem→solution reasoning
- ⚠️ May lose some useful middle context in complex problems

## Advanced Testing

If bookend works well, try the hybrid approach:
```python
# Use smart_bookend_truncate for adaptive behavior
from nemo_skills.inference.eval.artsiv_utils.bookend_truncation import smart_bookend_truncate

data_point['turns'] = smart_bookend_truncate(
    data_point['turns'], self.cfg.max_seq_length, self.cfg.tokens_to_generate
)
```

This will:
1. Try bookend truncation first
2. Fall back to keeping only recent turns if needed
3. Handle edge cases better

## Debugging

Enable detailed logging to see what's happening:
```python
import logging
logging.getLogger('nemo_skills.inference.eval.artsiv_utils.bookend_truncation').setLevel(logging.DEBUG)
```

This will show:
- Which turns are kept/removed
- Token counts before/after
- Why certain decisions are made
