# Quick One-Line Test for Bookend Truncation

If you want to quickly test the bookend truncation theory without importing new modules, you can make this simple change in `artsiv.py`:

## Find this code (around line 394):
```python
data_point['turns'] = truncate_dialogue_history(
    data_point['turns'], self.cfg.max_seq_length, self.cfg.tokens_to_generate
)
```

## Replace with this hack that mimics bookend truncation:
```python
# QUICK TEST: Bookend truncation - keep only first and last turns
if len(data_point['turns']) > 3:
    last_turn = data_point['turns'][-1]
    second_last = data_point['turns'][-2] if last_turn.get('tool_output') else None
    data_point['turns'] = [data_point['turns'][0]] + ([second_last] if second_last else []) + [last_turn]
    LOG.info(f"BOOKEND TEST: Reduced to {len(data_point['turns'])} turns (first + last)")
else:
    data_point['turns'] = truncate_dialogue_history(
        data_point['turns'], self.cfg.max_seq_length, self.cfg.tokens_to_generate
    )
```

This quick hack will:
- For conversations > 3 turns: Keep only first + last 1-2 turns
- For short conversations: Use normal truncation

If this improves your scores, then the bookend truncation theory is correct!
