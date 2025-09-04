# Summarization Fix TODO List

## Immediate Fixes Needed

- [ ] Fix turn structure preservation - summaries should augment, not replace
- [ ] Implement proper state tracking that survives summarization  
- [ ] Fix the loop issue where agent repeats the same tool call
- [ ] Ensure tool_call and tool_output association is maintained
- [ ] Add better logging to track when/why summarization happens

## Code Changes Required

### 1. In dialog_processor module:
- [ ] Rewrite `apply_context_summarization` to preserve turn structure
- [ ] Add turn validation after summarization
- [ ] Implement `ConversationState` class for state tracking
- [ ] Add unit tests for all summarization functions

### 2. In locagent.py:
- [ ] Change summarization trigger logic (not just turn count)
- [ ] Add state validation before/after summarization
- [ ] Implement selective summarization (only completed chunks)
- [ ] Add recovery mechanism if summarization fails

### 3. Testing:
- [ ] Create test cases with conversations that trigger summarization
- [ ] Verify agent can continue after summarization
- [ ] Test with different context lengths
- [ ] Validate token reduction actually happens

## Recommended Implementation Order

1. **Phase 1: Fix Core Issues**
   - Fix turn structure preservation
   - Add state tracking
   - Fix the loop bug

2. **Phase 2: Improve Logic**
   - Better summarization triggers
   - Selective summarization
   - Accurate token counting

3. **Phase 3: Testing & Validation**
   - Comprehensive test suite
   - Performance benchmarks
   - Edge case handling

## Quick Test Commands

```bash
# Test without summarization (current state)
python -m nemo_skills.inference.eval.locagent \
    inference.max_tokens=8192 \
    total_steps=20

# When testing summarization fixes
python -m nemo_skills.inference.eval.locagent \
    inference.max_tokens=8192 \
    total_steps=20 \
    enable_turn_summarization=true \
    min_turns_for_summarization=10
```

## Success Criteria

- [ ] Agent completes 20 turns without getting stuck
- [ ] Summarization reduces token count by at least 30%
- [ ] Agent maintains problem-solving continuity
- [ ] No repeated tool calls due to state loss
- [ ] Clear logging shows summarization decisions
