# Summarization Issue Analysis

## Problem Description

When setting `max_turns` to 20, the system was generating 20 LLM calls but the conversation was stuck in a loop with barely one actual turn progressing. The agent kept making the same tool call repeatedly without moving forward in the problem-solving process.

## Root Cause Analysis

### 1. Turn Structure Corruption

The primary issue was that the summarization logic was modifying the conversation turn structure in a way that broke the agent's ability to track conversation state properly.

**What was happening:**
- The agent would make a tool call (e.g., `view_file`)
- The tool output would be added to the current turn
- During summarization, the turn structure was being modified
- The modified structure caused the agent to lose track of what had been done
- The agent would repeat the same tool call thinking it hadn't been executed

### 2. Context Window Mismanagement

The summarization was being triggered too aggressively:
```python
min_turns_for_summarization: int = 5  # Was triggering too early
```

This meant that after just 5 turns, the system would start trying to summarize, potentially disrupting the natural flow of problem-solving.

### 3. Summary Integration Issues

The summarized turns were being injected back into the conversation with special markers like:
- `[INVESTIGATION STRATEGY]`
- `[ASSISTANT'S REASONING]`
- `_is_summary` flags

These markers and the modified turn structure were confusing the agent's state tracking.

### 4. Async/Sync Mismatch

There were both sync and async versions of summarization functions being imported and used:
```python
summarize_turn = dialog_processor.summarize_turn  # Sync
summarize_turn_async = dialog_processor.summarize_turn_async  # Async
```

This could have led to race conditions or incomplete summarization operations.

## Specific Code Issues

### Issue 1: Turn Structure Modification
```python
# In apply_context_summarization
summarized_turns = await apply_context_summarization(
    data_point['turns'], self.cfg.max_summary_sentences
)
prepared_data_point['turns'] = summarized_turns  # Replacing original turns
```

The summarized turns were replacing the original turns entirely, losing important state information.

### Issue 2: Emergency Summarization
```python
if current_tokens > available_tokens:
    # Emergency summarization was too aggressive
    emergency_summarized = await apply_context_summarization(
        prepared_data_point['turns'], max_sentences=4
    )
```

When token limits were exceeded, emergency summarization with only 4 sentences was losing too much context.

### Issue 3: Tool Output Handling
```python
# Store the full tool output - the assistant needs to see the raw data
# We'll summarize the assistant's reasoning about it, not the data itself
tool_output_to_store = tool_call_result
```

While the comment says tool outputs shouldn't be summarized, the summarization logic may have been inadvertently modifying them.

## Why The Loop Occurred

1. **State Loss**: When turns were summarized, the agent lost track of which tools had been called
2. **Context Confusion**: The summarized content didn't preserve enough detail for the agent to understand what had already been attempted
3. **Turn Association Break**: The relationship between tool calls and their outputs was being disrupted by the summarization process

## Recommended Fixes for Future Implementation

### 1. Preserve Turn Structure
Instead of replacing turns, keep original turns and add summaries as metadata:
```python
turn['summary'] = summary_text  # Add summary as metadata
turn['original_content'] = turn['inputs']  # Preserve original
```

### 2. Selective Summarization
Only summarize completed investigation chunks, not ongoing work:
```python
def can_summarize_turn(turn, next_turn):
    # Only summarize if this investigation chunk is complete
    return (turn.get('tool_call') and 
            next_turn.get('tool_call') and
            turn['tool_call']['tool'] != next_turn['tool_call']['tool'])
```

### 3. Better Token Estimation
Implement more accurate token counting before summarizing:
```python
def should_summarize(turns, max_tokens):
    actual_tokens = count_tokens_accurately(turns)
    return actual_tokens > (max_tokens * 0.8)  # 80% threshold
```

### 4. Maintain State Continuity
Create a state tracking mechanism that survives summarization:
```python
class ConversationState:
    def __init__(self):
        self.tools_called = []
        self.files_viewed = set()
        self.current_investigation = None
```

### 5. Test Summarization Incrementally
- Start with summarizing only non-critical turns
- Test with small conversations first
- Validate that the agent can still progress after summarization
- Use unit tests to verify turn structure preservation

### 6. Separate Summary Display
Instead of modifying the actual conversation history, maintain summaries separately:
```python
conversation = {
    'turns': [...],  # Original turns
    'summaries': {
        'turn_5_to_10': 'Summary text...',
        'turn_11_to_15': 'Summary text...'
    }
}
```

## Testing Strategy

When re-implementing summarization:

1. **Unit Tests**: Test summarization functions in isolation
2. **Integration Tests**: Test with simple conversations first
3. **State Validation**: Verify agent state is preserved after summarization
4. **Progress Tests**: Ensure agent can make progress after summarization
5. **Token Count Tests**: Verify actual token reduction

## Conclusion

The summarization feature needs a complete redesign that:
- Preserves conversation state and structure
- Only summarizes when absolutely necessary
- Maintains the relationship between tool calls and outputs
- Allows the agent to understand what has been done previously

The current implementation was too aggressive in modifying the conversation structure, leading to the agent losing track of its progress and getting stuck in loops.
