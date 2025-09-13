# Response Length Management for Artsiv

## Problem Summary

The Artsiv was failing when configured with very large token generation limits (e.g., 81,920 tokens). The model would generate exactly the maximum allowed tokens and get cut off mid-response, unable to output the required `<locations>` tags to complete the task.

## Solution Overview

We've implemented a comprehensive response length management system that:

1. **Prevents Hard Token Limit Cutoffs**: Ensures a buffer of tokens is always reserved so the model can properly complete its response with the required XML tags.

2. **Monitors Response Length**: Tracks token usage during generation and warns when responses are getting too long.

3. **Implements Retry Logic**: If a response is too long, the system can retry with stricter token limits.

4. **Detects Truncation**: Identifies when a response was likely cut off at the token limit.

## Key Components

### 1. Configuration Parameters

```yaml
# Response length management settings
enable_response_length_management: bool = True  # Enable the feature
max_response_tokens: int = 60000  # Normal response limit
max_thinking_tokens: int = 70000  # Response with thinking tags
max_final_turn_tokens: int = 40000  # Final turn (should be concise)
response_length_retry_limit: int = 20000  # Strict limit for retries
enable_response_truncation: bool = True  # Allow truncation if needed
inject_length_warnings: bool = True  # Warn model about length
max_allowed_generation_tokens: int = 75000  # Hard cap
generation_buffer_tokens: int = 5000  # Reserved buffer
```

### 2. Token Buffer Management

When you configure `tokens_to_generate=81920`, the system automatically adjusts this to `76920` (81920 - 5000 buffer) to ensure the model has space to complete its output properly.

### 3. Response Length Monitoring

The system monitors each response and:
- Warns when responses exceed 75% of the limit
- Errors when responses exceed the limit
- Provides detailed statistics about token usage

### 4. Truncation Detection

If the model generates exactly the configured token limit, the system:
- Logs a warning that the response was likely truncated
- Marks the response with `_likely_truncated` flag
- Fails the task if no valid tool/location was extracted

### 5. Retry Mechanism

When a response is too long:
1. First retry uses the configured `response_length_retry_limit` (20,000 tokens)
2. Injects a warning message to the model about length constraints
3. Up to 2 retries are attempted before truncation

## Usage with Large Context Models

For models with large context windows (e.g., 262,144 tokens), the system:
- Allows generous token limits for generation (up to 75,000 tokens)
- Maintains a safety buffer to prevent cutoffs
- Monitors and warns about excessive verbosity
- Ensures the model can always complete its structured output

## Example Configuration

For your use case with:
- `max_seq_length=262144`
- `tokens_to_generate=81920`

The system will:
1. Adjust tokens_to_generate to 76,920 (leaving 5,000 token buffer)
2. Monitor responses for excessive length
3. Warn if responses exceed 60,000 tokens (normal) or 70,000 (with thinking)
4. Detect if the model hits the exact token limit
5. Retry with stricter limits if needed

## Benefits

1. **Prevents Silent Failures**: No more responses cut off mid-generation
2. **Maintains Flexibility**: Still allows very long responses when needed
3. **Provides Visibility**: Clear logging of token usage and issues
4. **Graceful Degradation**: Retries and truncation instead of hard failures
5. **Configurable**: All limits can be adjusted based on your needs

## Future Improvements

1. Dynamic token limit adjustment based on turn number
2. Smart truncation that preserves XML structure better
3. Token usage analytics across multiple runs
4. Adaptive limits based on model behavior
