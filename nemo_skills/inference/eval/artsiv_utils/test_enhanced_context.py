#!/usr/bin/env python3
"""
Test script for enhanced context management.

This demonstrates how the enhanced context management handles long contexts.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

from nemo_skills.inference.eval.artsiv_utils.enhanced_context_management import (
    TokenCounter, 
    count_dialogue_tokens,
    enhanced_truncate_dialogue,
    check_context_before_generation
)


def create_long_dialogue(num_turns=50):
    """Create a dialogue with many turns to test truncation."""
    turns = []
    
    # Initial problem statement
    turns.append({
        'inputs': 'Fix the bug in the authentication system where users are getting logged out randomly. ' * 10,
        'assistant': 'I need to investigate the authentication system to find the cause of random logouts.',
        'tool_call': {'tool': 'search', 'query': 'authentication logout'},
        'tool_output': 'Found 15 files related to authentication...\n' * 100
    })
    
    # Add many investigation turns
    for i in range(1, num_turns):
        turns.append({
            'inputs': f'Turn {i} tool output with lots of code:\n' + 'def authenticate():\n    pass\n' * 50,
            'assistant': f'I found something interesting in turn {i}. Let me investigate further.',
            'tool_call': {'tool': 'view_file', 'path': f'file_{i}.py'},
            'tool_output': 'File content:\n' + 'import something\n' * 200
        })
    
    return turns


def test_token_counting():
    """Test token counting accuracy."""
    print("=== Testing Token Counting ===")
    
    counter = TokenCounter()
    
    test_texts = [
        "Hello world",  # Simple text
        "def foo():\n    return 42",  # Code
        "path/to/file.py:L123-L456",  # Path with special chars
        "{'key': 'value', 'list': [1, 2, 3]}",  # JSON
    ]
    
    for text in test_texts:
        tokens = counter.count_tokens(text)
        print(f"Text: {text[:50]}...")
        print(f"Tokens: {tokens}")
        print()


def test_enhanced_truncation():
    """Test enhanced truncation with long dialogue."""
    print("\n=== Testing Enhanced Truncation ===")
    
    # Create a long dialogue
    turns = create_long_dialogue(30)
    print(f"Created dialogue with {len(turns)} turns")
    
    # Test truncation
    max_seq_length = 8192  # Small limit to force truncation
    tokens_to_generate = 2048
    
    truncated_turns, stats = enhanced_truncate_dialogue(
        turns, 
        max_seq_length, 
        tokens_to_generate,
        safety_margin=0.9
    )
    
    print(f"\nTruncation Results:")
    print(f"- Original: {stats['original_turns']} turns, ~{stats['original_tokens']} tokens")
    print(f"- Final: {stats['final_turns']} turns, ~{stats['final_tokens']} tokens")
    print(f"- Strategy used: {stats['truncation_strategy']}")
    print(f"- Truncation applied: {stats['truncation_applied']}")


def test_proactive_check():
    """Test proactive context checking."""
    print("\n\n=== Testing Proactive Context Check ===")
    
    # Create a config-like object
    class MockConfig:
        max_seq_length = 4096
        tokens_to_generate = 1024
        context_safety_margin = 0.9
        model = 'gpt-4'
    
    cfg = MockConfig()
    
    # Test with dialogue that fits
    short_dialogue = {'turns': create_long_dialogue(5)}
    will_fit, error_msg, stats = check_context_before_generation(short_dialogue, cfg)
    print(f"\nShort dialogue check:")
    print(f"- Will fit: {will_fit}")
    print(f"- Current tokens: {stats['current_tokens']}")
    print(f"- Available tokens: {stats['available_tokens']}")
    
    # Test with dialogue that's too long
    long_dialogue = {'turns': create_long_dialogue(50)}
    will_fit, error_msg, stats = check_context_before_generation(long_dialogue, cfg)
    print(f"\nLong dialogue check:")
    print(f"- Will fit: {will_fit}")
    print(f"- Error: {error_msg}")
    print(f"- Current tokens: {stats['current_tokens']}")
    print(f"- Available tokens: {stats['available_tokens']}")


def main():
    """Run all tests."""
    print("Enhanced Context Management Test Suite")
    print("=" * 50)
    
    test_token_counting()
    test_enhanced_truncation()
    test_proactive_check()
    
    print("\n\nTest complete!")


if __name__ == "__main__":
    main()
