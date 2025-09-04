#!/usr/bin/env python3
"""
Test script for final turn prompt injection.

This demonstrates how the final turn prompt helps the model provide locations
even when reaching the step limit.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

from nemo_skills.inference.eval.locagent_utils.final_turn_prompt import (
    inject_final_turn_instruction,
    should_inject_final_turn,
    create_location_reminder_turn
)


def create_sample_dialogue():
    """Create a sample dialogue that's approaching the step limit."""
    turns = [
        {
            'inputs': 'Fix the authentication bug where users get logged out randomly.',
            'assistant': 'I need to investigate the authentication system to find the issue.',
            'tool_call': {'tool': 'search', 'query': 'authentication logout'},
            'tool_output': 'Found files: auth/middleware.py, auth/session.py, auth/handlers.py'
        },
        {
            'inputs': 'Here are the search results showing relevant files.',
            'assistant': 'Let me examine the session handling code.',
            'tool_call': {'tool': 'view_file', 'path': 'auth/session.py'},
            'tool_output': 'File content shows session timeout logic...'
        },
        {
            'inputs': 'The session file shows timeout handling.',
            'assistant': 'I see potential issues with the session timeout. Let me check the middleware.',
            'tool_call': {'tool': 'view_file', 'path': 'auth/middleware.py'},
            'tool_output': 'Middleware shows session validation logic...'
        }
    ]
    return turns


def test_final_turn_detection():
    """Test the logic for detecting when to inject final turn prompt."""
    print("=== Testing Final Turn Detection ===\n")
    
    test_cases = [
        # (cur_step, total_steps, status, threshold, expected)
        (18, 20, None, 1.0, False),  # Not final turn yet
        (19, 20, None, 1.0, True),   # Final turn
        (19, 20, "success", 1.0, False),  # Already succeeded
        (15, 20, None, 0.8, False),  # Below 80% threshold
        (16, 20, None, 0.8, True),   # At 80% threshold
    ]
    
    for cur_step, total_steps, status, threshold, expected in test_cases:
        result = should_inject_final_turn(
            cur_step, total_steps, status, 
            enable_final_turn_prompt=True,
            final_turn_threshold=threshold
        )
        print(f"Step {cur_step + 1}/{total_steps}, status={status}, threshold={threshold}: "
              f"inject={result} (expected={expected})")
    print()


def test_instruction_injection():
    """Test different types of final turn instructions."""
    print("=== Testing Instruction Injection ===\n")
    
    turns = create_sample_dialogue()
    
    instruction_types = ["aligned", "standard", "urgent", "gentle", "detailed"]
    
    for inst_type in instruction_types:
        print(f"\n--- {inst_type.upper()} Instruction ---")
        modified_turns = inject_final_turn_instruction(
            turns, 
            is_final_turn=True,
            instruction_type=inst_type
        )
        
        # Show the last turn's input with the injected instruction
        last_input = modified_turns[-1]['inputs']
        # Truncate for display
        if len(last_input) > 200:
            display = last_input[:200] + "..."
        else:
            display = last_input
        print(f"Modified input: {display}")


def test_custom_instruction():
    """Test custom instruction injection."""
    print("\n\n=== Testing Custom Instruction ===\n")
    
    turns = create_sample_dialogue()
    
    custom = (
        "\n\n[FINAL ANALYSIS REQUIRED]\n"
        "You've investigated thoroughly. Now provide your conclusions:\n"
        "1. What is the most likely cause?\n"
        "2. Which files contain the issue?\n"
        "3. Specify exact locations using <locations>\n"
        "Even partial answers are valuable!"
    )
    
    modified_turns = inject_final_turn_instruction(
        turns,
        is_final_turn=True,
        custom_instruction=custom
    )
    
    print(f"Custom instruction added: {modified_turns[-1]['inputs'][-100:]}...")


def test_location_reminder():
    """Test standalone location reminder turn."""
    print("\n\n=== Testing Location Reminder Turn ===\n")
    
    # Without context
    reminder1 = create_location_reminder_turn()
    print("Basic reminder:")
    print(f"  {reminder1['inputs']}\n")
    
    # With context summary
    context = "You've investigated the authentication system and found issues in session handling."
    reminder2 = create_location_reminder_turn(context)
    print("Reminder with context:")
    print(f"  {reminder2['inputs']}")


def simulate_final_turns():
    """Simulate what happens in the final turns with and without the feature."""
    print("\n\n=== Simulating Final Turns ===\n")
    
    print("WITHOUT final turn prompt:")
    print("  Turn 19: Agent investigates another file...")
    print("  Turn 20: Agent says 'I need to check one more thing...'")
    print("  Result: FAILED - max_steps_exceeded")
    print("  Locations: None provided")
    
    print("\nWITH final turn prompt:")
    print("  Turn 19: Agent investigates another file...")
    print("  Turn 20: [FINAL TURN NOTICE injected]")
    print("           Agent: 'Based on my investigation, the issue is in:'")
    print("           <locations>")
    print("           auth/session.py:45-67")
    print("           auth/middleware.py:123-145")
    print("           </locations>")
    print("  Result: SUCCESS")
    print("  Locations: 2 predictions provided")


def main():
    """Run all tests."""
    print("Final Turn Prompt Feature Test Suite")
    print("=" * 50)
    
    test_final_turn_detection()
    test_instruction_injection()
    test_custom_instruction()
    test_location_reminder()
    simulate_final_turns()
    
    print("\n\nTest complete!")
    print("\nKey Takeaway: This feature helps recover value from investigations that")
    print("would otherwise fail due to reaching the step limit.")


if __name__ == "__main__":
    main()
