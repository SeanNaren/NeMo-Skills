"""
Final turn prompt injection for LocAgent.

This module provides functionality to modify the prompt on the final turn
to encourage the model to provide location predictions.
"""

import logging
import copy
from typing import List, Dict, Optional

LOG = logging.getLogger(__name__)


def inject_final_turn_instruction(
    turns: List[Dict], 
    is_final_turn: bool,
    instruction_type: str = "standard",
    custom_instruction: Optional[str] = None
) -> List[Dict]:
    """
    Inject a special instruction when it's the final turn.
    
    Args:
        turns: List of dialogue turns
        is_final_turn: Whether this is the last allowed turn
        instruction_type: Type of instruction to inject
        custom_instruction: Custom instruction text (overrides instruction_type)
        
    Returns:
        Modified turns with final turn instruction if applicable
    """
    if not is_final_turn or not turns:
        return turns
    
    # Deep copy to avoid modifying original
    modified_turns = copy.deepcopy(turns)
    
    # Define different types of final turn instructions
    instructions = {
        "standard": (
            "\n\n**FINAL TURN NOTICE**\n"
            "You have reached your **maximum allowed tool calls**. You must now emit `<locations>` based on your investigation.\n\n"
            "**Required Action:**\n"
            "1. Review all code you have inspected\n"
            "2. Identify **every** file and line range that needs editing\n"
            "3. Output your findings using the exact format:\n"
            "   ```xml\n"
            "   <think>\n"
            "   Your final analysis here...\n"
            "   </think>\n"
            "   \n"
            "   <locations>\n"
            "   path/to/file.py:L<start>-L<end>\n"
            "   another/file.py:L<start>-L<end>\n"
            "   </locations>\n"
            "   ```\n\n"
            "**Important:** If you haven't found the exact locations, provide your **best assessment** "
            "based on the evidence gathered. Include all suspicious areas."
        ),
        "urgent": (
            "\n\n**[URGENT - FINAL TURN]**\n"
            "**No more tool calls allowed!** You MUST provide `<locations>` NOW.\n\n"
            "Format your response EXACTLY as:\n"
            "```xml\n"
            "<think>Final reasoning...</think>\n"
            "\n"
            "<locations>\n"
            "file.py:L<start>-L<end>\n"
            "</locations>\n"
            "```\n"
            "**Submit your best guess based on investigation so far.**"
        ),
        "gentle": (
            "\n\n**Note: Final Turn**\n"
            "This is your last opportunity. Please conclude with `<locations>` based on your findings.\n\n"
            "Remember to use:\n"
            "- `<think>` tags for your final analysis\n"
            "- `<locations>` tags with format: `path/to/file.py:L<start>-L<end>`\n\n"
            "Even partial findings are valuable - include all areas you suspect need changes."
        ),
        "detailed": (
            "\n\n**FINAL TURN - LOCATION SUBMISSION REQUIRED**\n\n"
            "**You have exhausted your tool call budget.** Follow these steps:\n\n"
            "1. **Synthesize your findings** from all inspected code\n"
            "2. **Identify edit locations** considering:\n"
            "   - Primary bug location(s)\n"
            "   - Related helper functions\n"
            "   - Test files that may need updates\n"
            "   - Configuration or interface changes\n\n"
            "3. **Format your response** EXACTLY as:\n"
            "   ```xml\n"
            "   <think>\n"
            "   // Summarize your investigation findings\n"
            "   // Explain why each location needs editing\n"
            "   </think>\n"
            "   \n"
            "   <locations>\n"
            "   src/main/module.py:L45-L67\n"
            "   src/utils/helper.py:L123-L145\n"
            "   tests/test_module.py:L89-L92\n"
            "   </locations>\n"
            "   ```\n\n"
            "**Strict rules:**\n"
            "- NO more `<tool_call>` blocks allowed\n"
            "- Must include `<think>` and `<locations>` tags\n"
            "- Better to over-include than miss locations"
        ),
        "aligned": (
            "\n\n**Interaction protocol update**\n"
            "You have reached the maximum number of tool calls. You **must** now reply with a `<locations>` block.\n\n"
            "**Strict rules for this final response:**\n"
            "- Assistant message must follow the EXACT structure: `<think>...</think>` followed by `<locations>...</locations>`\n"
            "- **DO NOT** issue any more `<tool_call>` blocks\n"
            "- **DO NOT** include any text outside the required tags\n"
            "- Based on all code you have inspected, emit **every** file and line range that needs editing\n"
            "- If uncertain about exact lines, include your best assessment\n\n"
            "**Output format:**\n"
            "```xml\n"
            "<think>\n"
            "Based on my investigation, I have identified the following locations that need editing...\n"
            "</think>\n"
            "\n"
            "<locations>\n"
            "path/to/file.py:L<start>-L<end>\n"
            "another/file.rs:L<start>-L<end>\n"
            "</locations>\n"
            "```\n\n"
            "Remember: It's better to over-inspect and over-include than to miss a required edit location."
        )
    }
    
    # Get the instruction text
    if custom_instruction:
        instruction_text = custom_instruction
    else:
        instruction_text = instructions.get(instruction_type, instructions["standard"])
    
    # Find the last turn with inputs
    last_input_idx = -1
    for i in range(len(modified_turns) - 1, -1, -1):
        if 'inputs' in modified_turns[i] and modified_turns[i]['inputs']:
            last_input_idx = i
            break
    
    if last_input_idx >= 0:
        # Append instruction to the last input
        modified_turns[last_input_idx]['inputs'] += instruction_text
        LOG.info(f"Injected final turn instruction (type: {instruction_type})")
    else:
        # If no inputs found, create a new turn with the instruction
        modified_turns.append({
            'inputs': instruction_text,
            'assistant': '',
            'tool_call': None,
            'tool_output': ''
        })
        LOG.info("Added new turn with final turn instruction")
    
    return modified_turns


def should_inject_final_turn(
    cur_step: int,
    total_steps: int,
    status: Optional[str],
    enable_final_turn_prompt: bool = True,
    final_turn_threshold: float = 1.0
) -> bool:
    """
    Determine if we should inject the final turn instruction.
    
    Args:
        cur_step: Current step number (0-indexed)
        total_steps: Total allowed steps
        status: Current status (None if still investigating)
        enable_final_turn_prompt: Whether the feature is enabled
        final_turn_threshold: Fraction of steps before triggering (1.0 = only last turn)
        
    Returns:
        True if we should inject the final turn instruction
    """
    if not enable_final_turn_prompt or status is not None:
        return False
    
    # Check if we're at or past the threshold
    steps_completed = cur_step + 1
    threshold_step = int(total_steps * final_turn_threshold)
    
    return steps_completed >= threshold_step


def create_location_reminder_turn(context_summary: Optional[str] = None) -> Dict:
    """
    Create a standalone turn that reminds the model to provide locations.
    
    Args:
        context_summary: Optional summary of what has been investigated
        
    Returns:
        A turn dictionary with location reminder
    """
    base_message = (
        "**Location Submission Required**\n\n"
        "Based on your investigation, you must now emit `<locations>`. "
        "Format your response as:\n"
        "```xml\n"
        "<think>Your analysis...</think>\n"
        "\n"
        "<locations>\n"
        "file.py:L<start>-L<end>\n"
        "</locations>\n"
        "```"
    )
    
    if context_summary:
        message = f"**Investigation Summary:**\n{context_summary}\n\n{base_message}"
    else:
        message = base_message
    
    return {
        'inputs': f"{message}",
        'assistant': '',
        'tool_call': None,
        'tool_output': ''
    }
