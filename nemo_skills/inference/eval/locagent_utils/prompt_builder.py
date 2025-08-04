from typing import Dict, List


class PromptBuilder:
    """Builds multi-turn prompts for the assistant."""

    def __init__(self, prompt_template):
        self.prompt_template = prompt_template

    def build_multi_turn_prompt(self, turns_data: List[Dict]) -> str:
        """Build a multi-turn prompt with proper formatting."""
        # System message
        system_dict = {
            "system": self.prompt_template.config.system,
            "text_begin": self.prompt_template.config.template.text_begin,
            "system_begin": self.prompt_template.config.template.system_begin,
            "system_end": self.prompt_template.config.template.system_end,
        }
        prompt_string = self.prompt_template.SYSTEM_FORMAT.format(**system_dict)

        # Add turns
        for turn in turns_data:
            user_message = self.prompt_template.build_user_message(turn)
            user_dict = {
                "user": user_message,
                "user_begin": self.prompt_template.config.template.user_begin,
                "user_end": self.prompt_template.config.template.user_end,
                "assistant_begin": self.prompt_template.config.template.assistant_begin,
            }

            if "assistant" in turn:
                # Complete turn
                assistant_dict = {
                    "assistant": turn["assistant"],
                    "assistant_end": self.prompt_template.config.template.assistant_end,
                }
                turn_format = self.prompt_template.TURN_BEGIN_FORMAT + self.prompt_template.TURN_END_FORMAT
                prompt_string += turn_format.format(**user_dict, **assistant_dict)
            else:
                # Last turn - waiting for assistant response
                turn_format = self.prompt_template.TURN_BEGIN_FORMAT
                prompt_string += turn_format.format(**user_dict)

        return prompt_string
