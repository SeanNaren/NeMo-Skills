import hashlib
import json
from datetime import datetime
from typing import Dict, List

from transformers import AutoTokenizer

from config import Config
from core.logger import ConversationTracker

from ..dialog_processor import DialogProcessor
from .logger import logger
from .prompt_builder import PromptBuilder
from .tool_executor import ToolExecutor


class ConversationManager:
    def __init__(
        self,
        llm,
        prompt_builder: PromptBuilder,
        tool_executor: ToolExecutor,
        config: Config,
    ):
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.tool_executor = tool_executor
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_model_name)
        self.conversation_tracker = None

    def chat_loop(
        self,
        problem_statement: str,
        repo_tree: str,
        repo_dir: str,
        conversation_tracker: ConversationTracker = None,
    ) -> tuple[List[str], List[str]]:
        """Run the conversation loop with the assistant."""
        self.conversation_tracker = conversation_tracker
        initial_prompt = self._build_initial_prompt(problem_statement, repo_tree)
        conversation_turns = []
        raw_outputs = []  # Store all raw model outputs
        previous_tool_call = None  # Track previous tool call to detect duplicates

        # Add initial user message to conversation tracker
        if self.conversation_tracker:
            self.conversation_tracker.add_user_message(initial_prompt, 1)

        logger().info(f"💬 Starting conversation ({self.config.max_num_turns} turns max)")
        for turn_num in range(self.config.max_num_turns):
            logger().info(f"        {'=' * 20} TURN {turn_num + 1}/{self.config.max_num_turns} {'=' * 20}")

            # Try to generate a response with retry logic
            outputs, raw_output, processed_output, success = self._generate_response_with_retry(
                turn_num, initial_prompt, conversation_turns, repo_tree
            )

            if not success:
                logger().error(f"        Failed to generate valid response after all retries")
                if self.conversation_tracker:
                    self.conversation_tracker.mark_failed("Failed to generate valid response after all retries")
                return [], raw_outputs

            # Store raw output
            raw_outputs.append(
                {
                    "turn": turn_num + 1,
                    "raw_response": raw_output,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            logger().verbose(f"        Processing model output")

            # Handle response
            if processed_output["type"] == "locations":
                locations = [loc["raw"] for loc in processed_output["locations"] if "raw" in loc]
                logger().success(f"        Found {len(locations)} location predictions")

                # Add assistant response to conversation tracker
                if self.conversation_tracker:
                    assistant_response = raw_output
                    self.conversation_tracker.add_assistant_message(assistant_response, turn_num + 1)
                    # Update metadata to reflect the final turn and locations found
                    self.conversation_tracker.update_total_turns(turn_num + 1)
                    self.conversation_tracker.update_locations_found(len(locations))

                return locations, raw_outputs
            elif processed_output["type"] == "tool_calls":
                logger().verbose(f"        Processing {len(processed_output['tool_calls'])} tool calls")

                # Check for consecutive duplicate tool calls
                current_tool_call = processed_output["tool_calls"][0] if processed_output["tool_calls"] else None
                if current_tool_call and previous_tool_call and self._is_same_tool_call(current_tool_call, previous_tool_call):
                    logger().warning(f"        Detected consecutive duplicate tool call: {current_tool_call.get('tool', 'unknown')}")
                    logger().info(f"        Regenerating response with higher temperature to encourage creativity and break the loop")

                    # Try to generate a response with retry logic (force different temperature)
                    outputs, raw_output, processed_output, success = self._generate_response_with_retry(
                        turn_num,
                        initial_prompt,
                        conversation_turns,
                        repo_tree,
                        force_retry=True,
                    )

                    if not success:
                        logger().error(f"        Failed to generate valid response after duplicate detection")
                        if self.conversation_tracker:
                            self.conversation_tracker.mark_failed("Failed to generate valid response after duplicate detection")
                        return [], raw_outputs

                    # Update raw outputs with the new response
                    raw_outputs[-1] = {
                        "turn": turn_num + 1,
                        "raw_response": raw_output,
                        "timestamp": datetime.now().isoformat(),
                    }

                    # Re-process the new output
                    if processed_output["type"] == "locations":
                        locations = [loc["raw"] for loc in processed_output["locations"] if "raw" in loc]
                        logger().success(f"        Found {len(locations)} location predictions after duplicate retry")

                        # Add assistant response to conversation tracker
                        if self.conversation_tracker:
                            assistant_response = raw_output
                            self.conversation_tracker.add_assistant_message(assistant_response, turn_num + 1)
                            self.conversation_tracker.update_total_turns(turn_num + 1)
                            self.conversation_tracker.update_locations_found(len(locations))

                        return locations, raw_outputs
                    elif processed_output["type"] == "tool_calls":
                        logger().verbose(f"        Processing {len(processed_output['tool_calls'])} tool calls after duplicate retry")
                        current_tool_call = processed_output["tool_calls"][0] if processed_output["tool_calls"] else None

                # Update previous tool call for next iteration
                previous_tool_call = current_tool_call

                self._handle_tool_calls(
                    processed_output,
                    conversation_turns,
                    repo_dir,
                    turn_num + 1,
                    initial_prompt,
                )
            else:
                # Store the assistant's response even though it's not in the expected format
                if self.conversation_tracker:
                    assistant_response = raw_output
                    self.conversation_tracker.add_assistant_message(assistant_response, turn_num + 1)
                    self.conversation_tracker.mark_failed("Assistant response not in expected format")
                logger().error(f"        Unknown output type: {processed_output['type']}")
                return [], raw_outputs

        logger().warning("        Maximum conversation turns reached")
        if self.conversation_tracker:
            self.conversation_tracker.mark_failed("Maximum conversation turns reached without finding locations")
        return [], raw_outputs

    def _build_initial_prompt(self, problem_statement: str, repo_tree: str) -> str:
        """Build the initial user prompt."""
        prompt = f"""### Problem Description
        `{problem_statement}`
"""

        # Only include repository structure if enabled in config
        if self.config.include_repo_structure:
            prompt += f"""
### Repository Structure
{repo_tree}
"""

        return prompt

    def _build_turns_data(self, initial_prompt: str, conversation_turns: List[Dict]) -> List[Dict]:
        """Build turns data for multi-turn prompting."""
        if not conversation_turns:
            return [{"question": initial_prompt}]

        turns_data = []

        # Handle the first turn
        if "assistant" in conversation_turns[0]:
            turns_data.append(
                {
                    "question": initial_prompt,
                    "assistant": conversation_turns[0]["assistant"],
                }
            )
        else:
            # First turn only has user message (tool result)
            turns_data.append({"question": initial_prompt})
            turns_data.append({"question": conversation_turns[0]["user"]})
            return turns_data

        # Add subsequent turns
        for turn in conversation_turns[1:]:
            if "assistant" in turn and "user" in turn:
                # Complete turn with both assistant and user
                turns_data.append({"question": turn["user"], "assistant": turn["assistant"]})
            elif "user" in turn:
                # Turn with only user message (tool result)
                turns_data.append({"question": turn["user"]})
            elif "assistant" in turn:
                # Turn with only assistant message
                turns_data.append({"assistant": turn["assistant"]})

        return turns_data

    def _truncate_prompt(self, prompt: str, excess_tokens: int, tokenizer) -> str:
        """Intelligently truncate a prompt that's too long while preserving important parts."""
        # Split prompt into sections
        sections = prompt.split("\n\n")

        # For simple prompts, be more conservative with truncation
        if len(sections) <= 3:
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            target_length = len(tokens) - excess_tokens

            # Ensure we keep at least 80% of the original prompt
            min_length = max(1000, int(len(tokens) * 0.8))
            target_length = max(target_length, min_length)

            # Decode back to text, ensuring we don't cut in the middle of a token
            truncated_tokens = tokens[:target_length]
            return tokenizer.decode(truncated_tokens, skip_special_tokens=True)

        # For multi-turn conversations, choose truncation strategy based on config
        if self.config.truncate_from_end:
            # Strategy: Keep beginning (system + problem) + recent turns from end
            # Always keep the first 2 sections (system message + initial problem)
            kept_sections = sections[:2]

            # Calculate tokens for kept sections
            kept_tokens = len(tokenizer.encode("\n\n".join(kept_sections), add_special_tokens=False))

            # Reserve space for response (at least 2000 tokens)
            available_tokens = self.config.max_context_length - kept_tokens - 2000

            # If we don't have enough space even for the essential parts, be more aggressive
            if available_tokens <= 0:
                # Keep only the essential parts and truncate them if needed
                essential_prompt = "\n\n".join(kept_sections)
                tokens = tokenizer.encode(essential_prompt, add_special_tokens=False)
                target_length = self.config.max_context_length - 2000  # Reserve 2000 for response
                if target_length > 0:
                    truncated_tokens = tokens[:target_length]
                    return tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                else:
                    # Last resort: keep just the problem description
                    return kept_sections[1] if len(kept_sections) > 1 else kept_sections[0]

            # Add recent turns from the end, but be more conservative
            recent_sections = []
            for section in reversed(sections[2:]):
                section_tokens = len(tokenizer.encode(section, add_special_tokens=False))
                if kept_tokens + section_tokens <= available_tokens:
                    recent_sections.insert(0, section)
                    kept_tokens += section_tokens
                else:
                    # If we can't fit the full section, try to fit part of it
                    if section_tokens > 500:  # Only try to split large sections
                        partial_tokens = available_tokens - kept_tokens
                        if partial_tokens > 200:  # Only if we have meaningful space
                            section_tokens_list = tokenizer.encode(section, add_special_tokens=False)
                            partial_section = tokenizer.decode(
                                section_tokens_list[:partial_tokens],
                                skip_special_tokens=True,
                            )
                            recent_sections.insert(0, partial_section + "\n[Note: Section truncated]")
                    break

            # Combine kept sections
            truncated_prompt = "\n\n".join(kept_sections + recent_sections)

            # Add a note about truncation if significant parts were removed
            if len(recent_sections) < len(sections) - 2:
                truncated_prompt += f"\n\n[Note: Some conversation history was truncated due to length]"

            return truncated_prompt
        else:
            # Strategy: Keep end (recent turns) + beginning (system + problem)
            # Always keep the last section (most recent user message)
            kept_sections = [sections[-1]] if sections else []

            # Calculate tokens for kept sections
            kept_tokens = len(tokenizer.encode("\n\n".join(kept_sections), add_special_tokens=False))

            # Reserve space for response (at least 2000 tokens)
            available_tokens = self.config.max_context_length - kept_tokens - 2000

            # If we don't have enough space even for the essential parts, be more aggressive
            if available_tokens <= 0:
                # Keep only the essential parts and truncate them if needed
                essential_prompt = "\n\n".join(kept_sections)
                tokens = tokenizer.encode(essential_prompt, add_special_tokens=False)
                target_length = self.config.max_context_length - 2000  # Reserve 2000 for response
                if target_length > 0:
                    truncated_tokens = tokens[:target_length]
                    return tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                else:
                    # Last resort: keep just the most recent message
                    return kept_sections[0] if kept_sections else ""

            # Add sections from the beginning (system + problem + early turns)
            early_sections = []
            for section in sections[:-1]:  # Exclude the last section which we already kept
                section_tokens = len(tokenizer.encode(section, add_special_tokens=False))
                if kept_tokens + section_tokens <= available_tokens:
                    early_sections.append(section)
                    kept_tokens += section_tokens
                else:
                    # If we can't fit the full section, try to fit part of it
                    if section_tokens > 500:  # Only try to split large sections
                        partial_tokens = available_tokens - kept_tokens
                        if partial_tokens > 200:  # Only if we have meaningful space
                            section_tokens_list = tokenizer.encode(section, add_special_tokens=False)
                            partial_section = tokenizer.decode(
                                section_tokens_list[:partial_tokens],
                                skip_special_tokens=True,
                            )
                            early_sections.append(partial_section + "\n[Note: Section truncated]")
                    break

            # Combine sections: early sections + kept sections (recent)
            truncated_prompt = "\n\n".join(early_sections + kept_sections)

            # Add a note about truncation if significant parts were removed
            if len(early_sections) < len(sections) - 1:
                truncated_prompt += f"\n\n[Note: Some conversation history was truncated due to length]"

            return truncated_prompt

    def _generate_response_with_retry(
        self,
        turn_num: int,
        initial_prompt: str,
        conversation_turns: List[Dict],
        repo_tree: str,
        force_retry: bool = False,
    ) -> tuple:
        """Generate response with retry logic using different temperatures."""
        if force_retry:
            # For duplicate tool calls, use higher temperatures to encourage creativity
            # Start with higher temperatures to break out of repetitive patterns
            higher_temperatures = [
                0.8,
                0.9,
                1.0,
            ]  # Higher temperatures for more creative responses
            temperatures_to_try = higher_temperatures
            logger().info(f"        Force retry mode: using higher temperatures {higher_temperatures} to break repetitive pattern")
        else:
            temperatures_to_try = [self.config.temperature] + self.config.retry_temperatures

        for retry_idx, temperature in enumerate(temperatures_to_try):
            if retry_idx > 0:
                logger().warning(f"        Retry {retry_idx}/{self.config.max_retries} with temperature {temperature}")

            # Build prompt
            if not conversation_turns:
                full_prompt = self.prompt_builder.prompt_template.fill({"question": initial_prompt})
            else:
                turns_data = self._build_turns_data(initial_prompt, conversation_turns)
                full_prompt = self.prompt_builder.build_multi_turn_prompt(turns_data)

            # Generate response
            num_tokens = len(self.tokenizer.encode(full_prompt, add_special_tokens=False))
            repo_tree_tokens = len(self.tokenizer.encode(repo_tree, add_special_tokens=False))
            repo_tree_percentage = (repo_tree_tokens / num_tokens * 100) if num_tokens > 0 else 0
            logger().info(f"        Prompt tokens: {num_tokens}, Repo tree tokens: {repo_tree_tokens} ({repo_tree_percentage:.1f}%)")
            max_tokens = self.config.max_context_length - num_tokens - 100

            # Handle case where prompt is too long
            if max_tokens <= 0:
                logger().warning(f"        Prompt too long ({num_tokens} tokens), truncating context")
                # Calculate how much we need to reduce the prompt by
                excess_tokens = abs(max_tokens) + 2000  # Add buffer for response

                # Try to truncate the prompt intelligently
                truncated_prompt = self._truncate_prompt(full_prompt, excess_tokens, self.tokenizer)
                num_tokens = len(self.tokenizer.encode(truncated_prompt, add_special_tokens=False))
                max_tokens = max(1000, self.config.max_context_length - num_tokens - 1000)  # Ensure reasonable response length
                full_prompt = truncated_prompt

                logger().info(f"        Truncated prompt to {num_tokens} tokens, max_tokens: {max_tokens}")
            else:
                logger().verbose(f"        Generating response (max tokens: {max_tokens})")

            # Ensure minimum reasonable response length
            max_tokens = max(1000, max_tokens)
            logger().verbose(f"        Final max_tokens: {max_tokens}")

            logger().info(f"        Generating response with temperature {temperature}...")

            try:
                # Generate with temperature
                outputs = self.llm.generate(
                    prompts=[full_prompt],
                    tokens_to_generate=max_tokens,
                    temperature=temperature,
                )

                raw_output = outputs[0]["generation"] if isinstance(outputs, list) else str(outputs)

                # Check if response is valid (not empty or just whitespace)
                if raw_output and raw_output.strip():
                    # Additional check: if this is a tool call response, verify it can be parsed
                    processed_output = DialogProcessor.process_output(outputs, self.config)
                    if processed_output["type"] == "tool_calls" and not processed_output["tool_calls"]:
                        # Tool calls were expected but none were found - likely malformed
                        logger().warning(f"        Malformed tool call response with temperature {temperature}")
                        continue

                    logger().success(f"        Generated valid response with temperature {temperature}")
                    return outputs, raw_output, processed_output, True
                else:
                    logger().warning(f"        Empty response with temperature {temperature}")

            except Exception as e:
                logger().warning(f"        Generation failed with temperature {temperature}: {e}")

        # If we get here, all retries failed
        logger().error(f"        All retries failed")
        return None, "", None, False

    def _handle_tool_calls(
        self,
        processed_output: Dict,
        conversation_turns: List[Dict],
        repo_dir: str,
        turn_number: int,
        initial_prompt: str = None,
    ):
        """Handle tool calls and update conversation."""
        tool_calls = processed_output.get("tool_calls", [])
        logger().verbose(f"        Model made {len(tool_calls)} tool calls")

        # Check if we have any tool calls to process
        if not tool_calls:
            logger().warning("        No tool calls found in processed output")
            # Add a default response to continue the conversation
            assistant_response = "I need to use tools to help you. Let me explore the repository structure first."
            if self.conversation_tracker:
                self.conversation_tracker.add_assistant_message(assistant_response, turn_number)
            conversation_turns.append(
                {
                    "assistant": assistant_response,
                    "user": "Please use the repo_tree tool to explore the repository.",
                }
            )
            return

        tool_call = tool_calls[0]
        tool_result = self.tool_executor.execute_tool(tool_call, repo_dir)
        logger().verbose(f"        Tool execution completed, result length: {len(tool_result)} characters")

        # If this is a view tool call and summarization is enabled, generate a summary of the file content
        if tool_call.get("tool") == "view" and self.config.summarize_file_content:
            file_summary = self._generate_file_summary(tool_result, tool_call)
            # Use the summary instead of the full file content
            tool_result = file_summary
            logger().verbose(f"        Generated file summary: {len(file_summary)} characters")
        elif tool_call.get("tool") == "view" and not self.config.summarize_file_content:
            logger().verbose(f"        File summarization disabled, using full content")

        # Add tool result to conversation (no second generation needed)
        logger().verbose(f"        Adding tool result to conversation")

        # Add to conversation tracker
        if self.conversation_tracker:
            self.conversation_tracker.add_tool_call(tool_call.get("tool", "unknown"), tool_call, tool_result, turn_number)

        # Add to conversation - the model already generated its response, we just add the tool result
        conversation_turns.append({"user": f"Tool Result:\n{tool_result}"})

    def _generate_file_summary(self, file_content: str, tool_call: Dict) -> str:
        """Generate a concise summary of the file content."""
        try:
            # Extract file path and line range from the tool call
            file_path = tool_call.get("path", "unknown")
            view_range = tool_call.get("view_range")

            # The file_content already contains the summary request, so we can use it directly
            # Generate summary using the same LLM
            logger().verbose(f"        Generating summary for {file_path}")
            summary_outputs = self.llm.generate(prompts=[file_content])
            summary = summary_outputs[0]["generation"] if isinstance(summary_outputs, list) else str(summary_outputs)

            # Clean up the summary - extract just the summary part
            summary = summary.strip()

            # Remove any thinking content
            if "<think>" in summary and "</think>" in summary:
                summary = summary.split("</think>")[1].strip()

            # Create the final summary format
            line_range_str = f"L{view_range[0]}-L{view_range[1]}" if view_range else "entire file"
            final_summary = f"File: {file_path} ({line_range_str})\nSummary: {summary}"

            return final_summary

        except Exception as e:
            logger().warning(f"        Failed to generate file summary: {e}")
            # Fallback to original content if summary generation fails
            return file_content

    def _is_same_tool_call(self, tool_call1: Dict, tool_call2: Dict) -> bool:
        """Check if two tool calls are the same (for duplicate detection)."""
        if not tool_call1 or not tool_call2:
            return False

        # Compare tool type
        if tool_call1.get("tool") != tool_call2.get("tool"):
            return False

        # For view tool, compare path and view_range
        if tool_call1.get("tool") == "view":
            return tool_call1.get("path") == tool_call2.get("path") and tool_call1.get("view_range") == tool_call2.get("view_range")

        # For codebase_search tool, compare query
        elif tool_call1.get("tool") == "codebase_search":
            return tool_call1.get("query") == tool_call2.get("query")

        # For repo_tree tool, they're considered the same if both are repo_tree
        elif tool_call1.get("tool") == "repo_tree":
            return True

        # For unknown tools, compare all fields
        return tool_call1 == tool_call2

    def _get_tool_call_hash(self, tool_call: Dict) -> str:
        """Create a hash of a tool call for comparison."""
        if not tool_call:
            return ""

        # Create a normalized representation for hashing
        if tool_call.get("tool") == "view":
            # For view tool, use path and view_range
            normalized = {
                "tool": "view",
                "path": tool_call.get("path", ""),
                "view_range": tool_call.get("view_range"),
            }
        elif tool_call.get("tool") == "codebase_search":
            # For codebase_search tool, use query
            normalized = {
                "tool": "codebase_search",
                "query": tool_call.get("query", ""),
            }
        elif tool_call.get("tool") == "repo_tree":
            # For repo_tree tool, just use the tool type
            normalized = {"tool": "repo_tree"}
        else:
            # For unknown tools, use all fields
            normalized = tool_call.copy()

        # Create a simple hash
        hash_str = json.dumps(normalized, sort_keys=True)
        return hashlib.md5(hash_str.encode()).hexdigest()[:8]
