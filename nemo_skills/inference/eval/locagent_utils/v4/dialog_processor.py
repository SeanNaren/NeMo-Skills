import asyncio
import copy
import json
import logging
import os
import re
from typing import Dict, List, Optional

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string.
    
    Uses different heuristics for code vs natural language:
    - Code: ~3.5 characters per token (accounts for operators, keywords, indentation)
    - English text: ~4 characters per token
    - Mixed content: weighted average based on code indicators
    """
    if not text:
        return 0
    
    # Indicators that suggest code content
    code_indicators = [
        'def ', 'class ', 'import ', 'from ', 'return ', 'if ', 'elif ', 'else:',
        'for ', 'while ', 'try:', 'except:', 'finally:', 'with ',
        '()', '[]', '{}', '==', '!=', '<=', '>=', '//', '"""', "'''",
        '    ',  # indentation
    ]
    
    # Count code-like patterns
    code_score = 0
    text_lower = text.lower()
    for indicator in code_indicators:
        code_score += text_lower.count(indicator)
    
    # Estimate proportion of code vs text
    # Higher score = more code-like
    code_ratio = min(1.0, code_score / (len(text) / 100))  # Normalize by text length
    
    # Also check for common code patterns
    import_count = text.count('import ')
    def_count = text.count('def ')
    indent_count = text.count('\n    ')  # 4-space indentation
    
    # Adjust code ratio based on strong indicators
    if import_count > 2 or def_count > 2 or indent_count > 10:
        code_ratio = min(1.0, code_ratio + 0.3)
    
    # Calculate tokens based on content type
    # Code tends to have shorter tokens (more operators, keywords)
    # Pure code: ~3.5 chars/token
    # Pure English: ~4.5 chars/token  
    # (slightly higher than 4 to be conservative and avoid underestimation)
    chars_per_token = 4.5 - (code_ratio * 1.0)  # Range: 3.5 to 4.5
    
    estimated_tokens = int(len(text) / chars_per_token)
    
    # Add a small buffer (5%) to be conservative and avoid underestimating
    return int(estimated_tokens * 1.05)


def estimate_dialogue_tokens(turns: List[dict]) -> int:
    """Estimate the total number of tokens in the dialogue history."""
    total_tokens = 0
    
    for i, turn in enumerate(turns):
        try:
            # Count tokens in inputs
            if 'inputs' in turn and turn['inputs']:
                total_tokens += estimate_tokens(str(turn['inputs']))
            
            # Count tokens in assistant responses
            if 'assistant' in turn and turn['assistant']:
                total_tokens += estimate_tokens(str(turn['assistant']))
                
            # Count tokens in tool calls (sent as JSON)
            if 'tool_call' in turn and turn['tool_call']:
                tool_call_str = str(turn['tool_call'])
                total_tokens += estimate_tokens(tool_call_str)
                
            # Count tokens in tool outputs 
            if 'tool_output' in turn and turn['tool_output']:
                total_tokens += estimate_tokens(str(turn['tool_output']))
                
        except Exception as e:
            LOG.error(f"Error estimating tokens for turn {i}: {e}")
            LOG.error(f"Turn {i} type: {type(turn)}")
            if isinstance(turn, dict):
                LOG.error(f"Turn {i} keys: {list(turn.keys())}")
            else:
                LOG.error(f"Turn {i} value: {turn}")
            raise
        
        # Count tokens in raw assistant response (if different from main response)
        if 'assistant_raw' in turn and turn.get('assistant_raw') != turn.get('assistant'):
            total_tokens += estimate_tokens(str(turn['assistant_raw']))
    
    # Add reasonable overhead for formatting
    # System prompt and XML structure adds some overhead but not massive amounts
    overhead = len(turns) * 50  # Overhead per turn for role tags, formatting
    return total_tokens + overhead


def truncate_dialogue_history(turns: List[dict], max_seq_length: int, tokens_to_generate: int) -> List[dict]:
    """Truncate dialogue history to fit within context limits.
    
    Strategy: Remove oldest [assistant_output, tool_output] pairs while preserving:
    1. The initial user message (turn 0 inputs)
    2. Conversation continuity
    
    Args:
        turns: List of dialogue turns
        max_seq_length: Maximum context length in tokens
        tokens_to_generate: Tokens to reserve for the next generation
        
    Returns:
        Truncated list of turns
    """
    if not turns:
        return turns
    
    target_tokens = max_seq_length - tokens_to_generate
    current_tokens = estimate_dialogue_tokens(turns)
    
    if current_tokens <= target_tokens:
        return turns
    
    LOG.info(f"Dialogue history has ~{current_tokens} tokens, exceeding target of {target_tokens}")
    LOG.debug(f"Initial turn structure before truncation: {len(turns)} turns")
    for i, turn in enumerate(turns[:3]):  # Log first 3 turns
        if isinstance(turn, dict):
            has_tool_call = 'tool_call' in turn and turn['tool_call']
            assistant_len = len(str(turn.get('assistant', '')))
            LOG.debug(f"Turn {i}: has_tool_call={has_tool_call}, assistant_len={assistant_len}")
    
    # Deep copy to avoid modifying the original
    truncated_turns = copy.deepcopy(turns)
    
    # Track how many turn pairs we've removed
    removed_pairs = 0
    
    # Remove oldest complete turn pairs until we fit in context
    while estimate_dialogue_tokens(truncated_turns) > target_tokens:
        if len(truncated_turns) <= 1:
            # Only the initial user message remains, can't remove more
            LOG.warning("Cannot truncate further - only initial user message remains")
            break
        
        # Strategy: Remove the oldest complete assistant+tool pair
        # This means finding a turn with an assistant response followed by a tool output turn
        # Skip turn 0 (initial problem + first assistant response) if possible
        
        found_pair = False
        start_idx = 1 if len(truncated_turns) > 2 else 0  # Try to preserve turn 0
        for i in range(start_idx, len(truncated_turns) - 1):
            # Check if turn i has an assistant response and turn i+1 exists
            try:
                # Add debugging for turn structure
                if not isinstance(truncated_turns[i], dict):
                    LOG.error(f"Turn {i} in truncate_dialogue_history is not a dict: {type(truncated_turns[i])}")
                    continue
                    
                if 'assistant' in truncated_turns[i] and i + 1 < len(truncated_turns):
                    # If this turn has a tool_call, we should preserve a summary of the assistant response
                    # Otherwise the turn structure becomes confusing (tool call without explanation)
                    if 'tool_call' in truncated_turns[i] and truncated_turns[i]['tool_call']:
                        # For turn 0, always keep more context as it's the initial reasoning
                        max_chars = 300 if i == 0 else 200
                        
                        # Keep a brief explanation of why the tool was called
                        assistant_text = str(truncated_turns[i].get('assistant', ''))
                        if len(assistant_text) > max_chars:
                            # Truncate to preserve the reasoning
                            truncated_turns[i]['assistant'] = assistant_text[:max_chars] + "... [truncated for context]"
                            LOG.info(f"Truncated assistant response in turn {i} to preserve tool call context")
                        else:
                            # Keep the full assistant response if it's already short
                            LOG.info(f"Keeping short assistant response in turn {i} with tool call")
                    else:
                        # No tool call, safe to clear the assistant response
                        LOG.info(f"Removing assistant response from turn {i}")
                        truncated_turns[i]['assistant'] = ""  # Keep key but clear content
                    
                    # Always remove raw fields to save space
                    if 'assistant_raw' in truncated_turns[i]:
                        del truncated_turns[i]['assistant_raw']
                    if 'assistant_raw_w_think' in truncated_turns[i]:
                        del truncated_turns[i]['assistant_raw_w_think']
                    
                    # Remove the entire next turn (tool output)
                    LOG.info(f"Removing turn {i+1} (tool output)")
                    truncated_turns.pop(i + 1)
                    
                    removed_pairs += 1
                    found_pair = True
                    break
            except Exception as e:
                LOG.error(f"Error checking turn {i} in truncate_dialogue_history: {e}")
                LOG.error(f"Turn {i}: {truncated_turns[i] if i < len(truncated_turns) else 'index out of range'}")
                raise
        
        if not found_pair:
            # No more complete pairs to remove
            LOG.warning("No more complete assistant+tool pairs to remove")
            break
    
    final_tokens = estimate_dialogue_tokens(truncated_turns)
    LOG.info(f"Truncated dialogue to ~{final_tokens} tokens, removed {removed_pairs} assistant+tool pairs")
    
    # Ensure all turns have required fields
    for i, turn in enumerate(truncated_turns):
        if isinstance(turn, dict):
            # Ensure 'assistant' field exists (even if empty)
            if 'assistant' not in turn:
                turn['assistant'] = ""
                LOG.debug(f"Added missing 'assistant' field to turn {i}")
    
    return truncated_turns


def summarize_text(text: str, max_sentences: int = 5) -> str:
    """Summarize text to a maximum number of sentences.
    
    This function extracts the most important sentences from the text,
    prioritizing:
    1. Tool calls and their parameters
    2. File paths and locations
    3. Error messages
    4. Search queries and results summary
    5. Key findings or conclusions
    
    Args:
        text: The text to summarize
        max_sentences: Maximum number of sentences in the summary
        
    Returns:
        Summarized text
    """
    if not text or not text.strip():
        return text
    
    # Convert to string if it's a dict (for tool calls, etc.)
    if isinstance(text, dict):
        text = json.dumps(text, indent=2)
    
    # Split into sentences (rough approximation)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    # If already short enough, return as is
    if len(sentences) <= max_sentences:
        return text
    
    # Prioritize sentences based on content
    priority_sentences = []
    regular_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # High priority: tool-related information
        if any(keyword in sentence.lower() for keyword in [
            'tool:', 'path:', 'query:', 'error:', 'file:', 'found', 'result',
            'view_range', 'search', 'class', 'function', 'line', 'snippet'
        ]):
            priority_sentences.append(sentence)
        else:
            regular_sentences.append(sentence)
    
    # Build summary
    summary_sentences = []
    
    # Add priority sentences first (up to max_sentences)
    for sentence in priority_sentences[:max_sentences]:
        summary_sentences.append(sentence)
    
    # Fill remaining space with regular sentences
    remaining_slots = max_sentences - len(summary_sentences)
    if remaining_slots > 0 and regular_sentences:
        # Take first and last regular sentences if we have space
        if remaining_slots >= 2 and len(regular_sentences) > 1:
            summary_sentences.append(regular_sentences[0])
            summary_sentences.append(regular_sentences[-1])
        else:
            summary_sentences.extend(regular_sentences[:remaining_slots])
    
    # If we have too many priority sentences, truncate and add ellipsis
    if len(priority_sentences) > max_sentences:
        summary_sentences = summary_sentences[:max_sentences-1]
        summary_sentences.append("... (content truncated)")
    
    return ' '.join(summary_sentences)


class LLMSummarizer:
    """Handles LLM-based summarization using the parent's LLM instance."""
    
    def __init__(self, llm_instance=None, model_config: Optional[dict] = None):
        """Initialize the summarizer with the parent's LLM instance.
        
        Args:
            llm_instance: The LLM instance from the parent GenerationPipeline
            model_config: Configuration containing summarization settings
        """
        self.llm = llm_instance
        self.model_config = model_config or {}
        self.is_initialized = llm_instance is not None
        
        # Store generation parameters for summarization
        # Match the parameter structure used by the main generation task
        self.generation_params = {
            'temperature': self.model_config.get('summarization_temperature', 0.1),
            'top_p': 0.95,
            'top_k': 0,  # Match main task default
            'tokens_to_generate': self.model_config.get('summarization_max_tokens', 300),  # Increased default to prevent truncation
            'stop_phrases': [],  # Empty list for summaries
            'repetition_penalty': 1.0,  # Match main task default
        }
        
        if self.is_initialized:
            LOG.info(f"LLM summarizer initialized using parent LLM instance of type: {type(llm_instance).__name__}")
            LOG.info(f"LLM has generate_asyncio method: {hasattr(llm_instance, 'generate_asyncio')}")
            LOG.info(f"Summarization parameters: {self.generation_params}")
        else:
            LOG.warning("LLM summarizer NOT initialized - llm_instance is None")
    

    

    
    async def summarize_text_async(self, text: str, context_type: str = "general") -> str:
        """Summarize text using the parent's LLM instance.
        
        Args:
            text: Text to summarize
            context_type: Type of content (e.g., "tool_output", "code_snippet", "search_results")
            
        Returns:
            Summarized text
        """
        if not self.is_initialized or not self.llm:
            # No fallback - raise an error to understand why
            raise RuntimeError(f"LLM summarizer not initialized: is_initialized={self.is_initialized}, llm={self.llm is not None}")
        
        try:
            # Validate inputs
            if not text or not text.strip():
                LOG.warning("Empty text provided for summarization")
                return ""  # Return empty string for empty input
            
            if not self.llm:
                raise RuntimeError("No LLM instance available for summarization")
            
            # Create a focused prompt based on context type
            if context_type == "assistant_reasoning":
                prompt = f"""<IMPORTANT>
You are being asked to SUMMARIZE text, NOT to make tool calls or search for anything.
Your ONLY job is to create a brief summary of the assistant's reasoning below.
DO NOT output any <tool_call> blocks. Only output plain text summary.
</IMPORTANT>

Summarize the assistant's key insights and conclusions (2-3 short sentences MAX). Focus on what the assistant learned and its current hypothesis.

Assistant's reasoning:
{text}

Summary of key insights:"""
            elif context_type == "search_results":
                prompt = f"""<IMPORTANT>
You are being asked to SUMMARIZE text, NOT to make tool calls or search for anything.
Your ONLY job is to create a brief summary of the content below.
DO NOT output any <tool_call> blocks. Only output plain text summary.
</IMPORTANT>

Create a strategic summary (2-3 short sentences MAX) of these search results. Focus on what the results tell us about the codebase structure and where to investigate next.

Search results:
{text}

Strategic summary (what was found and what it means for the investigation):"""
            else:
                prompt = f"""<IMPORTANT>
You are being asked to SUMMARIZE text, NOT to make tool calls or search for anything.
Your ONLY job is to create a brief summary of the content below.
DO NOT output any <tool_call> blocks. Only output plain text summary.
</IMPORTANT>

Create a strategic summary (2-3 short sentences MAX) that captures the key insights and what they mean for the investigation.

Content:
{text}

Strategic summary:"""
            
            LOG.info(f"Calling LLM for summarization with prompt length: {len(prompt)} chars")
            LOG.debug(f"Generation params: {self.generation_params}")
            LOG.debug(f"First 500 chars of prompt: {prompt[:500]}...")
            LOG.debug(f"Last 200 chars of prompt: ...{prompt[-200:]}")
            
            # Check if prompt is too long
            if len(prompt) > 50000:  # Arbitrary limit, adjust as needed
                LOG.warning(f"Prompt is very long ({len(prompt)} chars), this might cause issues")
            
            # Use the parent's LLM instance to generate summary
            try:
                LOG.debug("About to call self.llm.generate_asyncio")
                generation_results = await self.llm.generate_asyncio(
                    prompts=[prompt],
                    **self.generation_params
                )
                LOG.debug("Completed self.llm.generate_asyncio call")
            except Exception as gen_error:
                LOG.error(f"LLM generate_asyncio raised exception: {gen_error}")
                LOG.error(f"Exception type: {type(gen_error).__name__}")
                raise
            
            LOG.debug(f"LLM generation_results type: {type(generation_results)}")
            
            # Handle both dict (single result) and list (batch results) formats
            if isinstance(generation_results, dict):
                # Single result returned directly as dict
                result_item = generation_results
                LOG.debug("LLM returned single result as dict")
            elif isinstance(generation_results, list) and len(generation_results) > 0:
                # Batch result, take first item
                result_item = generation_results[0]
                LOG.debug("LLM returned batch results as list")
            elif generation_results == 0:
                raise RuntimeError("LLM returned 0 instead of generation results")
            else:
                raise RuntimeError(f"LLM returned unexpected format: {type(generation_results).__name__}, value: {generation_results}")
            
            LOG.debug(f"Result item type: {type(result_item)}, keys: {list(result_item.keys()) if isinstance(result_item, dict) else 'not a dict'}")
            
            # Check for inference error
            if isinstance(result_item, dict) and 'inference_error' in result_item:
                error_msg = result_item.get('inference_error', 'Unknown error')
                LOG.error(f"LLM returned inference error: {error_msg}")
                raise RuntimeError(f"LLM inference error: {error_msg}")
            
            # Extract the generated summary
            summary = result_item.get('generation', '')
            
            # Handle case where generation is 0.0 or 0 (error case)
            if summary == 0.0 or summary == 0:
                LOG.error(f"LLM returned numeric 0 as generation, full result: {result_item}")
                raise RuntimeError("LLM returned 0 as generation, indicating an error")
            
            summary = str(summary).strip()
            
            if summary:
                # Clean up the summary - remove any repeated instructions
                lines = summary.split('\n')
                # Remove any lines that look like instructions
                cleaned_lines = [line for line in lines if not line.strip().startswith(("Summarize", "Summary:", "Tool output:", "Search results:"))]
                if cleaned_lines:
                    summary = '\n'.join(cleaned_lines).strip()
                
                # Ensure summary isn't too long
                sentences = re.split(r'(?<=[.!?])\s+', summary)
                if len(sentences) > 5:
                    summary = ' '.join(sentences[:5])
                
                LOG.info(f"Successfully created LLM summary of {len(summary)} chars from {len(text)} chars")
                if not summary:
                    raise RuntimeError("LLM returned empty summary after cleaning")
                return summary
            else:
                LOG.error(f"Empty 'generation' field in LLM response. Full result: {result_item}")
                raise RuntimeError("LLM returned empty generation field")
                
        except Exception as e:
            LOG.error(f"LLM summarization failed with exception: {e}")
            import traceback
            LOG.error(f"Full traceback: {traceback.format_exc()}")
            # Re-raise the exception to understand what's happening
            raise
    

    
    def summarize_text_sync(self, text: str, context_type: str = "general") -> str:
        """Synchronous wrapper for summarize_text_async."""
        return asyncio.run(self.summarize_text_async(text, context_type))


# Global summarizer instance (will be initialized by locagent)
_llm_summarizer: Optional[LLMSummarizer] = None


def set_llm_summarizer(summarizer: LLMSummarizer):
    """Set the global LLM summarizer instance."""
    global _llm_summarizer
    _llm_summarizer = summarizer


async def summarize_text_with_llm(text: str, context_type: str = "general") -> str:
    """Summarize text using LLM - no fallback to rule-based."""
    LOG.info(f"\n=== LLM SUMMARIZATION CALL ===")
    LOG.info(f"Context type: {context_type}, Input text length: {len(text)} chars")
    LOG.debug(f"Input text preview: {text[:200]}..." if len(text) > 200 else f"Input text: {text}")
    LOG.info(f"Summarizer status: exists={_llm_summarizer is not None}, initialized={_llm_summarizer.is_initialized if _llm_summarizer else 'N/A'}")
    
    if _llm_summarizer and _llm_summarizer.is_initialized:
        LOG.info("Calling LLM summarizer...")
        try:
            result = await _llm_summarizer.summarize_text_async(text, context_type)
            LOG.info(f"LLM SUCCESS: returned {len(result) if result else 0} chars")
            LOG.debug(f"Summary preview: {result[:200]}..." if result and len(result) > 200 else f"Summary: {result}")
            LOG.info(f"=== END LLM SUMMARIZATION (SUCCESS) ===")
            return result
        except Exception as e:
            LOG.error(f"LLM FAILED: {e}", exc_info=True)
            LOG.info(f"=== END LLM SUMMARIZATION (FAILED) ===")
            raise
    else:
        error_msg = f"LLM summarizer not available: _llm_summarizer={_llm_summarizer is not None}, initialized={_llm_summarizer.is_initialized if _llm_summarizer else 'N/A'}"
        LOG.error(error_msg)
        LOG.info(f"=== END LLM SUMMARIZATION (NOT AVAILABLE) ===")
        raise RuntimeError(error_msg)


async def summarize_turn_async(turn: dict, max_sentences: int = 5, is_initial_turn: bool = False, min_content_length: int = 500) -> dict:
    """Summarize a dialogue turn to reduce token usage (async version).
    
    Args:
        turn: The dialogue turn to summarize
        max_sentences: Maximum sentences per field
        is_initial_turn: Whether this is the initial user problem statement
        min_content_length: Minimum content length to trigger summarization
        
    Returns:
        Summarized turn
    """
    LOG.debug(f"Summarizing turn with min_content_length={min_content_length}, is_initial_turn={is_initial_turn}")
    summarized_turn = {}
    
    # Handle inputs field
    if 'inputs' in turn:
        # Never summarize the initial problem statement
        if is_initial_turn:
            summarized_turn['inputs'] = turn['inputs']
        else:
            # This is a tool output - check if it's a string that needs summarization
            input_content = turn['inputs']
            if isinstance(input_content, str) and len(input_content) > min_content_length:  # Only summarize long content
                # Determine context type based on content
                context_type = "tool_output"
                if "Search results for:" in input_content:
                    context_type = "search_results"
                
                # Use LLM summarization
                summary = await summarize_text_with_llm(input_content, context_type)
                # Add marker to indicate this is a summary
                summarized_turn['inputs'] = f"[STRATEGIC SUMMARY: {context_type}]\n{summary}\n[END SUMMARY]"
                summarized_turn['inputs_summary'] = True
                summarized_turn['_inputs_original_length'] = len(input_content)
            else:
                # Keep as-is if it's not a string or is short
                LOG.debug(f"Keeping inputs as-is (length={len(str(input_content))}, threshold={min_content_length})")
                summarized_turn['inputs'] = input_content
    
    # Handle assistant responses
    if 'assistant' in turn and turn['assistant'] is not None:
        assistant_content = turn['assistant']
        
        # Check if it's a tool call structure
        if isinstance(assistant_content, dict):
            if assistant_content.get('type') == 'tool_calls':
                # Keep tool call structure as-is
                summarized_turn['assistant'] = assistant_content
            elif assistant_content.get('type') == 'locations':
                # Keep location predictions as-is
                summarized_turn['assistant'] = assistant_content
            else:
                # Other dict structures - summarize their string representation
                content_str = str(assistant_content)
                if len(content_str) > min_content_length:  # Only summarize long content
                    summary = await summarize_text_with_llm(content_str, "general")
                    summarized_turn['assistant'] = f"[SUMMARIZED assistant_response]\n{summary}\n[END SUMMARY]"
                    summarized_turn['assistant_summary'] = True
                    summarized_turn['_assistant_original_length'] = len(content_str)
                else:
                    summarized_turn['assistant'] = assistant_content
        else:
            # String or other content - check length
            content_str = str(assistant_content)
            if len(content_str) > min_content_length:  # Only summarize long content
                summary = await summarize_text_with_llm(content_str, "general")
                summarized_turn['assistant'] = f"[SUMMARIZED assistant_response]\n{summary}\n[END SUMMARY]"
                summarized_turn['assistant_summary'] = True
                summarized_turn['_assistant_original_length'] = len(content_str)
            else:
                summarized_turn['assistant'] = assistant_content
    
    # Handle tool_call field (keep as-is, it's structured data)
    if 'tool_call' in turn:
        summarized_turn['tool_call'] = turn['tool_call']
    
    # Handle tool_output field (summarize if long)
    if 'tool_output' in turn and turn['tool_output']:
        tool_output_content = str(turn['tool_output'])
        if len(tool_output_content) > min_content_length:
            summary = await summarize_text_with_llm(tool_output_content, "tool_output")
            summarized_turn['tool_output'] = f"[STRATEGIC SUMMARY: tool_output]\n{summary}\n[END SUMMARY]"
            summarized_turn['tool_output_summary'] = True
            summarized_turn['_tool_output_original_length'] = len(tool_output_content)
        else:
            summarized_turn['tool_output'] = turn['tool_output']
    
    # Handle locations field if present
    if 'locations' in turn:
        summarized_turn['locations'] = turn['locations']
    
    # Copy over any additional raw fields (like assistant_raw, assistant_raw_w_think)
    for key in turn:
        if key.endswith('_raw') or key.endswith('_raw_w_think'):
            # Don't include raw fields in summarized turns to save space
            pass
        elif key not in summarized_turn:
            # Copy over any other fields we haven't explicitly handled
            summarized_turn[key] = turn[key]
    
    # Don't include raw versions in summarized history
    # But keep a marker that this turn was summarized
    summarized_turn['summarized'] = True
    
    return summarized_turn


def summarize_turn(turn: dict, max_sentences: int = 5, is_initial_turn: bool = False) -> dict:
    """Synchronous wrapper for summarize_turn_async."""
    return asyncio.run(summarize_turn_async(turn, max_sentences, is_initial_turn))


async def create_investigation_summary(turns: list, max_sentences: int = 5) -> str:
    """Create a summary of the investigation so far.
    
    This summarizes what the model has been looking for across all turns,
    focusing on the key files, functions, and patterns explored.
    
    Args:
        turns: List of dialogue turns (excluding the most recent)
        max_sentences: Maximum sentences in the summary
        
    Returns:
        Investigation summary string
    """
    if not turns or len(turns) <= 1:
        LOG.info(f"=== INVESTIGATION SUMMARY: Not enough turns ({len(turns) if turns else 0}) ===")
        return ""
    
    LOG.info(f"=== INVESTIGATION SUMMARY START ===")
    LOG.info(f"Creating summary for {len(turns)} turns, max_sentences={max_sentences}")
    
    # Collect all the investigation actions
    investigation_parts = []
    
    LOG.info(f"Processing {len(turns)-1} turns (excluding most recent)")
    for i, turn in enumerate(turns[:-1]):  # Exclude the most recent turn
        if i == 0:
            # Skip the initial problem statement turn
            LOG.debug(f"Turn {i}: Skipping initial problem statement")
            continue
            
        # Focus on capturing the assistant's reasoning, hypotheses, and conclusions
        if 'assistant' in turn and turn['assistant']:
            assistant_text = str(turn['assistant']).strip()
            LOG.debug(f"Turn {i}: Assistant response length: {len(assistant_text)}")
            
            if assistant_text:
                # Capture the full assistant reasoning (not just first 300 chars)
                # This is what we want to summarize - the thinking process
                if len(assistant_text) > 800:
                    part = f"Turn {i}: {assistant_text[:800]}..."
                else:
                    part = f"Turn {i}: {assistant_text}"
                investigation_parts.append(part)
                LOG.debug(f"Turn {i}: Added full assistant reasoning")
                
        # Just note what action was taken (tool call) without the output
        # The assistant's interpretation of the output is what matters
        if 'tool_call' in turn and turn['tool_call']:
            tool_type = turn['tool_call'].get('tool', 'unknown')
            action_note = None
            
            if tool_type == 'view_file':
                path = turn['tool_call'].get('path', 'unknown')
                view_range = turn['tool_call'].get('view_range', '')
                if view_range:
                    action_note = f"[Viewed {path} lines {view_range}]"
                else:
                    action_note = f"[Viewed {path}]"
            elif tool_type == 'codebase_search':
                query = turn['tool_call'].get('query', 'unknown')
                action_note = f"[Searched: '{query}']"
            elif tool_type == 'connected_tree':
                file = turn['tool_call'].get('file', 'repository')
                action_note = f"[Examined connections: {file}]"
            elif tool_type == 'repo_tree':
                action_note = f"[Viewed repository structure]"
                
            if action_note:
                investigation_parts.append(action_note)
                LOG.debug(f"Turn {i}: Added action note: {action_note}")
                
        # Don't extract from tool outputs - we care about the assistant's 
        # interpretation in the next turn, not the raw data
    
    # Create the context for summarization
    investigation_text = "\n".join(investigation_parts)
    
    LOG.info(f"Collected {len(investigation_parts)} investigation parts")
    LOG.debug(f"Total investigation text length: {len(investigation_text)} chars")
    
    if not investigation_text.strip():
        LOG.warning("No investigation text to summarize")
        LOG.info(f"=== INVESTIGATION SUMMARY END (EMPTY) ===")
        return ""
    
    # Use LLM to create a focused summary
    prompt = f"""<IMPORTANT>
You are being asked to CREATE A SUMMARY, NOT to make tool calls or search for anything.
Your ONLY job is to summarize the investigation progress below.
DO NOT output any <tool_call> blocks. Only output plain text summary.
</IMPORTANT>

Summarize the assistant's reasoning and thought process from this investigation in {max_sentences} short sentences MAX. Focus on capturing the assistant's hypotheses, conclusions, and strategic thinking.

Focus on:
- What the assistant thinks is the root cause or issue
- The assistant's current hypothesis or theory about the problem
- What the assistant concluded from examining files/search results
- The assistant's strategy and what it plans to investigate next
- Key insights or patterns the assistant noticed

DO NOT just list actions taken. Summarize the THINKING and REASONING.

Assistant's investigation reasoning:
{investigation_text}

Summary of assistant's thought process (max {max_sentences} sentences):"""
    
    try:
        LOG.info(f"Calling LLM for investigation summary, prompt length: {len(prompt)} chars")
        LOG.debug(f"Investigation summary prompt:\n{prompt[:500]}..." if len(prompt) > 500 else f"Investigation summary prompt:\n{prompt}")
        summary = await summarize_text_with_llm(prompt, "investigation_summary")
        LOG.info(f"LLM returned summary: {len(summary)} chars")
        LOG.debug(f"Summary content: {summary[:300]}..." if len(summary) > 300 else f"Summary content: {summary}")
        LOG.info(f"=== INVESTIGATION SUMMARY END (SUCCESS) ===")
        return summary.strip()
    except Exception as e:
        LOG.error(f"Failed to create investigation summary: {e}", exc_info=True)
        LOG.info(f"=== INVESTIGATION SUMMARY END (FAILED) ===")
        # Return empty string instead of failing
        return ""


async def apply_context_summarization(turns: list, max_sentences: int = 5) -> list:
    """Apply context-aware summarization to reduce token usage.
    
    This creates an investigation summary for older turns while keeping
    the most recent tool output intact.
    
    Args:
        turns: List of dialogue turns
        max_sentences: Maximum sentences in the investigation summary
        
    Returns:
        List of turns with context summarization applied
    """
    LOG.info(f"=== CONTEXT SUMMARIZATION START ===")
    LOG.info(f"Input: {len(turns)} turns, max_sentences={max_sentences}")
    
    # Log initial token count
    original_tokens = estimate_dialogue_tokens(turns)
    LOG.info(f"Original dialogue tokens: ~{original_tokens}")
    
    if not turns or len(turns) <= 2:
        LOG.info(f"Not enough turns for summarization (need > 2, have {len(turns)})")
        LOG.info(f"=== CONTEXT SUMMARIZATION END (SKIPPED) ===")
        return turns
    
    import copy
    result_turns = []
    
    # Always keep the first turn (problem statement) as-is
    result_turns.append(copy.deepcopy(turns[0]))
    LOG.info("Kept turn 0 (problem statement)")
    
    # Create investigation summary of all turns except the last one
    investigation_summary = await create_investigation_summary(turns, max_sentences)
    
    if investigation_summary:
        # Add the investigation summary as a special turn
        # This captures the assistant's reasoning and thought process from earlier turns
        summary_content = f"[ASSISTANT'S REASONING FROM EARLIER INVESTIGATION]\n{investigation_summary}\n[END REASONING SUMMARY]"
        summary_turn = {
            'inputs': summary_content,
            'assistant': '',  # Empty but present
            'tool_call': None,
            'tool_output': '',
            '_is_summary': True
        }
        result_turns.append(summary_turn)
        LOG.info(f"Added assistant reasoning summary: {len(summary_content)} chars")
        LOG.debug(f"Summary content preview: {summary_content[:200]}..." if len(summary_content) > 200 else f"Summary content: {summary_content}")
    else:
        LOG.warning("No investigation summary was created")
    
    # Add the most recent turn(s) in full
    # Always keep the last turn, and if it's a tool output, also keep the assistant request that triggered it
    if len(turns) >= 2:
        last_turn = turns[-1]
        
        # Log last turn structure for debugging
        last_turn_info = {
            'index': len(turns) - 1,
            'has_inputs': bool(last_turn.get('inputs')),
            'has_assistant': bool(last_turn.get('assistant')),
            'has_tool_call': bool(last_turn.get('tool_call')),
            'has_tool_output': bool(last_turn.get('tool_output')),
            'inputs_len': len(str(last_turn.get('inputs', ''))),
            'assistant_len': len(str(last_turn.get('assistant', ''))),
            'tool_output_len': len(str(last_turn.get('tool_output', '')))
        }
        LOG.info(f"Last turn analysis: {last_turn_info}")
        
        # Check if we need to include the previous turn as well
        include_previous = False
        if 'tool_output' in last_turn and last_turn.get('tool_output'):
            # This is a tool output turn, we should include the assistant request that triggered it
            include_previous = True
            LOG.info("Will include previous turn (last turn has tool output)")
        elif 'inputs' in last_turn and last_turn.get('inputs') and not last_turn.get('assistant'):
            # This is a new user input, keep the full turn
            include_previous = False
            LOG.info("Will NOT include previous turn (last turn is user input)")
        
        if include_previous and len(turns) > 2:
            # Include the second-to-last turn (the assistant request)
            second_last_turn = turns[-2]
            result_turns.append(copy.deepcopy(second_last_turn))
            LOG.info("Kept turn -2 (assistant request before tool output)")
        
        # Always include the last turn
        result_turns.append(copy.deepcopy(last_turn))
        LOG.info("Kept turn -1 (most recent)")
    
    # Calculate and log token reduction
    final_tokens = estimate_dialogue_tokens(result_turns)
    reduction_pct = ((original_tokens - final_tokens) / original_tokens * 100) if original_tokens > 0 else 0
    
    LOG.info(f"Summarization result: {len(turns)} turns -> {len(result_turns)} turns")
    LOG.info(f"Token reduction: ~{original_tokens} -> ~{final_tokens} ({reduction_pct:.1f}% reduction)")
    LOG.info(f"=== CONTEXT SUMMARIZATION END ===")
    
    return result_turns


class DialogProcessor:
    """Processes dialog output to extract tool calls and locations."""

    @staticmethod
    def extract_response(dialog_text: str, config=None):
        # First check if this is a Python dict representation (single quotes)
        # This handles cases like: {'type': 'locations', 'locations': '...'}
        if dialog_text.strip().startswith("{") and dialog_text.strip().endswith("}"):
            extracted = DialogProcessor._extract_python_dict(dialog_text)
            if extracted:
                return extracted
        
        # Standard tag-based extraction
        if "<tool_call>" in dialog_text:
            LOG.info("Found <tool_call> block in output")
            return DialogProcessor._extract_tool_calls(dialog_text)
        elif "<locations>" in dialog_text:
            LOG.info("Found <locations> block in output")
            return DialogProcessor._extract_locations(dialog_text)
        else:
            # Check if implicit tool detection is enabled
            enable_implicit = getattr(config, 'enable_implicit_tool_detection', True) if config else True
            if enable_implicit:
                LOG.warning("No <tool_call> or <locations> found, checking for implicit tool calls")
                return DialogProcessor._extract_implicit_tool_calls(dialog_text, config)
            else:
                LOG.warning("No <tool_call> or <locations> found, and implicit tool detection is disabled")
                return None

    # USED
    @staticmethod
    def _extract_tool_calls(dialog_text: str) -> Dict:
        """Extract tool calls from dialog text with unified handling of various formats.
        
        Supports:
        1. Standard format: <tool_call>{...}</tool_call>
        2. Unclosed tags: <tool_call>{...
        3. Malformed JSON with extra braces
        4. Two JSON formats:
           - Format 1: {"tool": "view_file", "path": "...", ...}
           - Format 2: {"tool_name": {...params...}}
        """
        
        # Step 1: Extract raw content between tags (or after unclosed tag)
        raw_content = DialogProcessor._extract_tag_content(dialog_text, "tool_call")
        if not raw_content:
            return None
            
        # Step 2: Parse JSON with error recovery
        tool_data = DialogProcessor._parse_json_with_recovery(raw_content, "tool_call")
        if not tool_data:
            return None
            
        # Step 3: Normalize to unified format
        return DialogProcessor._normalize_tool_format(tool_data)
    
    @staticmethod
    def _extract_tag_content(text: str, tag_name: str) -> str:
        """Extract content from XML-style tags with fallback for unclosed tags."""
        # Try properly closed tag first
        pattern = f"<{tag_name}>(.*?)</{tag_name}>"
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # Fallback: unclosed tag - extract everything after the tag
        pattern = f"<{tag_name}>\\s*(.*?)$"
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            LOG.warning(f"Detected unclosed <{tag_name}> tag")
            return match.group(1).strip()
            
        return None
    
    @staticmethod
    def _parse_json_with_recovery(json_str: str, context: str = "") -> dict:
        """Parse JSON with recovery strategies for common issues."""
        if not json_str:
            return None
            
        LOG.debug(f"Attempting to parse JSON for {context} (first 200 chars): {json_str[:200]}...")
        
        # Strategy 1: Parse as-is
        try:
            data = json.loads(json_str)
            LOG.debug(f"Successfully parsed JSON as-is for {context}")
            return data
        except json.JSONDecodeError as e:
            LOG.debug(f"Initial parse failed for {context}: {e}")
        
        # Strategy 2: Find complete JSON by brace matching
        complete_json = DialogProcessor._extract_complete_json(json_str)
        if complete_json and complete_json != json_str:
            try:
                data = json.loads(complete_json)
                LOG.debug(f"Successfully parsed after brace matching for {context}")
                return data
            except json.JSONDecodeError as e:
                LOG.debug(f"Brace-matched parse failed for {context}: {e}")
        
        # Strategy 3: Common fixes
        # Fix extra closing braces (e.g., }})
        if json_str.rstrip().endswith('}}') and json_str.count('}') > json_str.count('{'):
            fixed = json_str.rstrip()[:-1]
            try:
                data = json.loads(fixed)
                LOG.debug(f"Successfully parsed after removing extra brace for {context}")
                return data
            except json.JSONDecodeError:
                pass
        
        # Fix trailing content after last valid }
        last_brace = json_str.rfind('}')
        if last_brace > 0 and last_brace < len(json_str) - 1:
            trimmed = json_str[:last_brace + 1]
            try:
                data = json.loads(trimmed)
                LOG.debug(f"Successfully parsed after trimming trailing content for {context}")
                return data
            except json.JSONDecodeError:
                pass
        
        LOG.warning(f"All JSON parsing attempts failed for {context}")
        return None
    
    @staticmethod
    def _extract_complete_json(text: str) -> str:
        """Extract complete JSON object by matching braces, handling strings properly."""
        if not text or not text.strip().startswith('{'):
            return None
            
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\' and in_string:
                escape_next = True
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
                
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return text[:i + 1]
        
        return None
    
    @staticmethod
    def _normalize_tool_format(tool_data: dict) -> dict:
        """Normalize different tool formats to a unified structure."""
        if not isinstance(tool_data, dict):
            LOG.warning(f"Tool data is not a dictionary: {type(tool_data)}")
            return None
            
        # Format 1: {"tool": "view_file", "path": "...", ...}
        # This is our target format, just wrap it
        if "tool" in tool_data:
            LOG.debug("Detected Format 1 tool call (with 'tool' key)")
            return {"type": "tool_calls", "tool_call": tool_data}
        
        # Format 2: {"tool_name": {...params...}}
        # Need to flatten this structure
        tool_names = list(tool_data.keys())
        if len(tool_names) == 1:
            tool_name = tool_names[0]
            params = tool_data[tool_name]
            
            if isinstance(params, dict):
                # Merge tool name with parameters
                unified_tool_call = {"tool": tool_name, **params}
                LOG.debug(f"Detected Format 2 tool call, normalized to: {unified_tool_call}")
                return {"type": "tool_calls", "tool_call": unified_tool_call}
            else:
                LOG.warning(f"Tool parameters are not a dictionary for tool '{tool_name}': {type(params)}")
        
        LOG.warning(f"Unrecognized tool format with keys: {list(tool_data.keys())}")
        return None
    
    @staticmethod
    def _extract_python_dict(dialog_text: str) -> Dict:
        """Extract response from Python dict representation (with single quotes).
        
        Handles cases like:
        {'type': 'locations', 'locations': 'file1.py:L10-L20\\nfile2.py:L30-L40'}
        """
        try:
            # Use ast.literal_eval to safely parse Python literals
            import ast
            data = ast.literal_eval(dialog_text.strip())
            
            if not isinstance(data, dict):
                return None
            
            LOG.debug(f"Successfully parsed Python dict: {data}")
            
            # Handle different response types
            if data.get('type') == 'locations':
                # Parse the locations string if it's a string
                locations = data.get('locations', '')
                if isinstance(locations, str):
                    # Parse location strings into structured format
                    parsed_locations = []
                    for line in locations.split('\n'):
                        line = line.strip()
                        if line:
                            location_match = re.match(r"([^:]+):L(\d+)-L(\d+)", line)
                            if location_match:
                                file_path, start_line, end_line = location_match.groups()
                                parsed_locations.append({
                                    "file_path": file_path,
                                    "start_line": int(start_line),
                                    "end_line": int(end_line),
                                    "raw": line
                                })
                    
                    return {"type": "locations", "locations": parsed_locations}
                else:
                    # Already structured
                    return data
            
            elif data.get('type') == 'tool_calls':
                # Already in correct format
                return data
            
            # Try to interpret as a tool call
            elif "tool" in data or len(data) == 1:
                return DialogProcessor._normalize_tool_format(data)
                
        except (ValueError, SyntaxError) as e:
            LOG.debug(f"Failed to parse as Python dict: {e}")
        
        return None

    # USED
    @staticmethod
    def _extract_locations(dialog_text: str) -> Dict:
        """Extract location predictions from dialog text."""
        # Use unified tag extraction
        locations_text = DialogProcessor._extract_tag_content(dialog_text, "locations")
        if not locations_text:
            return {"type": "locations", "locations": []}
        
        # Parse location lines
        locations = []
        for line in locations_text.split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                location_match = re.match(r"([^:]+):L(\d+)-L(\d+)", line)
                if location_match:
                    file_path, start_line, end_line = location_match.groups()
                    locations.append({
                        "file_path": file_path,
                        "start_line": int(start_line),
                        "end_line": int(end_line),
                        "raw": line
                    })
                else:
                    locations.append({"raw": line})

        return {"type": "locations", "locations": locations}

    # USED
    @staticmethod
    def _extract_implicit_tool_calls(dialog_text: str, config=None) -> Dict:
        """Extract tool calls that appear as JSON without <tool_call> wrapper."""
        # Check for JSON after </think> tag
        if "</think>" in dialog_text:
            # Split and clean up the text after </think>
            after_think = dialog_text.split("</think>")[1].strip()
            # Replace escaped newlines and multiple newlines with a single space
            after_think = re.sub(r"\\n|\n+", " ", after_think)

            # First try: Look for any JSON-like structure with a path field
            json_pattern = r"\{(?:[^{}]|\"[^\"]*\")*\"path\"(?:[^{}]|\"[^\"]*\")*\}"
            json_match = re.search(json_pattern, after_think)

            if json_match:
                try:
                    json_str = json_match.group(0)
                    # Clean up any remaining escapes or whitespace
                    json_str = json_str.replace("\\", "")
                    tool_data = json.loads(json_str)

                    # Determine tool type based on content
                    if "path" in tool_data:
                        tool_data["tool"] = "view_file"
                        # Add view_range if not present
                        if "view_range" not in tool_data:
                            tool_data["view_range"] = None
                    elif "query" in tool_data:
                        tool_data["tool"] = "codebase_search"
                        # Check if query is empty or whitespace
                        if not tool_data.get("query", "").strip():
                            LOG.warning(f"Empty query in codebase_search tool call")
                            return None
                    elif "repo_tree" in tool_data or len(tool_data) == 0:
                        tool_data["tool"] = "repo_tree"
                    else:
                        tool_data["tool"] = "unknown"
                    LOG.info(f"Found JSON tool call after </think>: {tool_data}")
                    return {"type": "tool_calls", "tool_call": tool_data}
                except json.JSONDecodeError as e:
                    LOG.warning(f"Failed to parse JSON after </think>: {e}")

            # Second try: Simple file path request
            # Look for quoted or unquoted file paths after </think>
            file_extensions = getattr(config, 'file_extensions', None) if config else None
            if file_extensions:
                # Create pattern from configured file extensions
                extensions_pattern = "|".join(re.escape(ext) for ext in file_extensions)
                file_path_pattern = rf'(?:\'|")?([^\'"\s]+?\.(?:{extensions_pattern}))(?:\'|")?'
            else:
                # Fallback to default extensions if none provided
                file_path_pattern = r'(?:\'|")?([^\'"\s]+?\.(?:py|cpp|h|hpp|java|js|ts|rb|go|rs|cs|php))(?:\'|")?'
            
            file_path_match = re.search(file_path_pattern, after_think)
            if file_path_match:
                tool_data = {
                    "tool": "view_file",
                    "path": file_path_match.group(1),
                    "view_range": None,
                }
                LOG.info(f"Found simple file path request: {tool_data['path']}")
                return {"type": "tool_calls", "tool_call": tool_data}

            # Third try: Simple search query request
            # Look for quoted or unquoted search terms after </think>
            # Common patterns: "search for X", "find X", "look for X", "X function", "X class"
            search_patterns = [
                r'search\s+for\s+(?:\'|")?([A-Za-z_][A-Za-z0-9_]*)(?:\'|")?',
                r'find\s+(?:\'|")?([A-Za-z_][A-Za-z0-9_]*)(?:\'|")?',
                r'look\s+for\s+(?:\'|")?([A-Za-z_][A-Za-z0-9_]*)(?:\'|")?',
                r'(?:\'|")?([A-Za-z_][A-Za-z0-9_]*)\s+function(?:\'|")?',
                r'(?:\'|")?([A-Za-z_][A-Za-z0-9_]*)\s+class(?:\'|")?',
                r'(?:\'|")?([A-Za-z_][A-Za-z0-9_]*)\s+method(?:\'|")?',
                r'(?:\'|")?([A-Za-z_][A-Za-z0-9_]*)\s+variable(?:\'|")?',
            ]

            for pattern in search_patterns:
                search_match = re.search(pattern, after_think, re.IGNORECASE)
                if search_match:
                    search_term = search_match.group(1).strip()
                    # Clean up the search term
                    search_term = re.sub(r'[\'"]', "", search_term)
                    if search_term and len(search_term) > 1:  # Ensure it's not just whitespace
                        tool_data = {
                            "tool": "codebase_search",
                            "query": search_term,
                        }
                        LOG.info(f"Found simple search request: {tool_data['query']}")
                        return {"type": "tool_calls", "tool_call": tool_data}

        # Look for specific tool call patterns
        # View tool: {"path": "...", "view_range": [...]}
        view_pattern = r'\{[^{}]*"path"[^{}]*(?:"view_range"[^{}]*)?}'
        # Repo tree tool: {} or {"repo_tree": true} or similar
        repo_tree_pattern = r'\{[^{}]*"repo_tree"[^{}]*\}'
        # Codebase search tool: {"query": "..."}
        codebase_search_pattern = r'\{[^{}]*"query"[^{}]*\}'

        # Try view tool first
        json_match = re.search(view_pattern, dialog_text)
        if json_match:
            try:
                tool_data = json.loads(json_match.group())
                tool_data["tool"] = "view_file"
                LOG.warning(f"Found implicit view tool call: {tool_data}")
                return {"type": "tool_calls", "tool_call": tool_data}
            except json.JSONDecodeError as e:
                LOG.warning(f"Failed to parse implicit view tool call JSON: {e}")

        # Try codebase search tool
        json_match = re.search(codebase_search_pattern, dialog_text)
        if json_match:
            try:
                tool_data = json.loads(json_match.group())
                tool_data["tool"] = "codebase_search"
                # Check if query is empty or whitespace
                if not tool_data.get("query", "").strip():
                    LOG.warning(f"Empty query in codebase_search tool call")
                else:
                    LOG.info(f"Found implicit codebase_search tool call: {tool_data}")
                    return {"type": "tool_calls", "tool_call": tool_data}
            except json.JSONDecodeError as e:
                LOG.warning(f"Failed to parse implicit codebase_search tool call JSON: {e}")

        # Try repo tree tool
        json_match = re.search(repo_tree_pattern, dialog_text)
        if json_match:
            try:
                tool_data = json.loads(json_match.group())
                tool_data["tool"] = "repo_tree"
                LOG.info(f"Found implicit repo_tree tool call: {tool_data}")
                return {"type": "tool_calls", "tool_call": tool_data}
            except json.JSONDecodeError as e:
                LOG.warning(f"Failed to parse implicit repo_tree tool call JSON: {e}")

        # Try simple search query detection in general dialog text
        # Look for common search patterns throughout the dialog
        search_patterns_general = [
            r'search\s+for\s+(?:\'|")?([A-Za-z_][A-Za-z0-9_]*)(?:\'|")?',
            r'find\s+(?:\'|")?([A-Za-z_][A-Za-z0-9_]*)(?:\'|")?',
            r'look\s+for\s+(?:\'|")?([A-Za-z_][A-Za-z0-9_]*)(?:\'|")?',
            r'(?:\'|")?([A-Za-z_][A-Za-z0-9_]*)\s+function(?:\'|")?',
            r'(?:\'|")?([A-Za-z_][A-Za-z0-9_]*)\s+class(?:\'|")?',
            r'(?:\'|")?([A-Za-z_][A-Za-z0-9_]*)\s+method(?:\'|")?',
            r'(?:\'|")?([A-Za-z_][A-Za-z0-9_]*)\s+variable(?:\'|")?',
            # Also look for standalone terms that might be function/class names
            r'(?:\'|")?([A-Za-z_][A-Za-z0-9_]*[A-Z][A-Za-z0-9_]*)(?:\'|")?',  # CamelCase
            r'(?:\'|")?([a-z_][a-z0-9_]*)(?:\'|")?',  # snake_case
            # Specific pattern for quoted identifiers
            r'["\']([A-Za-z_][A-Za-z0-9_]*[A-Za-z0-9_]*)["\']',  # Quoted identifiers
        ]

        for pattern in search_patterns_general:
            search_match = re.search(pattern, dialog_text, re.IGNORECASE)
            if search_match:
                search_term = search_match.group(1).strip()
                # Clean up the search term
                search_term = re.sub(r'[\'"]', "", search_term)
                # Filter out common words that shouldn't be searched
                common_words_list = getattr(config, 'common_words_filter', None) if config else None
                common_words = set(common_words_list)
                if (
                    search_term
                    and len(search_term) > 2  # Ensure it's not just a short word
                    and search_term.lower() not in common_words
                    and not search_term.isdigit()
                ):  # Don't search for pure numbers
                    tool_data = {
                        "tool": "codebase_search",
                        "query": search_term,
                    }
                    LOG.info(f"Found implicit search request: {tool_data['query']}")
                    return {"type": "tool_calls", 'tool_call': tool_data}

        # Check for empty JSON object (repo_tree tool)
        empty_json_match = re.search(r"^\s*\{\s*\}\s*$", dialog_text.strip())
        if empty_json_match:
            tool_data = {"tool": "repo_tree"}
            LOG.info(f"Found implicit empty repo_tree tool call: {tool_data}")
            return {"type": "tool_calls", "tool_call": tool_data}

        # Last resort: Check for dictionary format: {'type': 'tool_calls', 'tool_call': {...}}
        dict_pattern = r"\{'type':\s*'tool_calls',\s*'tool_call':\s*\{[^}]+\}\}"
        dict_match = re.search(dict_pattern, dialog_text)
        if dict_match:
            try:
                dict_str = dict_match.group(0)
                # Convert single quotes to double quotes for JSON parsing
                dict_str = dict_str.replace("'", '"')
                parsed_dict = json.loads(dict_str)
                
                if parsed_dict.get('type') == 'tool_calls' and 'tool_call' in parsed_dict:
                    tool_data = parsed_dict['tool_call']
                    LOG.warning(f"Detected strange format tool call (backup handling): {tool_data}. This should not happen often.")
                    return {"type": "tool_calls", "tool_call": tool_data}
            except json.JSONDecodeError as e:
                LOG.warning(f"Failed to parse dictionary format tool call: {e}")

        LOG.warning("No <tool_call>, <locations>, or implicit tool calls found in dialog output")
        return None
