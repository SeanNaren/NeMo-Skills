# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import os

import anthropic
import httpx

from nemo_skills.utils import get_logger_name

from .base import BaseModel, EndpointType
from .utils import trim_after_stop_phrases

LOG = logging.getLogger(get_logger_name(__file__))

# Transient errors that warrant a retry
_RETRYABLE_EXCEPTIONS = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
)


def _openai_messages_to_anthropic(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Split system messages out and convert the rest to Anthropic format.

    Returns (system_prompt, anthropic_messages).
    """
    system_parts: list[str] = []
    anthropic_msgs: list[dict] = []

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")

        if role in ("system", "developer"):
            system_parts.append(content if isinstance(content, str) else str(content))
            continue

        # Regular user/assistant message
        anthropic_msgs.append({"role": role, "content": content})

    system = "\n".join(system_parts) if system_parts else None
    return system, anthropic_msgs


class AnthropicModel(BaseModel):
    """Model backend for Anthropic Claude models with extended thinking support.

    Uses the Anthropic Python SDK directly to communicate with the API,
    enabling extended thinking via the ``thinking`` parameter. This is
    required when connecting through a proxy (e.g. NVIDIA infrastructure)
    that supports the native Anthropic Messages API.
    """

    MODEL_PROVIDER = "anthropic"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: str = "5000",
        model: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
        **kwargs,
    ):
        model = model or os.getenv("NEMO_SKILLS_ANTHROPIC_MODEL")
        self.model = model
        if model is None:
            raise ValueError("model argument is required for Anthropic model.")

        if base_url is None:
            base_url = os.getenv("NEMO_SKILLS_ANTHROPIC_BASE_URL", f"http://{host}:{port}")

        # Call super to set up shared infrastructure (tokenizer, semaphore, etc.)
        super().__init__(
            model=model,
            base_url=base_url,
            max_retries=max_retries,
            **kwargs,
        )

        # The Anthropic SDK appends /v1/messages internally, so strip /v1 if present
        anthropic_base_url = base_url.rstrip("/").removesuffix("/v1") if base_url else None

        api_key = self.litellm_kwargs["api_key"]
        self.anthropic_client = anthropic.AsyncAnthropic(
            api_key=api_key,
            base_url=anthropic_base_url,
            max_retries=max_retries,
        )

    def _get_api_key(self, api_key: str | None, api_key_env_var: str | None, base_url: str) -> str | None:
        api_key = super()._get_api_key(api_key, api_key_env_var, base_url)
        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("NVIDIA_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY or NVIDIA_API_KEY is required for Anthropic models.")
        return api_key

    def _build_completion_request_params(self, **kwargs) -> dict:
        raise NotImplementedError("Text completion is not supported by Anthropic models. Use chat endpoint.")

    def _build_chat_request_params(self, **kwargs) -> dict:
        raise NotImplementedError("AnthropicModel uses generate_async directly; this method is not used.")

    async def generate_async(
        self,
        prompt: str | list[dict],
        endpoint_type: EndpointType = None,
        tokens_to_generate: int | None = None,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = -1,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = None,
        stop_phrases: list[str] | None = None,
        top_logprobs: int | None = None,
        timeout: float | int | None = 14400,
        remove_stop_phrases: bool = True,
        stream: bool = False,
        reasoning_effort: str | None = None,
        tools: list[dict] | None = None,
        include_response: bool = False,
        extra_body: dict = None,
        response_format=None,
    ) -> dict:
        if not isinstance(prompt, list):
            raise ValueError("Anthropic models only support chat endpoint. Prompt must be a list of messages.")

        # Convert OpenAI-format messages to Anthropic format
        system, messages = _openai_messages_to_anthropic(prompt)

        # Build request kwargs
        create_kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": tokens_to_generate or 4096,
        }

        if system is not None:
            create_kwargs["system"] = system

        if stop_phrases:
            create_kwargs["stop_sequences"] = stop_phrases

        if reasoning_effort:
            budget_fractions = {
                "low": 0.25,
                "medium": 0.50,
                "high": 0.75,
                "max": 0.95,
            }
            fraction = budget_fractions.get(reasoning_effort, 0.75)
            budget_tokens = max(1024, int(create_kwargs["max_tokens"] * fraction))
            create_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget_tokens,
            }
            # Anthropic requires temperature=1.0 with extended thinking
            create_kwargs["temperature"] = 1.0
        else:
            create_kwargs["temperature"] = temperature
            if top_p != 0.95:
                create_kwargs["top_p"] = top_p

        if top_k > 0:
            create_kwargs["top_k"] = top_k

        if timeout:
            create_kwargs["timeout"] = timeout

        # Always stream to avoid 504 gateway timeouts on long-running
        # requests (especially with extended thinking).  The SDK helper
        # accumulates chunks and returns a complete Message object.
        # Retry on transient connection errors (proxy drops, incomplete reads).
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                async with self.concurrent_semaphore:
                    async with self.anthropic_client.messages.stream(**create_kwargs) as stream:
                        response = await stream.get_final_message()
                break
            except _RETRYABLE_EXCEPTIONS as exc:
                if attempt == max_retries:
                    raise
                wait = 2 ** attempt
                LOG.warning(
                    "Anthropic request failed (attempt %d/%d): %s. Retrying in %ds...",
                    attempt + 1,
                    max_retries + 1,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)

        # Parse the response
        result = self._parse_anthropic_response(response)

        if remove_stop_phrases and stop_phrases:
            result["generation"] = trim_after_stop_phrases(result["generation"], stop_phrases)

        if include_response:
            result["response"] = response

        return result

    def _parse_anthropic_response(self, response) -> dict:
        """Parse an Anthropic Messages API response into the standard result dict."""
        text_parts = []
        thinking_parts = []

        for block in response.content:
            if block.type == "thinking":
                thinking_parts.append(block.thinking)
            elif block.type == "text":
                text_parts.append(block.text)

        generation = "\n".join(text_parts) if text_parts else ""
        reasoning_content = "\n".join(thinking_parts) if thinking_parts else ""

        # Token counts
        input_tokens = response.usage.input_tokens if hasattr(response.usage, "input_tokens") else 0
        output_tokens = response.usage.output_tokens if hasattr(response.usage, "output_tokens") else 0

        result = {
            "generation": generation,
            "num_generated_tokens": output_tokens,
        }

        if reasoning_content:
            result["reasoning_content"] = reasoning_content

        # Extract thinking token count from usage if available
        cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
        if cache_creation or cache_read:
            result["cache_creation_input_tokens"] = cache_creation
            result["cache_read_input_tokens"] = cache_read

        # Estimate reasoning vs answer tokens from content blocks
        if thinking_parts:
            # Anthropic doesn't give a direct breakdown in usage, but we can
            # check for the extended thinking usage fields if available
            thinking_tokens = getattr(response.usage, "thinking_tokens", None)
            if thinking_tokens is not None:
                result["num_reasoning_tokens"] = thinking_tokens
                result["num_answer_tokens"] = output_tokens - thinking_tokens
            else:
                # Rough estimate: proportion of text length
                thinking_len = sum(len(t) for t in thinking_parts)
                total_len = thinking_len + sum(len(t) for t in text_parts)
                if total_len > 0:
                    estimated_thinking_tokens = int(output_tokens * thinking_len / total_len)
                    result["num_reasoning_tokens"] = estimated_thinking_tokens
                    result["num_answer_tokens"] = output_tokens - estimated_thinking_tokens

        # Map stop reason to OpenAI-style finish_reason
        stop_reason = response.stop_reason
        if stop_reason == "end_turn":
            result["finish_reason"] = "stop"
        elif stop_reason == "max_tokens":
            result["finish_reason"] = "length"
        elif stop_reason == "stop_sequence":
            result["finish_reason"] = "stop"
        else:
            result["finish_reason"] = stop_reason or "stop"

        # Serialize output for conversation history
        serialized_output = []
        for block in response.content:
            if hasattr(block, "model_dump"):
                serialized_output.append(block.model_dump())
            else:
                serialized_output.append({"type": block.type, "text": getattr(block, "text", "")})
        result["serialized_output"] = [{"role": "assistant", "content": serialized_output}]

        return result
