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

from openai import AsyncOpenAI

from nemo_skills.utils import get_logger_name

from .base import EndpointType
from .openai import OpenAIModel
from .utils import trim_after_stop_phrases

LOG = logging.getLogger(get_logger_name(__file__))

# Transient errors that warrant a retry
_RETRYABLE_EXCEPTIONS = (
    Exception,  # Broad catch for connection/timeout errors during streaming
)


class OpenAIStreamingModel(OpenAIModel):
    """OpenAI model that uses native SDK streaming for the responses API.

    Bypasses litellm for the responses endpoint to get proper streaming
    with keepalive events, preventing 504 gateway timeouts on long-running
    reasoning requests (e.g., codex models via NVIDIA inference API).

    For chat and text endpoints, delegates to the parent OpenAIModel (litellm).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Native OpenAI async client for responses API streaming.
        self._openai_client = AsyncOpenAI(
            api_key=self.litellm_kwargs["api_key"],
            base_url=self.base_url,
        )

    async def generate_async(
        self,
        prompt,
        endpoint_type=None,
        tokens_to_generate=None,
        temperature=0.0,
        top_p=0.95,
        top_k=-1,
        min_p=0.0,
        repetition_penalty=1.0,
        random_seed=None,
        stop_phrases=None,
        top_logprobs=None,
        timeout=14400,
        remove_stop_phrases=True,
        stream=False,
        reasoning_effort=None,
        tools=None,
        include_response=False,
        extra_body=None,
        response_format=None,
    ):
        # Only intercept responses endpoint; delegate everything else to parent
        if endpoint_type != EndpointType.responses:
            return await super().generate_async(
                prompt=prompt,
                endpoint_type=endpoint_type,
                tokens_to_generate=tokens_to_generate,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                random_seed=random_seed,
                stop_phrases=stop_phrases,
                top_logprobs=top_logprobs,
                timeout=timeout,
                remove_stop_phrases=remove_stop_phrases,
                stream=stream,
                reasoning_effort=reasoning_effort,
                tools=tools,
                include_response=include_response,
                extra_body=extra_body,
                response_format=response_format,
            )

        assert isinstance(prompt, list), "Responses completion requests must be a list."

        # Build native responses API params
        native_params = self._build_native_responses_params(
            input_messages=prompt,
            tokens_to_generate=tokens_to_generate,
            reasoning_effort=reasoning_effort,
            tools=tools,
        )

        # Always stream to receive keepalive events and prevent gateway timeouts.
        # Retry on transient connection errors (proxy drops, incomplete reads).
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                async with self.concurrent_semaphore:
                    response_stream = await self._openai_client.responses.create(
                        **native_params, stream=True
                    )
                    result = await self._collect_responses_stream_async(
                        response_stream, include_response=include_response
                    )
                break
            except _RETRYABLE_EXCEPTIONS as exc:
                if attempt == max_retries:
                    raise
                wait = 2**attempt
                LOG.warning(
                    "OpenAI responses streaming failed (attempt %d/%d): %s. Retrying in %ds...",
                    attempt + 1,
                    max_retries + 1,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)

        if remove_stop_phrases and stop_phrases:
            result["generation"] = trim_after_stop_phrases(result["generation"], stop_phrases)

        return result

    def _build_native_responses_params(
        self,
        input_messages,
        tokens_to_generate=None,
        reasoning_effort=None,
        tools=None,
    ):
        """Build params for the native OpenAI responses.create() call."""
        # Convert system → developer role (same as _build_chat_request_params)
        messages = [
            {**msg, "role": "developer"} if msg.get("role") == "system" else msg for msg in input_messages
        ]

        params = {
            "model": self.model,
            "input": messages,
        }

        if tokens_to_generate is not None:
            params["max_output_tokens"] = tokens_to_generate

        if reasoning_effort:
            params["reasoning"] = {"effort": reasoning_effort}

        if tools:
            params["tools"] = tools

        return params
