# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import copy
import logging
import re
import sys
import time
import uuid
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import tensorrt_llm
import tensorrt_llm.bindings.executor as trtllm
import torch
from fastapi import FastAPI, HTTPException
from mpi4py import MPI
from pydantic import BaseModel
from tensorrt_llm.runtime.model_runner_cpp import ExternalDraftTokensConfig, ModelRunnerCpp
from transformers import AutoTokenizer

app = FastAPI(title="TensorRT-LLM Server")


# keeping it here to make this file self-contained. This is duplicated from model.py
def trim_after_stop_phrases(text: str, stop_phrases: List[str]) -> str:
    """Removes everything after the last stop token."""
    if not stop_phrases:
        return text
    # Escape all special characters in stop phrases
    escaped_stop_phrases = [re.escape(sp) for sp in stop_phrases]
    return re.split("|".join(escaped_stop_phrases), text, maxsplit=1)[0]


def parse_input(input_texts: str, tokenizer):
    batch_input_ids = [
        tokenizer.encode(
            input_text,
            add_special_tokens=False,
        )
        for input_text in input_texts
    ]
    batch_input_ids = [torch.tensor(x, dtype=torch.int32).unsqueeze(0) for x in batch_input_ids]
    input_lengths = [x.size(1) for x in batch_input_ids]

    return batch_input_ids, input_lengths


def get_output(output_ids, input_length, max_output_len, tokenizer, eos_token) -> tuple[str, list[str], int]:
    """Returns detokenized text and the number of tokens."""
    output_begin = input_length
    output_end = input_length + max_output_len
    outputs = output_ids[output_begin:output_end]
    eos_ids = (outputs == eos_token).nonzero(as_tuple=True)[-1]
    if len(eos_ids) > 0:
        outputs = outputs[: eos_ids[0]]
    outputs = outputs.tolist()
    return tokenizer.decode(outputs), tokenizer.convert_ids_to_tokens(outputs), len(outputs)


def load_tokenizer(tokenizer_dir: str):

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        legacy=False,
        padding_side='left',
        truncation_side='left',
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    end_id = tokenizer.eos_token_id

    return tokenizer, pad_id, end_id


def generate(
    runner,
    batch_input_ids: List[torch.Tensor],
    *,
    position_ids: List[torch.Tensor] = None,
    encoder_input_ids: List[torch.Tensor] = None,
    encoder_input_features: List[torch.Tensor] = None,  # TODO: add to doc string
    encoder_output_lengths: List[int] = None,
    cross_attention_masks: List[torch.Tensor] = None,  # TODO: add to doc string
    mrope_params=None,
    sampling_config=None,
    lora_uids: Optional[list] = None,
    lookahead_config: list[int] | None = None,
    streaming: bool = False,
    stopping_criteria=None,
    logits_processor_names: list[str] | None = None,
    max_new_tokens: int = 1,
    end_id: int | None = None,
    pad_id: int | None = None,
    bad_words_list: list[list[int]] | None = None,
    stop_words_list: list[list[int]] | None = None,
    return_dict: bool = False,
    output_sequence_lengths: bool = False,
    output_log_probs: bool = False,
    output_cum_log_probs: bool = False,
    prompt_table=None,
    prompt_tasks: Optional[str] = None,
    input_token_extra_ids: List[List[int]] = None,
    return_all_generated_tokens: bool = False,
    tokenizer=None,
    timeout=None,
    input_lengths=None,
    **kwargs,
):
    """
    Generates sequences of token ids.
    The generation-controlling parameters are set in the sampling_config; it will be set to a default one if not passed.
    You can override any sampling_config's attributes by passing corresponding parameters.

    Args:
        batch_input_ids (List[torch.Tensor]):
            A list of input id tensors. Each tensor is of shape (sequence_length, ).
        position_ids (List[torch.Tensor]):
            A list of position id tensors. Each tensor is of shape (sequence_length, ).
        encoder_input_ids (List[torch.Tensor]):
            A list of encoder input id tensors for encoder-decoder models (optional). Each tensor is of shape (sequence_length, ).
        encoder_input_features: (List[torch.Tensor]):
            A list of encoder input feature tensors for multimodal encoder-decoder models (optional). Each tensor is of shape (sequence_length, feature_dim).
        encoder_output_lengths: (List[int]):
            A list of encoder output lengths (optional) if encoder output has different length from encoder input (due to convolution down-sampling, etc.)
        sampling_config (SamplingConfig):
            The sampling configuration to be used as base parametrization for the generation call.
            The passed **kwargs matching the sampling_config's attributes will override them.
            If the sampling_config is not provided, a default will be used.
        prompt_table (str or torch.Tensor):
            The file path of prompt table (.npy format, exported by nemo_prompt_convert.py) or the prompt table itself.
        prompt_tasks (str):
            The prompt tuning task ids for the input batch, in format of comma-separated list (e.g., 0,3,1,0).
        input_token_extra_ids (List[List[int]]):
            Input token extra ids for using p-tuning and KV Cache reuse together
        lora_uids (list):
            The uids of LoRA weights for the input batch. Use -1 to disable the LoRA module.
        streaming (bool):
            Whether or not to use streaming mode for generation.
        stopping_criteria (StoppingCriteria):
            Custom stopping criteria.
        logits_processor_names (List[str]):
            Custom logits processor names.
        return_all_generated_tokens (bool):
            Whether the full output is returned at each streaming step
        kwargs (Dict[str, Any]:
            Ad hoc parametrization of sampling_config.
            The passed **kwargs matching the sampling_config's attributes will override them.
    Returns:
        torch.Tensor or dict:
            If return_dict=False, the method returns generated output_ids.
            If return_dict=True, the method returns a dict of output_ids,
            sequence_lengths (if sampling_config.output_sequence_lengths=True),
            context_logits and generation_logits (if self.gather_context_logits=True and
            self.gather_generation_logits=True, respectively).
    """
    assert streaming
    # TODO: Check if these can be supported now and support them
    if stopping_criteria is not None:
        raise RuntimeError("Stopping criteria is not supported in C++ session.")

    if not runner.use_kv_cache and max_new_tokens > 1:
        raise RuntimeError('Disabled KV cache is intended for context phase only now.')

    # If we are in a multi-gpu scenario, only rank 0 continues
    if not runner.session.can_enqueue_requests():
        return []

    # Convert tensor input to plain lists
    batch_input_ids_list = [a.tolist() for a in batch_input_ids]
    encoder_input_ids_list = [a.tolist() for a in encoder_input_ids] if encoder_input_ids else None

    if sampling_config is None:
        # Convert from old API of SamplingConfig
        # Note: Due to a Python3.10 bug one cannot use inspect on it currently
        accepted_parameters = [
            "num_beams",
            "top_k",
            "top_p",
            "top_p_min",
            "top_p_reset_ids",
            "top_p_decay",
            "temperature",
            "min_tokens",
            "beam_search_diversity_rate",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "length_penalty",
            "early_stopping",
            "no_repeat_ngram_size",
            "random_seed",
            "num_return_sequences",
            "min_p",
            "beam_width_array",
        ]
        rename_params = {"num_beams": "beam_width", "random_seed": "seed"}
        sampling_params = {k: v for k, v in kwargs.items() if k in accepted_parameters}
        for k, v in rename_params.items():
            if k in sampling_params:
                sampling_params[v] = sampling_params.pop(k)
        if "top_p" in sampling_params and sampling_params["top_p"] == 0.0:
            sampling_params["top_p"] = None
        if sampling_params.get("top_p_min") == 0.0:
            sampling_params["top_p_min"] = None

        # TODO: improve usage of SamplingConfig. For example,
        # construct SamplingConfig for each request, rather than one for the whole batch.
        # Here we use beam width array for each request for Variable-Beam-Width-Search.
        batch_size = len(batch_input_ids)
        use_sampling_config_for_each_request = False
        # Just placeholder for non-Variable-Beam-Width-Search
        sampling_config_list = [None] * batch_size
        if (
            "beam_width_array" in sampling_params
            and sampling_params["beam_width_array"] is not None
            and len(sampling_params["beam_width_array"]) == batch_size
        ):
            use_sampling_config_for_each_request = True
            sp_copy = copy.deepcopy(sampling_params)
            for i in range(batch_size):
                bwa = sampling_params["beam_width_array"][i]
                sp_copy["beam_width_array"] = bwa
                sp_copy["beam_width"] = max(bwa)
                sampling_config_list[i] = trtllm.SamplingConfig(**sp_copy)
            # Just placeholder for Variable-Beam-Width-Search and for `runner._check_inputs`
            max_beam_width = max(sc.beam_width for sc in sampling_config_list)
            sampling_params["beam_width"] = max_beam_width
            sampling_params["beam_width_array"] = [max_beam_width] * 8
        sampling_config = trtllm.SamplingConfig(**sampling_params)
    else:
        sampling_config = copy.deepcopy(sampling_config)

    runner._check_inputs(batch_input_ids_list, encoder_input_ids_list, sampling_config, max_new_tokens)

    output_config = trtllm.OutputConfig(
        return_context_logits=runner.gather_context_logits,
        return_generation_logits=runner.gather_generation_logits,
        return_log_probs=output_log_probs,
    )

    prompt_tuning_configs = runner._prepare_ptuning_executor(
        batch_input_ids_list, prompt_table, prompt_tasks, input_token_extra_ids
    )
    mrope_configs = runner._prepare_mrope_executor(batch_input_ids_list, mrope_params)

    stop_words_list_none = runner._prepare_words_list(None, len(batch_input_ids_list))
    bad_words_list = runner._prepare_words_list(bad_words_list, len(batch_input_ids_list))
    logits_processor_names = runner._prepare_names_list(logits_processor_names, len(batch_input_ids_list))

    lora_configs = runner._prepare_lora_configs(lora_uids, len(batch_input_ids_list))
    request_lookahead_config = None
    if lookahead_config is not None:
        [w, n, g] = lookahead_config
        request_lookahead_config = trtllm.LookaheadDecodingConfig(w, n, g)
    skip_cross_attn_blocks = kwargs.get('skip_cross_attn_blocks', None)

    # Draft-Target-Model speculative decoding
    if (
        "draft_tokens_list" in kwargs.keys()
        and kwargs["draft_tokens_list"] is not None
        and "draft_logits_list" in kwargs.keys()
        and kwargs["draft_logits_list"] is not None
    ):
        # Use logits to accept
        external_draft_tokens_configs = [
            ExternalDraftTokensConfig(draft_tokens, draft_logits)
            for draft_tokens, draft_logits in zip(kwargs["draft_tokens_list"], kwargs["draft_logits_list"])
        ]
        is_draft_target_model = True
    elif "draft_tokens_list" in kwargs.keys() and kwargs["draft_tokens_list"] is not None:
        # Use tokens to accept
        external_draft_tokens_configs = [
            ExternalDraftTokensConfig(draft_tokens) for draft_tokens in kwargs["draft_tokens_list"]
        ]
        is_draft_target_model = True
    else:
        external_draft_tokens_configs = [None] * len(batch_input_ids_list)
        is_draft_target_model = False

    language_adapter_uids = [None] * len(batch_input_ids_list)

    requests = [
        trtllm.Request(
            input_token_ids=input_ids,
            encoder_input_token_ids=encoder_input_ids_list[i] if encoder_input_ids is not None else None,
            encoder_output_length=encoder_output_lengths[i] if encoder_output_lengths is not None else None,
            encoder_input_features=(
                encoder_input_features[i].contiguous() if encoder_input_features is not None else None
            ),
            position_ids=position_ids[i].tolist() if position_ids is not None else None,
            cross_attention_mask=(
                cross_attention_masks[i].contiguous()
                if (cross_attention_masks is not None and cross_attention_masks[i] is not None)
                else None
            ),
            max_tokens=max_new_tokens,
            pad_id=pad_id,
            end_id=end_id,
            stop_words=stop_words,
            bad_words=bad_words,
            sampling_config=(
                sampling_config_each_request if use_sampling_config_for_each_request else sampling_config
            ),
            lookahead_config=request_lookahead_config,
            streaming=streaming,
            output_config=output_config,
            prompt_tuning_config=prompt_tuning_config,
            mrope_config=mrope_config,
            lora_config=lora_config,
            return_all_generated_tokens=return_all_generated_tokens,
            logits_post_processor_name=logits_post_processor_name,
            external_draft_tokens_config=external_draft_tokens_config,
            skip_cross_attn_blocks=skip_cross_attn_blocks,
            language_adapter_uid=language_adapter_uid,
        )
        for i, (
            input_ids,
            stop_words,
            bad_words,
            prompt_tuning_config,
            mrope_config,
            lora_config,
            logits_post_processor_name,
            external_draft_tokens_config,
            language_adapter_uid,
            sampling_config_each_request,
        ) in enumerate(
            zip(
                batch_input_ids_list,
                stop_words_list_none,
                bad_words_list,
                prompt_tuning_configs,
                mrope_configs,
                lora_configs,
                logits_processor_names,
                external_draft_tokens_configs,
                language_adapter_uids,
                sampling_config_list,
            )
        )
    ]

    request_ids = runner.session.enqueue_requests(requests)
    assert len(request_ids) == 1

    stream_kwargs = dict(
        runner=runner,
        request_id=request_ids[0],
        end_id=end_id,
        return_dict=return_dict,
        output_sequence_lengths=output_sequence_lengths,
        output_log_probs=output_log_probs,
        output_cum_log_probs=output_cum_log_probs,
        batch_input_ids=batch_input_ids,
        batch_input_ids_list=batch_input_ids_list,
        streaming=streaming,
        return_all_generated_tokens=return_all_generated_tokens,
        sampling_config=sampling_config,
        is_draft_target_model=is_draft_target_model,
        stop_words_list=stop_words_list,
        tokenizer=tokenizer,
        timeout=timeout,
        input_lengths=input_lengths,
    )

    return request_ids[0], stream_kwargs


def _stream(
    runner,
    request_id,
    end_id,
    return_dict,
    output_sequence_lengths,
    output_log_probs,
    output_cum_log_probs,
    batch_input_ids,
    streaming,
    batch_input_ids_list,
    return_all_generated_tokens,
    stop_words_list,
    tokenizer,
    input_lengths,
    sampling_config=None,
    timeout=None,
    is_draft_target_model: bool = False,
    repetition_check_chars: Optional[int] = None,
    repetition_limit: Optional[int] = None,
):
    if stop_words_list is None:
        stop_words_list = []
    request_ids = [request_id]
    num_sequences = runner._get_num_sequences(sampling_config)
    output_ids = [
        [copy.deepcopy(batch_input_ids_list[batch_idx]) for _ in range(num_sequences)]
        for batch_idx in range(len(request_ids))
    ]
    # checking this often for stop phrases / repetitions
    num_tokens_to_check = 20
    repetition_check_tokens = 200

    stop_on_repetitions = repetition_check_chars is not None and repetition_limit is not None

    start_time = time.time()

    finished_reqs = 0
    last_seq_length_stop = 0
    last_seq_length_rep = 0
    stopped_on_repetition = False

    while finished_reqs < len(request_ids):
        multi_responses = runner.session.await_responses(request_ids)
        responses = [response for responses in multi_responses for response in responses]
        for response in responses:
            if response.result.is_final:
                finished_reqs += 1

        output = runner._fill_output(
            responses=responses,
            output_ids=output_ids,
            end_id=end_id,
            return_dict=return_dict,
            output_sequence_lengths=output_sequence_lengths,
            output_log_probs=output_log_probs,
            output_cum_log_probs=output_cum_log_probs,
            batch_input_ids=batch_input_ids,
            batch_input_ids_list=batch_input_ids_list,
            streaming=streaming,
            request_ids=request_ids,
            return_all_generated_tokens=return_all_generated_tokens,
            sampling_config=sampling_config,
            is_draft_target_model=is_draft_target_model,
            output_generation_logits=False,
        )
        seq_length = output['sequence_lengths'][0].item()

        if timeout:
            current_time = time.time() - start_time
            if current_time >= timeout:
                runner.session.cancel_request(request_ids[0])
                break

        if seq_length - last_seq_length_stop >= num_tokens_to_check:
            # we are checking in overlapping pieces to ensure nothing is missed
            look_back = max(last_seq_length_stop - num_tokens_to_check // 2, input_lengths[0])
            generation_suffix = output['output_ids'][0, 0, look_back:seq_length]
            suffix_length = seq_length - look_back
            out_string = get_output(generation_suffix, 0, suffix_length, tokenizer, end_id)[0]
            last_seq_length_stop = seq_length

            matching_stop_word = None
            for stop_word in stop_words_list:
                if stop_word in out_string:
                    matching_stop_word = stop_word
                    break

            if matching_stop_word is not None:
                runner.session.cancel_request(request_ids[0])
                break

        if stop_on_repetitions and seq_length - last_seq_length_rep >= repetition_check_tokens:
            last_seq_length_rep = seq_length
            # detokenizing everything so far to make sure generation
            # is not corrupted by stitching partial detokenizations
            full_gen_so_far = get_output(
                output['output_ids'][0, 0], input_lengths[0], output['sequence_lengths'][0], tokenizer, end_id
            )[0]
            # checking for repetition
            if len(full_gen_so_far) < repetition_check_chars:
                continue

            suffix = full_gen_so_far[-repetition_check_chars:]
            cnt = full_gen_so_far[:-repetition_check_chars].count(suffix)
            if cnt >= repetition_limit:
                stopped_on_repetition = True
                runner.session.cancel_request(request_ids[0])
                break

    out_string, out_tokens, num_generated_tokens = get_output(
        output['output_ids'][0, 0], input_lengths[0], output['sequence_lengths'][0], tokenizer, end_id
    )

    # TODO: the number of tokens is not exact, because we might trim the output a bit,
    #       but close enough for practical purposes
    matching_stop_word = None
    for stop_word in stop_words_list:
        if stop_word in out_string:
            matching_stop_word = stop_word
            break
    if matching_stop_word is not None:
        out_string = trim_after_stop_phrases(out_string, stop_words_list)
        # adding it back, since we only need to remove what's *after* the stop phrase
        out_string += matching_stop_word
    else:
        # trtllm removes end id if it was the stop reason
        # this is a hack to add it back, but we are going to include it even when
        # it was not generated by the model e.g. if we stopped due to max tokens
        out_string += tokenizer.decode(end_id)
    generation_time = int(round(time.time() - start_time))

    result = {
        'generation': out_string,
        'num_generated_tokens': num_generated_tokens,
        'generation_time': generation_time,
        'stopped_on_repetition': stopped_on_repetition,
    }

    if output_log_probs:
        result['tokens'] = out_tokens
        result['logprobs'] = output['log_probs'][0][0].tolist()

    return result


class TensorRTLLM:
    def __init__(
        self,
        model_path: str,
        max_batch_size: Optional[int] = None,
        max_input_len: Optional[int] = None,
        max_output_len: Optional[int] = None,
        max_beam_width: Optional[int] = None,
        kv_cache_free_gpu_memory_fraction: Optional[float] = None,
        disable_chunked_context: bool = False,
        repetition_check_chars: Optional[int] = None,
        repetition_limit: Optional[int] = None,
    ):
        self.tokenizer, self.pad_id, self.end_id = load_tokenizer(tokenizer_dir=model_path)

        runner_kwargs = dict(
            engine_dir=model_path,
            rank=tensorrt_llm.mpi_rank(),
            max_batch_size=max_batch_size,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            max_beam_width=max_beam_width,
            kv_cache_free_gpu_memory_fraction=kv_cache_free_gpu_memory_fraction,
            enable_chunked_context=not disable_chunked_context,
            kv_cache_enable_block_reuse=True,
        )

        self.runner = ModelRunnerCpp.from_dir(**runner_kwargs)
        self.repetition_check_chars = repetition_check_chars
        self.repetition_limit = repetition_limit

        self.active_generations = {}
        self.active_requests = {}
        self.executor = ThreadPoolExecutor(max_workers=1024)

    def get_output(
        self,
        generation_id,
        batch_input_ids,
        input_lengths,
        max_output_token,
        top_k,
        top_p,
        top_p_min,
        temperature,
        repetition_penalty,
        random_seed,
        stop_words_list,
        top_logprobs,
        timeout,
    ):
        try:
            request_id, stream_kwargs = generate(
                self.runner,
                batch_input_ids[0],
                max_new_tokens=max_output_token,
                end_id=self.end_id,
                pad_id=self.pad_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                top_p_min=top_p_min,
                repetition_penalty=repetition_penalty,
                random_seed=random_seed,
                output_log_probs=bool(top_logprobs is not None),
                # stop words in trtllm are supported on the token-level only and this representation is not unique
                # so instead of passing in all tokenizations (is that even possible?) of each phrase, we will
                # instead stream outputs and detokenize them to check for stop words - this is done inside
                # overriden generate/stream functions above
                tokenizer=self.tokenizer,
                stop_words_list=stop_words_list,
                input_lengths=input_lengths,
                return_dict=True,
                output_sequence_lengths=True,
                streaming=True,
                timeout=timeout,
            )
            # Forward repetition parameters:
            stream_kwargs['repetition_check_chars'] = self.repetition_check_chars
            stream_kwargs['repetition_limit'] = self.repetition_limit

            self.active_requests[generation_id] = request_id
            output = _stream(**stream_kwargs)

        except RuntimeError as e:
            logging.error("RuntimeError: %s", e)
            # TODO: return dictionary with a proper error reporting
            output = {
                "generation": f"RuntimeError: {e}",
                "num_generated_tokens": 10,
                "generation_time": 0,
                "stopped_on_repetition": False,
            }

        return output

    @torch.no_grad()
    def start_generation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        generation_id = str(uuid.uuid4())
        batch_input_ids, input_lengths = parse_input([data["prompt"]], self.tokenizer)

        future = self.executor.submit(
            self.get_output,
            generation_id,
            batch_input_ids,
            input_lengths,
            data["max_new_tokens"],
            data["top_k"],
            data["top_p"],
            data["top_p_min"],
            data["temperature"],
            data["repetition_penalty"],
            data["random_seed"],
            data["stop_words_list"],
            data["top_logprobs"],
            data["timeout"],
        )

        self.active_generations[generation_id] = future

        return generation_id

    def get_generation(self, generation_id: str) -> Dict[str, Any]:
        if generation_id not in self.active_generations:
            raise HTTPException(status_code=404, detail="Generation not found")

        future = self.active_generations[generation_id]

        if future.done():
            result = future.result()
            # Clean up completed generation
            del self.active_generations[generation_id]
            return result
        else:
            return None

    def cancel_request(self, request_id):
        self.runner.session.cancel_request(request_id)

    def cancel_generation(self, generation_id: str) -> Dict[str, Any]:
        if generation_id not in self.active_generations:
            return {"status": "not found"}

        future = self.active_generations[generation_id]
        request_id = self.active_requests[generation_id]
        self.cancel_request(request_id)
        future.cancel()
        # Clean up canceled generation
        del self.active_generations[generation_id]

        return {"status": "canceled"}


class GenerationRequest(BaseModel):
    prompt: str
    tokens_to_generate: int = 64
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: float = 1.0
    top_p_min: float = 0.0
    repetition_penalty: float = 1.2
    random_seed: int = 0
    stop_words_list: Optional[List[str]] = None
    top_logprobs: Optional[int] = None
    timeout: Optional[int] = None


class GenerationResponse(BaseModel):
    generation: Optional[str] = None
    num_generated_tokens: Optional[int] = None
    generation_time: Optional[int] = None
    tokens: Optional[list[str]] = None
    logprobs: Optional[list[float]] = None
    stopped_on_repetition: Optional[bool] = None


class GenerationResponseAsync(BaseModel):
    generation_id: str


class CancelGenerationResponse(BaseModel):
    status: str


class GetGenerationRequest(BaseModel):
    generation_id: str


class MPIWrapper:
    def __init__(
        self,
        model_path: str,
        max_batch_size: Optional[int] = None,
        max_input_len: Optional[int] = None,
        max_output_len: Optional[int] = None,
        max_beam_width: Optional[int] = None,
        kv_cache_free_gpu_memory_fraction: Optional[float] = None,
        disable_chunked_context: bool = False,
        repetition_check_chars: Optional[int] = None,
        repetition_limit: Optional[int] = None,
    ):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.model = TensorRTLLM(
            model_path=model_path,
            max_batch_size=max_batch_size,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            max_beam_width=max_beam_width,
            kv_cache_free_gpu_memory_fraction=kv_cache_free_gpu_memory_fraction,
            disable_chunked_context=disable_chunked_context,
            repetition_check_chars=repetition_check_chars,
            repetition_limit=repetition_limit,
        )
        self.app = None
        if self.rank == 0:
            self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        app = FastAPI(title="TensorRT-LLM Service")

        @app.put("/generate", response_model=GenerationResponse)
        async def generate(request: GenerationRequest):
            data = {
                "prompt": request.prompt,
                "max_new_tokens": request.tokens_to_generate,
                "temperature": request.temperature,
                "top_k": None if request.top_k == 0 else request.top_k,
                "top_p": request.top_p,
                "top_p_min": request.top_p_min,
                "repetition_penalty": request.repetition_penalty,
                "random_seed": request.random_seed,
                "stop_words_list": request.stop_words_list,
                "top_logprobs": request.top_logprobs,
                "timeout": request.timeout,
            }

            self.comm.Barrier()
            data = self.comm.bcast(data, root=0)

            generation_id = self.model.start_generation(data)

            while True:
                output = self.model.get_generation(generation_id)
                if output is not None:
                    return output
                await asyncio.sleep(0.5)

        @app.put("/generate_async", response_model=GenerationResponseAsync)
        async def generate_async(request: GenerationRequest):
            data = {
                "prompt": request.prompt,
                "max_new_tokens": request.tokens_to_generate,
                "temperature": request.temperature,
                "top_k": None if request.top_k == 0 else request.top_k,
                "top_p": request.top_p,
                "top_p_min": request.top_p_min,
                "repetition_penalty": request.repetition_penalty,
                "random_seed": request.random_seed,
                "stop_words_list": request.stop_words_list,
                "top_logprobs": request.top_logprobs,
                "timeout": request.timeout,
            }

            self.comm.Barrier()
            data = self.comm.bcast(data, root=0)

            generation_id = self.model.start_generation(data)
            return {'generation_id': generation_id}

        @app.put("/get_generation", response_model=GenerationResponse)
        async def get_generation(request: GetGenerationRequest):
            generation_id = request.generation_id

            output = self.model.get_generation(generation_id)
            if output is not None:
                return output
            return {'generation': None}

        @app.put("/cancel_generation", response_model=CancelGenerationResponse)
        async def cancel_generation(request: GetGenerationRequest):
            generation_id = request.generation_id
            return self.model.cancel_generation(generation_id)

        return app

    def worker_loop(self):
        """Worker loop for non-rank-0 processes"""
        while True:
            self.comm.Barrier()
            data = None
            data = self.comm.bcast(data, root=0)
            if data is None:
                continue
            self.model.start_generation(data)

    def run(self, host: str = "0.0.0.0", port: int = 5000):
        if self.rank == 0:
            import uvicorn

            uvicorn.run(self.app, host=host, port=port, ws_max_queue=1500)
        else:
            self.worker_loop()


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--max_batch_size", type=int, default=None, help="Maximum batch size")
    parser.add_argument("--max_input_len", type=int, default=None, help="Maximum input length")
    parser.add_argument("--max_output_len", type=int, default=None, help="Maximum output length")
    parser.add_argument("--max_beam_width", type=int, default=None, help="Maximum beam width")
    parser.add_argument(
        "--kv_cache_free_gpu_memory_fraction", type=float, default=None, help="Free GPU memory fraction for cache"
    )
    parser.add_argument("--disable_chunked_context", action="store_true", help="Disable chunked context")
    parser.add_argument(
        "--repetition_check_chars", type=int, default=None, help="Number of characters to check for repetition"
    )
    parser.add_argument(
        "--repetition_limit", type=int, default=None, help="Repetition limit: stops generation if exceeded"
    )
    args = parser.parse_args()

    wrapper = MPIWrapper(
        model_path=args.model_path,
        max_batch_size=args.max_batch_size,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        max_beam_width=args.max_beam_width,
        kv_cache_free_gpu_memory_fraction=args.kv_cache_free_gpu_memory_fraction,
        disable_chunked_context=args.disable_chunked_context,
        repetition_check_chars=args.repetition_check_chars,
        repetition_limit=args.repetition_limit,
    )
    wrapper.run(host=args.host, port=args.port)


if __name__ == "__main__":

    class LogFilter(logging.Filter):
        def filter(self, record):
            filter_strings = (
                "PUT /generate HTTP/1.1",
                "PUT /get_generation HTTP/1.1",
                "PUT /generate_async HTTP/1.1",
                "PUT /cancel_generation HTTP/1.1",
            )
            return all(filter_string not in record.getMessage() for filter_string in filter_strings)

    logging.getLogger('uvicorn.access').addFilter(LogFilter())
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    main()
