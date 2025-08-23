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

import json
import logging
import os
import random
import re
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
from typing import Dict, Optional, List, Any, Tuple
from statistics import mean, median, stdev
from multiprocessing import Pool
from functools import partial

# Suppress pydub warnings about missing ffmpeg
warnings.filterwarnings("ignore", "Couldn't find ffmpeg or avconv", RuntimeWarning, "pydub.utils")

from sdp.processors.base_processor import BaseProcessor
from tqdm.contrib.concurrent import process_map

from nemo_skills.evaluation.metrics.utils import is_correct_judgement
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import get_logger_name, unroll_files

LOG = logging.getLogger(get_logger_name(__file__))


class ReadData(BaseProcessor):
    def __init__(
        self,
        input_files: Optional[str] = None,
        preprocessed_dataset_files: Optional[str] = None,
        input_key="question",
        output_key=None,  # if None, we are not doing correctness checks
        skip_first: int = 0,
        add_correct: bool = True,
        add_incorrect: bool = False,
        add_unlabeled: bool = False,
        use_judgement: bool = False,
        keys_to_keep: list[str] | None = None,
        deduplicate: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_files = input_files
        self.preprocessed_dataset_files = preprocessed_dataset_files
        self.input_key = input_key
        self.output_key = output_key
        self.skip_first = skip_first
        self.add_correct = add_correct
        self.add_incorrect = add_incorrect
        self.add_unlabeled = add_unlabeled
        self.use_judgement = use_judgement
        self.keys_to_keep = keys_to_keep
        self.deduplicate = deduplicate

        if self.keys_to_keep is not None:
            self.keys_to_keep = set(self.keys_to_keep)
            self.keys_to_keep.add(self.input_key)
            if self.output_key is not None:
                self.keys_to_keep.add(self.output_key)
                self.keys_to_keep.add("symbolic_correct")
                self.keys_to_keep.add("judgement")

        if isinstance(self.input_files, str):
            if ',' in self.input_files:
                self.input_files = self.input_files.split(",")
            else:
                self.input_files = self.input_files.split(" ")

        if isinstance(self.preprocessed_dataset_files, str):
            if ',' in self.preprocessed_dataset_files:
                self.preprocessed_dataset_files = self.preprocessed_dataset_files.split(",")
            else:
                self.preprocessed_dataset_files = self.preprocessed_dataset_files.split(" ")

        if self.input_files is None and self.preprocessed_dataset_files is None:
            raise ValueError("Either `input_files` or `preprocessed_dataset_files` should be provided")

        if self.output_key is not None and not self.add_correct and not self.add_incorrect:
            raise ValueError("At least one of `add_correct` and `add_incorrect` should be True")

    def _read_preprocessed_data(self, file_handle) -> int:
        samples = []
        questions = set()
        for idx, line in enumerate(file_handle):
            if idx < self.skip_first:
                continue
            sample = json.loads(line)
            if self.keys_to_keep:
                sample = {k: v for k, v in sample.items() if k in self.keys_to_keep}
            questions.add(sample[self.input_key])
            samples.append(sample)

        return samples

    def _parallel_read_file(self, args):
        file_path, read_fn = args
        with open(file_path, "rt", encoding="utf-8") as file_handle:
            samples = read_fn(file_handle)
        return samples

    def _read_raw_data(self, file_handle) -> int:
        samples = []

        for idx, file_line in enumerate(file_handle):
            if idx < self.skip_first:
                continue
            # if different files have different number of lines
            if file_line is None:
                continue
            line_dict = json.loads(file_line)
            # can be empty for incomplete generations
            if not line_dict:
                continue

            if self.output_key is not None and not self.add_unlabeled:
                if not self.use_judgement:
                    if "symbolic_correct" not in line_dict:
                        LOG.warning("Found incomplete generations (symbolic_correct field is missing) - skipping")
                        continue

                    if not self.add_correct and line_dict["symbolic_correct"]:
                        continue

                    if not self.add_incorrect and not line_dict["symbolic_correct"]:
                        continue
                else:
                    if "judgement" not in line_dict:
                        LOG.warning("Found incomplete generations (judgement field is missing) - skipping")
                        continue

                    if not self.add_correct and is_correct_judgement(line_dict["judgement"]):
                        continue

                    if not self.add_incorrect and not is_correct_judgement(line_dict["judgement"]):
                        continue

            line_dict['filename'] = file_handle.name

            if self.keys_to_keep:
                line_dict = {k: v for k, v in line_dict.items() if k in self.keys_to_keep}

            samples.append(line_dict)

        return samples

    def _get_sample_hash(self, sample):
        if self.output_key is None:
            return sample[self.input_key]
        return (sample[self.input_key], sample[self.output_key])

    def _batch_deduplicate(self, batch):
        seen_predictions = set()
        unique_samples = []

        for sample in batch:
            sample_hash = self._get_sample_hash(sample)
            if sample_hash not in seen_predictions:
                seen_predictions.add(sample_hash)
                unique_samples.append(sample)

        return unique_samples

    def _chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def process(self):
        samples = []
        if self.input_files:
            args = [(file, self._read_raw_data) for file in unroll_files(self.input_files)]
            results = process_map(self._parallel_read_file, args, max_workers=4, chunksize=1)
            samples.extend(list(chain(*results)))
        if self.preprocessed_dataset_files:
            args = [(file, self._read_preprocessed_data) for file in unroll_files(self.preprocessed_dataset_files)]
            results = process_map(self._parallel_read_file, args, max_workers=None, chunksize=1)
            samples.extend(list(chain(*results)))

        if self.deduplicate:
            LOG.info("Total samples before deduplication: %d", len(samples))

            # Parallel deduplication
            chunk_size = 100000
            num_cores = max(100, len(samples) // chunk_size)

            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                # Process chunks in parallel
                futures = [
                    executor.submit(self._batch_deduplicate, chunk) for chunk in self._chunks(samples, chunk_size)
                ]

                # Final deduplication of results from all chunks
                seen_predictions = set()
                samples_count = 0

                with open(self.output_manifest_file, "wt", encoding="utf-8") as fout:
                    for future in futures:
                        chunk_results = future.result()
                        for sample in chunk_results:
                            sample_hash = self._get_sample_hash(sample)
                            if sample_hash not in seen_predictions:
                                seen_predictions.add(sample_hash)
                                fout.write(json.dumps(sample) + "\n")
                                samples_count += 1

            LOG.info("Total samples after deduplication: %d", samples_count)
        else:
            LOG.info("Total samples: %d", len(samples))
            with open(self.output_manifest_file, "wt", encoding="utf-8") as fout:
                for sample in samples:
                    fout.write(json.dumps(sample) + "\n")


class GroupSamples(BaseProcessor):
    def __init__(self, group_key='input', **kwargs):
        super().__init__(**kwargs)
        self.group_key = group_key

    def process(self):
        samples = defaultdict(list)
        with open(self.input_manifest_file, "rt", encoding="utf-8") as fin:
            for line in fin:
                sample = json.loads(line)
                samples[sample[self.group_key]].append(sample)

        with open(self.output_manifest_file, "wt", encoding="utf-8") as fout:
            for groupped_samples in samples.values():
                fout.write(json.dumps(groupped_samples) + "\n")


class ShuffleAndDownsampleData(BaseProcessor):
    def __init__(
        self,
        random_seed: int,
        do_shuffle: bool,
        num_samples: Optional[int] = None,
        sampling_method: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sampling_method = sampling_method
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.do_shuffle = do_shuffle

        if self.sampling_method not in [None, "random", "fair"]:
            raise ValueError(
                f"Sampling method {self.sampling_method} is not supported, use `None`, `random` or `fair`"
            )

        if self.sampling_method is None and self.num_samples is not None:
            raise ValueError("Number of samples can be specified only when sampling method is `random` or `fair`")

        if self.sampling_method is not None and self.num_samples is None:
            raise ValueError("Number of samples should be specified when sampling method is `random` or `fair`")

    def process(self):
        groupped_samples = []
        with open(self.input_manifest_file, "rt", encoding="utf-8") as fin:
            for line in fin:
                samples = json.loads(line)
                groupped_samples.append(samples)

        random.seed(self.random_seed)
        if self.sampling_method is None:
            output_instances = list(chain(*groupped_samples))
            if self.do_shuffle:
                random.shuffle(output_instances)
        if self.sampling_method == "random":
            output_instances = list(chain(*groupped_samples))
            if self.do_shuffle:
                random.shuffle(output_instances)
            output_instances = output_instances[: self.num_samples]
        elif self.sampling_method == "fair":
            soln_counter = 0
            output_instances = []
            num_input_samples = sum(len(samples) for samples in groupped_samples)
            if num_input_samples < self.num_samples:
                LOG.warning(
                    "Total SFT entries %d is not less than `num_output_samples` %d, skipping downsampling.",
                    num_input_samples,
                    self.num_samples,
                )
                output_instances = list(chain(*groupped_samples))
            # downsample only if num_input_samples > self.num_samples
            while len(output_instances) < self.num_samples and num_input_samples > self.num_samples:
                for quesn_idx in range(len(groupped_samples)):
                    if len(output_instances) == self.num_samples:
                        break
                    if len(groupped_samples[quesn_idx]) > soln_counter:
                        output_instances.append(groupped_samples[quesn_idx][soln_counter])
                soln_counter += 1
            if self.do_shuffle:
                random.shuffle(output_instances)

        with open(self.output_manifest_file, "wt", encoding="utf-8") as fout:
            for instance in output_instances:
                fout.write(json.dumps(instance) + "\n")


class WriteFinalSftManifest(BaseProcessor):
    def __init__(
        self,
        prompt_config: str,
        prompt_template: str,
        code_tags: str,
        chat_format: str | None = None,  # nemotron/llama/None
        input_key: str = "input",
        output_key: str = "output",
        metadata: Optional[Dict] = None,
        exclude_optional_keys: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_key = input_key
        self.output_key = output_key
        self.chat_format = chat_format
        self.metadata = metadata
        self.exclude_optional_keys = exclude_optional_keys
        if not self.metadata:
            self.metadata = {}

        self.prompt = None
        if prompt_config and prompt_template:
            self.prompt = get_prompt(prompt_config, prompt_template, code_tags)
        else:
            if prompt_template:
                LOG.warning(
                    "Prompt template is provided, but prompt config is missing! "
                    "Assuming 'user: {input_key}' and no special formatting for output."
                )
                self.prompt = get_prompt({"user": "{" + input_key + "}"}, prompt_template, code_tags)
            else:
                LOG.warning("Prompt details are missing! The processed data won't be formatted using any prompt.")

    def process(self):
        samples_count = 0
        seen_predictions = defaultdict(set)
        with (
            open(self.input_manifest_file, "rt", encoding="utf-8") as fin,
            open(self.output_manifest_file, "wt", encoding="utf-8") as fout,
        ):
            # only looping over the correct samples (unless asked for incorrect)
            for line in fin:
                elem = json.loads(line)
                question = elem[self.input_key]
                # deduplication
                if elem[self.output_key] in seen_predictions[question]:
                    continue
                seen_predictions[question].add(elem[self.output_key])
                if 'expected_answer' in elem:
                    elem['expected_answer'] = str(elem['expected_answer'])
                # take only required keys from the input if exclude_optional_keys is True
                output_sample = {}
                if not self.exclude_optional_keys:
                    output_sample = json.loads(line)
                elif "expected_answer" in elem:
                    output_sample["expected_answer"] = elem["expected_answer"]

                if self.chat_format is None:
                    generation = elem.pop(self.output_key)
                    if self.prompt:
                        output_sample["input"] = self.prompt.fill(input_dict=elem)
                        output_sample["output"] = generation
                        # not adding end-of-turn for incomplete generations
                        if output_sample.get("finish_reason", "stop") == "stop":
                            output_sample["output"] += self.prompt.config.template.assistant_end
                    else:
                        output_sample["input"] = elem[self.input_key]
                        output_sample["output"] = generation

                elif self.chat_format.lower() == "nemotron":
                    output_sample['conversations'] = [
                        {
                            'value': self.prompt.config.user.format(**elem) if self.prompt else elem[self.input_key],
                            'from': 'User',
                            'canonical_form': '',
                        },
                        {'value': elem.pop(self.output_key), 'from': 'Assistant', 'canonical_form': ''},
                    ]
                    output_sample['system'] = self.prompt.config.system if self.prompt else ''
                    output_sample['mask'] = 'User'
                elif self.chat_format.lower() == "llama":
                    output_sample['conversations'] = [
                        {
                            'value': self.prompt.config.user.format(**elem) if self.prompt else elem[self.input_key],
                            'from': '<|start_header_id|>user<|end_header_id|>',
                            'canonical_form': '',
                        },
                        {
                            'value': elem.pop(self.output_key),
                            'from': '<|start_header_id|>assistant<|end_header_id|>',
                            'canonical_form': '',
                        },
                    ]
                    output_sample['system'] = self.prompt.config.system if self.prompt else ''
                    output_sample['mask'] = '<|start_header_id|>user<|end_header_id|>'
                else:
                    raise ValueError(f"Chat format {self.chat_format} is not supported")
                output_sample.update(self.metadata)
                fout.write(json.dumps(output_sample) + "\n")
                samples_count += 1

        LOG.info("Prepared dataset size: %d", samples_count)


class WriteFinalRLManifest(BaseProcessor):
    def __init__(
        self,
        prompt_config: str,
        prompt_template: str,
        code_tags: str,
        task_name: str | None = None,
        input_key: str = "input",
        metadata: dict | None = None,
        exclude_optional_keys: bool = True,
        random_seed: int = 0,
        do_shuffle: bool = True,
        num_output_samples: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_key = input_key
        self.metadata = metadata
        self.exclude_optional_keys = exclude_optional_keys
        if not self.metadata:
            self.metadata = {}

        self.prompt = None
        if prompt_config and prompt_template:
            self.prompt = get_prompt(prompt_config, prompt_template, code_tags)
        else:
            LOG.warning("Prompt details are missing! The processed data won't be formatted using any prompt.")

        self.random_seed = random_seed
        self.do_shuffle = do_shuffle
        self.num_output_samples = num_output_samples
        self.task_name = task_name

    def process(self):
        samples_count = 0
        all_data = []
        with (open(self.input_manifest_file, "rt", encoding="utf-8") as fin,):
            # only looping over the correct samples (unless asked for incorrect)
            for line in fin:
                elem = json.loads(line)
                if 'expected_answer' in elem:
                    elem['expected_answer'] = str(elem['expected_answer'])
                # take only required keys from the input if exclude_optional_keys is True
                output_sample = {}
                if not self.exclude_optional_keys:
                    output_sample = json.loads(line)
                else:
                    # including only required keys if they are present
                    if "expected_answer" in elem:
                        output_sample["expected_answer"] = elem["expected_answer"]
                    if "problem" in elem:
                        output_sample["problem"] = elem["problem"]

                if self.prompt:
                    output_sample["input"] = self.prompt.fill(input_dict=elem)
                else:
                    output_sample["input"] = elem[self.input_key]

                if self.task_name:
                    output_sample["task_name"] = self.task_name

                output_sample.update(self.metadata)
                all_data.append(output_sample)
                samples_count += 1

        LOG.info("Full dataset size: %d", samples_count)

        if self.do_shuffle:
            random.seed(self.random_seed)
            random.shuffle(all_data)

        if self.num_output_samples is not None:
            all_data = all_data[: self.num_output_samples]
            LOG.info("Downsampled dataset size: %d", len(all_data))

        with open(self.output_manifest_file, "wt", encoding="utf-8") as fout:
            for sample in all_data:
                fout.write(json.dumps(sample) + "\n")


class ConversationDataProcessor(BaseProcessor):
    """
    Process conversation-based data from JSONL files.
    Filters records with status='success' and non-empty problem_statement.
    Creates training samples with location extraction and scoring.
    """
    
    def __init__(
        self,
        input_files: Optional[str] = None,
        calculate_metrics: bool = True,
        tokenizer_name: str = "Qwen/Qwen2.5-32B-Instruct",
        num_workers: Optional[int] = None,
        debug_mode: bool = False,
        max_samples_per_file: Optional[int] = None,
        prompt_config: Optional[str] = None,
        add_custom_tokens: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_files = input_files
        self.calculate_metrics = calculate_metrics
        self.tokenizer_name = tokenizer_name
        self.debug_mode = debug_mode
        self.max_samples_per_file = max_samples_per_file
        self.prompt_config = prompt_config
        self.add_custom_tokens = add_custom_tokens
        
        # In debug mode, use single worker and limit files
        if self.debug_mode:
            self.num_workers = 1
            LOG.info("Debug mode enabled: using single worker")
        else:
            self.num_workers = num_workers if num_workers is not None else min(32, (os.cpu_count() or 1))
        
        self.tokenizer = None
        self.prompt = None
        
        if self.calculate_metrics:
            try:
                from transformers import AutoTokenizer
                
                LOG.info("Loading tokenizer for CPU batch processing")
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, trust_remote_code=True)
                
                # Add custom tokens for structured output
                # Note: <think> and </think> are already in Qwen3 tokenizer
                if self.add_custom_tokens:
                    custom_tokens = ["<tool_call>", "</tool_call>", "<locations>", "</locations>"]
                    num_added_tokens = self.tokenizer.add_tokens(custom_tokens)
                    if num_added_tokens > 0:
                        LOG.info(f"Added {num_added_tokens} custom tokens to tokenizer: {custom_tokens}")
                    else:
                        LOG.info("Custom tokens already present in tokenizer")
                    
            except ImportError:
                LOG.warning("transformers not installed, skipping tokenizer initialization")
                self.calculate_metrics = False
        
        # Load prompt config if provided
        if self.prompt_config:
            try:
                from nemo_skills.prompt.utils import get_prompt
                self.prompt = get_prompt(self.prompt_config, prompt_template=None, code_tags=None)
            except Exception as e:
                LOG.warning(f"Failed to load prompt config {self.prompt_config}: {e}")
                self.prompt = None
        
        if isinstance(self.input_files, str):
            if ',' in self.input_files:
                self.input_files = self.input_files.split(",")
            else:
                self.input_files = self.input_files.split(" ")
        
        if self.input_files is None:
            raise ValueError("input_files must be provided")
    
    def extract_think_content(self, text: str) -> Tuple[str, str]:
        """Extract content inside and outside <think></think> tags."""
        think_pattern = r'<think\s*>(.*?)</think\s*>'
        think_matches = re.findall(think_pattern, text, re.DOTALL | re.IGNORECASE)
        think_content = ' '.join(think_matches)
        outside_content = re.sub(think_pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        return think_content, outside_content
    
    def parse_locations(self, text: str) -> List[Dict[str, Any]]:
        """Parse locations from <locations>file:Lstart-Lend</locations>."""
        locations = []
        location_blocks = re.findall(r'<locations>(.*?)</locations>', text, re.DOTALL)
        for block in location_blocks:
            lines = block.strip().split('\n')
            for line in lines:
                line = line.strip()
                if ':' in line and line:
                    match = re.match(r'^(.+?):(L\d+(?:-L\d+)?)$', line)
                    if match:
                        file_path = match.group(1)
                        line_range = match.group(2)
                        locations.append({'file': file_path, 'range': line_range})
        return locations
    
    def extract_locations_from_patch(self, patch: str) -> List[Dict[str, Any]]:
        """Extract changed line ranges from a git patch."""
        if not patch:
            return []

        locations = []
        current_file = None
        original_line = 0
        new_line = 0
        
        for line in patch.splitlines():
            if line.startswith("--- "):
                file_path = line[4:]
                if file_path.startswith(("a/", "b/")):
                    file_path = file_path[2:]
                current_file = file_path
            elif line.startswith("@@ "):
                m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
                if m:
                    original_line = int(m.group(1))
                    new_line = int(m.group(3))
            elif current_file and line:
                if line.startswith("-") and not line.startswith("---"):
                    locations.append({
                        'file_path': current_file, 
                        'start_line': original_line, 
                        'end_line': original_line, 
                        'raw': f"{current_file}:L{original_line}-L{original_line}"
                    })
                    original_line += 1
                elif line.startswith("+") and not line.startswith("+++"):
                    if not locations or locations[-1]['end_line'] != original_line - 1:
                        locations.append({
                            'file_path': current_file, 
                            'start_line': original_line, 
                            'end_line': original_line, 
                            'raw': f"{current_file}:L{original_line}-L{original_line}"
                        })
                    new_line += 1
                else:
                    original_line += 1
                    new_line += 1

        # Merge adjacent locations
        merged = []
        for loc in locations:
            if merged and loc['file_path'] == merged[-1]['file_path'] and loc['start_line'] <= merged[-1]['end_line'] + 1:
                merged[-1]['end_line'] = max(merged[-1]['end_line'], loc['end_line'])
                merged[-1]['raw'] = f"{merged[-1]['file_path']}:L{merged[-1]['start_line']}-L{merged[-1]['end_line']}"
            else:
                merged.append(loc)
        
        return merged
    
    def calculate_label_and_score(self, ground_truth_locations: List[Dict], predicted_locations: List[Dict]) -> Tuple[int, float]:
        """Calculate label and score based on location matching."""
        if not ground_truth_locations:
            return 0, 0.0
        if not predicted_locations:
            return 0, 0.0
        
        gt_files = {loc['file_path'] for loc in ground_truth_locations}
        pred_files = {loc['file'] for loc in predicted_locations}
        
        file_match = bool(gt_files & pred_files)
        label = 1 if file_match else 0
        
        if not file_match:
            score = 0.0
        else:
            gt_locations_set = set()
            for loc in ground_truth_locations:
                if loc['start_line'] == loc['end_line']:
                    range_str = f"L{loc['start_line']}"
                else:
                    range_str = f"L{loc['start_line']}-L{loc['end_line']}"
                gt_locations_set.add((loc['file_path'], range_str))
            
            pred_locations_set = {(loc['file'], loc['range']) for loc in predicted_locations}
            
            if gt_locations_set & pred_locations_set:
                score = 1.0
            else:
                score = 0.7
        
        return label, score
    
    def extract_user_query(self, inputs: Any) -> str:
        """Extract the user query from the inputs field."""
        if isinstance(inputs, dict):
            if 'query' in inputs:
                return inputs['query'].strip()
            elif 'text' in inputs:
                return inputs['text'].strip()
            else:
                return str(inputs).strip()
        elif isinstance(inputs, str):
            return inputs.strip()
        else:
            return str(inputs).strip()
    
    def process_record(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a single record to create multiple training samples using prefix subsequences.
        
        For each trajectory (multi-turn conversation), we create training samples by taking
        prefix subsequences: each sample contains the conversation history up to turn N as input,
        and the assistant response at turn N as the target output. This allows the model to
        learn from the complete reasoning process across multiple turns.
        """
        turns = record.get("turns", [])
        if not turns:
            return []
        
        training_samples = []
        patch = record.get("patch", "")
        ground_truth_locations = self.extract_locations_from_patch(patch)
        
        # Collect all texts for batch tokenization if metrics are needed
        all_formatted_inputs = []
        all_outputs = []
        all_think_contents = []
        all_outside_contents = []
        
        for turn_idx in range(len(turns)):
            # Build messages list for chat template
            messages = []
            
            # Add system message from prompt config
            system_content = ""
            if self.prompt and self.prompt.config.system:
                try:
                    # Try to format with record data, fallback to plain system message
                    system_content = self.prompt.config.system.format(**record)
                except (KeyError, ValueError):
                    # If formatting fails, use the system message as-is
                    system_content = self.prompt.config.system
            messages.append({"role": "system", "content": system_content})
            
            # Add conversation history up to current turn
            for i in range(turn_idx + 1):
                user_input = self.extract_user_query(turns[i].get("inputs", ""))
                messages.append({"role": "user", "content": user_input})
                
                # For history turns (not current), add assistant response with thinking
                if i < turn_idx:
                    assistant_response = turns[i].get("assistant_raw_w_think", "")
                    messages.append({"role": "assistant", "content": assistant_response})
            
            # Get assistant responses first
            assistant_raw = turns[turn_idx].get("assistant_raw", "")
            assistant_raw_w_think = turns[turn_idx].get("assistant_raw_w_think", "")
            
            # Apply chat template to get formatted input
            if not self.tokenizer:
                raise ValueError("Tokenizer is required for chat template formatting")
            if not hasattr(self.tokenizer, 'apply_chat_template'):
                raise ValueError(f"Tokenizer {self.tokenizer_name} does not support apply_chat_template method")
            
            # For training data, we need the complete conversation including the assistant response
            # So we add the assistant response to messages and format the complete conversation
            messages_with_response = messages.copy()
            messages_with_response.append({"role": "assistant", "content": assistant_raw_w_think})
            
            # Get the complete formatted conversation
            complete_conversation = self.tokenizer.apply_chat_template(
                messages_with_response, tokenize=False, add_generation_prompt=False
            )
            
            # Extract the input part (everything except the assistant response)
            formatted_input = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            predicted_locations = self.parse_locations(assistant_raw_w_think)
            label, score = self.calculate_label_and_score(ground_truth_locations, predicted_locations)
            
            # Create unique prefix subsequence ID with task instance and turn index
            # We construct training samples from each trajectory using prefix subsequences:
            # each sample contains the conversation history up to turn N as input, 
            # and the assistant response at turn N as the target output
            task_instance = record.get("instance_id", "")
            
            # Extract the properly formatted assistant output from the complete conversation
            # The tokenizer handles the assistant_end token automatically
            output_with_proper_formatting = complete_conversation[len(formatted_input):]
            
            if self.debug_mode and turn_idx == 0:
                LOG.info(f"Debug formatting:")
                LOG.info(f"  formatted_input length: {len(formatted_input)}")
                LOG.info(f"  complete_conversation length: {len(complete_conversation)}")
                LOG.info(f"  output_with_proper_formatting length: {len(output_with_proper_formatting)}")
                LOG.info(f"  output_with_proper_formatting preview: {output_with_proper_formatting[:100]}...")
            
            training_sample = {
                "task_instance": task_instance,
                "prefix_subsequence_id": f"{task_instance}_{turn_idx}",
                "expected_answer": assistant_raw,
                "input": formatted_input,
                "output": output_with_proper_formatting,
                "label": label,
                "score": score,
                "ground_truth_locations": [loc['raw'] for loc in ground_truth_locations],
                "predicted_locations": [f"{loc['file']}:{loc['range']}" for loc in predicted_locations],
                "turn_index": turn_idx,
                "total_turns": len(turns)
            }
            
            # Collect texts for batch tokenization
            if self.calculate_metrics and self.tokenizer:
                think_content, outside_content = self.extract_think_content(assistant_raw_w_think)
                all_formatted_inputs.append(formatted_input)
                all_outputs.append(output_with_proper_formatting)
                all_think_contents.append(think_content if think_content else "")
                all_outside_contents.append(outside_content if outside_content else "")
            
            training_samples.append(training_sample)
        
        # Batch tokenize all texts at once for better performance
        if self.calculate_metrics and self.tokenizer and all_formatted_inputs:
            # Combine all texts for batch processing
            all_texts = all_formatted_inputs + all_outputs + [t for t in all_think_contents if t] + [t for t in all_outside_contents if t]
            
            if all_texts:
                # Batch tokenize on CPU for optimal performance with multiple workers
                LOG.info(f"CPU batch tokenization for {len(all_texts)} texts...")
                all_token_ids = self.tokenizer(all_texts, add_special_tokens=False)['input_ids']
                all_token_counts = [len(tokens) for tokens in all_token_ids]
                
                # Split counts back to categories
                num_samples = len(training_samples)
                input_counts = all_token_counts[:num_samples]
                output_counts = all_token_counts[num_samples:2*num_samples]
                
                # Handle think and non-think counts
                think_counts = []
                non_think_counts = []
                think_start_idx = 2 * num_samples
                non_think_start_idx = think_start_idx + sum(1 for t in all_think_contents if t)
                
                think_idx = 0
                non_think_idx = 0
                for i in range(num_samples):
                    if all_think_contents[i]:
                        think_counts.append(all_token_counts[think_start_idx + think_idx])
                        think_idx += 1
                    else:
                        think_counts.append(0)
                    
                    if all_outside_contents[i]:
                        non_think_counts.append(all_token_counts[non_think_start_idx + non_think_idx])
                        non_think_idx += 1
                    else:
                        non_think_counts.append(0)
                
                # Update training samples with token counts
                for i, sample in enumerate(training_samples):
                    sample.update({
                        "input_tokens": input_counts[i],
                        "output_tokens": output_counts[i],
                        "think_tokens": think_counts[i],
                        "non_think_tokens": non_think_counts[i],
                        "total_tokens": input_counts[i] + output_counts[i]
                    })
        
        return training_samples
    
    def process_file(self, file_path: str) -> Tuple[List[Dict], int, int, int]:
        """Process a single JSONL file."""
        training_samples = []
        total_records = 0
        success_records = 0
        empty_problem_statement_records = 0
        processed_samples = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if we've reached the sample limit
                    if self.max_samples_per_file is not None and processed_samples >= self.max_samples_per_file:
                        LOG.info(f"Reached max_samples_per_file limit ({self.max_samples_per_file}) for {file_path}")
                        break
                    
                    try:
                        record = json.loads(line)
                        total_records += 1
                        
                        if self.debug_mode and total_records <= 3:
                            LOG.info(f"Record {total_records}: status='{record.get('status')}', has_problem_statement={bool(record.get('problem_statement', '').strip())}")
                        
                        if record.get('status') == 'success':
                            success_records += 1
                            problem_statement = record.get('problem_statement', '').strip()
                            if not problem_statement:
                                empty_problem_statement_records += 1
                                if self.debug_mode:
                                    LOG.warning(f"Record {total_records} has empty problem_statement")
                            else:
                                record_samples = self.process_record(record)
                                if self.debug_mode:
                                    LOG.info(f"Record {total_records} generated {len(record_samples)} training samples")
                                training_samples.extend(record_samples)
                                processed_samples += 1
                        elif self.debug_mode and total_records <= 3:
                            LOG.warning(f"Record {total_records} filtered out: status='{record.get('status')}'")
                    except json.JSONDecodeError as e:
                        if self.debug_mode:
                            LOG.error(f"JSON decode error on line: {e}")
                        pass
        except Exception as e:
            LOG.error(f"Error processing file {file_path}: {e}")
        
        return training_samples, total_records, success_records, empty_problem_statement_records
    
    @staticmethod
    def _process_file_wrapper(args):
        """Wrapper for multiprocessing that recreates tokenizer in each process."""
        file_path, tokenizer_name, calculate_metrics, debug_mode, max_samples_per_file, prompt_config, add_custom_tokens = args
        
        # Create a temporary processor instance for this worker
        processor = ConversationDataProcessor(
            input_files=[file_path],
            calculate_metrics=calculate_metrics,
            tokenizer_name=tokenizer_name,
            debug_mode=debug_mode,
            max_samples_per_file=max_samples_per_file,
            prompt_config=prompt_config,
            add_custom_tokens=add_custom_tokens,
            num_workers=1,  # Single worker for this instance
            output_manifest_file="/tmp/dummy_output.jsonl"  # Dummy output file since we return results directly
        )
        
        # Note: Custom tokens are added during processor initialization
        
        # Process just this one file
        result = processor.process_file(file_path)
        return result + (file_path,)  # Add file path to result for logging
    
    def process(self):
        """Process all input files and write training samples using multiprocessing."""
        all_samples = []
        total_records = 0
        total_success_records = 0
        total_empty_problem_statement = 0
        all_metrics = []
        
        # Collect all files to process
        all_files = []
        for file_pattern in self.input_files:
            all_files.extend(unroll_files([file_pattern]))
        
        # In debug mode, process only the first file
        if self.debug_mode and all_files:
            all_files = all_files[:1]
            LOG.info(f"Debug mode: processing only first file: {all_files[0]}")
            if self.max_samples_per_file:
                LOG.info(f"Debug mode: limiting to {self.max_samples_per_file} samples per file")
        
        LOG.info(f"Found {len(all_files)} files to process using {self.num_workers} workers")
        
        # Prepare arguments for multiprocessing
        process_args = [(f, self.tokenizer_name, self.calculate_metrics, self.debug_mode, self.max_samples_per_file, self.prompt_config, self.add_custom_tokens) for f in all_files]
        
        # Process files - use multiprocessing only if not in debug mode
        if self.debug_mode or self.num_workers == 1:
            # Process sequentially for debug mode or single worker
            results = []
            for args in process_args:
                LOG.info(f"Processing file: {args[0]}")
                result = ConversationDataProcessor._process_file_wrapper(args)
                results.append(result)
        else:
            # Process files in parallel using CPU batch tokenization
            with Pool(processes=self.num_workers) as pool:
                # Use tqdm for progress bar if available
                try:
                    from tqdm import tqdm
                    results = list(tqdm(
                        pool.imap_unordered(ConversationDataProcessor._process_file_wrapper, process_args),
                        total=len(all_files),
                        desc="Processing files"
                    ))
                except ImportError:
                    results = pool.map(ConversationDataProcessor._process_file_wrapper, process_args)
        
        # Aggregate results
        for samples, records, success, empty, file_path in results:
            LOG.info(f"Processed {file_path}: {len(samples)} samples, {records} total records, {success} success records, {empty} empty problem statements")
            if self.debug_mode and samples:
                LOG.info(f"First sample type: {type(samples[0])}")
                LOG.info(f"First sample keys: {samples[0].keys() if isinstance(samples[0], dict) else 'Not a dict'}")
                if isinstance(samples[0], dict) and len(str(samples[0])) < 500:
                    LOG.info(f"First sample content: {samples[0]}")
            elif self.debug_mode and not samples:
                LOG.warning(f"No samples generated from {file_path} - check if records have status='success' and non-empty problem_statement")
            all_samples.extend(samples)
            total_records += records
            total_success_records += success
            total_empty_problem_statement += empty
            
            if self.calculate_metrics:
                for sample in samples:
                    metrics = {
                        "label": sample["label"],
                        "score": sample["score"],
                        "num_turns": sample["turn_index"] + 1,
                        "turn_index": sample["turn_index"],
                        "total_turns": sample["total_turns"]
                    }
                    if "input_tokens" in sample:
                        metrics.update({
                            "input_tokens": sample["input_tokens"],
                            "output_tokens": sample["output_tokens"],
                            "think_tokens": sample["think_tokens"],
                            "non_think_tokens": sample["non_think_tokens"],
                            "total_tokens": sample["total_tokens"]
                        })
                    all_metrics.append(metrics)
        
        LOG.info(f"Total records processed: {total_records}")
        LOG.info(f"Success records found: {total_success_records}")
        LOG.info(f"Records filtered (empty problem_statement): {total_empty_problem_statement}")
        LOG.info(f"Total training samples created: {len(all_samples)}")
        
        if total_records > 0:
            LOG.info(f"Success rate: {total_success_records/total_records*100:.2f}%")
        if total_success_records > 0:
            LOG.info(f"Empty problem_statement rate: {total_empty_problem_statement/total_success_records*100:.2f}%")
        
        if all_metrics and self.calculate_metrics:
            self._log_statistics(all_metrics)
        
        with open(self.output_manifest_file, "wt", encoding="utf-8") as fout:
            for idx, sample in enumerate(all_samples):
                if self.debug_mode and idx < 3:  # Log first 3 samples in debug mode
                    LOG.info(f"Sample {idx}: {json.dumps(sample, indent=2)}")
                fout.write(json.dumps(sample) + "\n")
    
    def _log_statistics(self, all_metrics: List[Dict[str, Any]]):
        """Log comprehensive statistics."""
        turn_counts = [m['num_turns'] for m in all_metrics]
        LOG.info(f"\nTurn statistics:")
        LOG.info(f"  Average turns: {mean(turn_counts):.2f}")
        LOG.info(f"  Min/Max turns: {min(turn_counts)}/{max(turn_counts)}")
        
        labels = [m['label'] for m in all_metrics]
        positive_count = sum(labels)
        LOG.info(f"\nLabel statistics:")
        LOG.info(f"  Positive samples: {positive_count} ({positive_count/len(labels)*100:.2f}%)")
        
        scores = [m['score'] for m in all_metrics]
        LOG.info(f"\nScore statistics:")
        LOG.info(f"  Average score: {mean(scores):.3f}")
        LOG.info(f"  Score distribution: 0.0={sum(1 for s in scores if s == 0.0)}, 0.7={sum(1 for s in scores if s == 0.7)}, 1.0={sum(1 for s in scores if s == 1.0)}")
        
        if 'input_tokens' in all_metrics[0]:
            input_tokens = [m['input_tokens'] for m in all_metrics]
            output_tokens = [m['output_tokens'] for m in all_metrics]
            think_tokens = [m['think_tokens'] for m in all_metrics]
            LOG.info(f"\nToken statistics:")
            LOG.info(f"  Input tokens - Average: {mean(input_tokens):.2f}")
            LOG.info(f"  Output tokens - Average: {mean(output_tokens):.2f}")
            LOG.info(f"  Thinking tokens - {sum(think_tokens)/sum(output_tokens)*100:.1f}% of output")
