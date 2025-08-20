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
import random
import re
from dataclasses import asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from transformers import AutoTokenizer

from nemo_skills.code_execution.utils import format_code_output
from nemo_skills.prompt.few_shot_examples import examples_map
from nemo_skills.utils import get_logger_name, nested_dataclass

LOG = logging.getLogger(get_logger_name(__file__))


class BM25Retriever:
    def __init__(self, data_path: str, field: str):
        from rank_bm25 import BM25Okapi

        with open(data_path, "rt", encoding="utf-8") as fin:
            self.entries = [json.loads(x) for x in fin]

        corpus = [entry[field] for entry in self.entries]
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 1):
        tokenized_query = query.split(" ")
        return self.bm25.get_top_n(tokenized_query, self.entries, n=top_k)


@nested_dataclass(kw_only=True)
class FewShotExamplesConfig:
    prefix: str = ""
    template: str = ""
    suffix: str = ""

    examples_type: Optional[str] = None

    retrieval_field: Optional[str] = None  # e.g. question, reference_solution, etc.
    retrieval_file: Optional[str] = None  # needs to be provided if retrieval_field is not None
    retrieved_entries: int = 10  # need to set higher than few_shots to filter out exact matches
    retrieved_few_shots: int = 5
    randomize_retrieved_entries: bool = False
    max_retrieved_chars: int = 100000000  # no limit by default
    max_retrieved_chars_field: str = "reference_solution"
    retriever: Optional[Any] = None

    def __post_init__(self):
        """Error checks + building example_dicts and retriever if needed."""
        if self.examples_type is not None and self.retriever is not None:
            raise ValueError("examples_type and retriever cannot be used together")

        if self.retriever is not None:
            return

        if self.retrieval_field is not None:  # building retriever
            if self.retrieval_file is None:
                raise ValueError("retrieval_file must be provided if retrieval_field is not None")
            self.retriever = BM25Retriever(self.retrieval_file, field=self.retrieval_field)
        else:
            if self.retrieval_file is not None:
                raise ValueError("retrieval_field must be provided if retrieval_file is not None")


@nested_dataclass(kw_only=True)
class CodeTags:
    # used to execute code within these tags
    code_begin: str = '```python\n'
    code_end: str = '```\n'

    # used to extract the code output
    code_output_begin: str = '```output\n'
    code_output_end: str = '```\n'

    # used to post-process code output
    code_output_format: str = 'qwen'


@nested_dataclass(kw_only=True)
class PromptConfig:
    user: str
    system: str | None = None
    code_tags: CodeTags = None
    few_shot_examples: FewShotExamplesConfig = field(default_factory=FewShotExamplesConfig)


class Prompt:
    def __init__(self, config, tokenizer):
        # rebuilding prompt config to make sure post init is called again in
        # case some parameters were manually changed after the config was created
        self.config = PromptConfig(_init_nested=True, **asdict(config))
        self.tokenizer = tokenizer
        if self.tokenizer:
            # assuming it's the object already if not str
            if isinstance(self.tokenizer, str):
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)

    def build_filled_example(self, example_dict: Dict[str, Any]) -> str:
        """Builds a filled example string based on the example dictionary."""

        # replacing code/code-output separators in the examples if present
        example_dict = example_dict.copy()
        if 'solution' in example_dict and self.config.code_tags:

            def replace_code_output(match):
                code_output = match.group(2)
                formatted_output = format_code_output(
                    execution_dict={"process_status": "completed", "stdout": code_output, "stderr": ""},
                    code_output_begin=self.config.code_tags.code_output_begin,
                    code_output_end=self.config.code_tags.code_output_end,
                    code_output_format=self.config.code_tags.code_output_format,
                )
                return formatted_output

            pattern = r'({code_output_begin}\n)(.*?)({code_output_end})'
            example_dict["solution"] = re.sub(pattern, replace_code_output, example_dict["solution"], flags=re.DOTALL)

            example_dict["solution"] = example_dict["solution"].replace(
                "{code_begin}", self.config.code_tags.code_begin
            )
            example_dict["solution"] = example_dict["solution"].replace("{code_end}", self.config.code_tags.code_end)
            example_dict["solution"] = example_dict["solution"].replace("{code_output_begin}", "")
            example_dict["solution"] = example_dict["solution"].replace("{code_output_end}", "")

        return self.config.few_shot_examples.template.format(**example_dict)

    def build_examples_dict(self, input_dict):
        if self.config.few_shot_examples.examples_type:
            return examples_map[self.config.few_shot_examples.examples_type.format(**input_dict)]

        if self.config.few_shot_examples.retriever is None:
            return []

        example_dicts = self.config.few_shot_examples.retriever.retrieve(
            query=input_dict[self.config.few_shot_examples.retrieval_field],
            top_k=self.config.few_shot_examples.retrieved_entries,
        )
        reference = input_dict[self.config.few_shot_examples.retrieval_field]
        # filtering exact match if it's there
        while example_dicts and example_dicts[0][self.config.few_shot_examples.retrieval_field] == reference:
            example_dicts = example_dicts[1:]

        # removing too long solutions
        example_dicts = [
            example_dict
            for example_dict in example_dicts
            if len(example_dict[self.config.few_shot_examples.max_retrieved_chars_field])
            < self.config.few_shot_examples.max_retrieved_chars
        ]

        if len(example_dicts) < self.config.few_shot_examples.retrieved_few_shots:
            LOG.warning(
                'Too little examples (%d) found for the query "%s"',
                len(example_dicts),
                input_dict[self.config.few_shot_examples.retrieval_field],
            )

        # let's reverse the order to show the most relevant last
        examples = example_dicts[: self.config.few_shot_examples.retrieved_few_shots][::-1]
        if self.config.few_shot_examples.randomize_retrieved_entries:
            random.shuffle(examples)

        return examples

    def build_user_message(self, input_dict: Dict[str, str]) -> str:
        """Builds all examples string concatenated by delimiter."""
        example_dicts = self.build_examples_dict(input_dict)

        filled_examples = "".join([self.build_filled_example(example) for example in example_dicts])
        if not filled_examples:
            examples = ""
        else:
            examples = f"{self.config.few_shot_examples.prefix}{filled_examples}{self.config.few_shot_examples.suffix}"
        user = self.config.user.format(examples=examples, **input_dict)
        return user

    def get_code_execution_args(self):
        """Returns the code execution arguments."""
        if self.config.code_tags is None:
            raise ValueError(
                "Please provide 'code_tags' in your prompt configuration before calling get_code_execution_args()."
            )
        return {
            "code_begin": self.config.code_tags.code_begin,
            "code_end": self.config.code_tags.code_end,
            "code_output_begin": self.config.code_tags.code_output_begin,
            "code_output_end": self.config.code_tags.code_output_end,
            "code_output_format": self.config.code_tags.code_output_format,
        }

    def add_assistant_end_suffix(self, assistant_response: str) -> str:
        """Adds special tokens to the end of assistant response."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not set.")

        messages = [{'role': 'user', 'content': ''}]

        user_string = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        messages.append({'role': 'assistant', 'content': assistant_response})
        assistant_string = self.tokenizer.apply_chat_template(messages, tokenize=False)

        assert assistant_string.startswith(user_string), f"Something is wrong\n{user_string}\n||\n{assistant_string}"

        formatted_response = assistant_string[len(user_string) :]

        return formatted_response

    def fill(
        self,
        input_dict: Dict[str, str],
        start_assistant_response_key: str | None = None,
    ) -> str | List[dict]:
        """
        Fills the prompt with the input_dict.
        Operates in two modes:
        - If `self.tokenizer` is set, it will use it to format the prompt, returning a string.
        - If `self.tokenizer` is not set, it will assume chat format and return a list of dictionaries.

        Args:
            input_dict: The input dictionary to fill the prompt with.
            start_assistant_response_key: Whether to append the value of this key to the beginning of assistant response.

        Returns:
            The filled prompt - either a string or a list of dictionaries.
        """

        if self.config.system is not None:
            messages = [
                {"role": "system", "content": self.config.system},
            ]
        else:
            messages = []
        messages.append({"role": "user", "content": self.build_user_message(input_dict)})

        if start_assistant_response_key and self.tokenizer is None:
            raise ValueError(
                f"start_assistant_response_key is '{start_assistant_response_key}', but tokenizer is not set. "
                "It's not possible to start assistant response with openai messages "
                "format, so please set tokenizer to a valid value."
            )

        if self.tokenizer is not None:
            try:
                messages_string = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except ValueError as e:
                if "Cannot use chat template functions because tokenizer.chat_template is not set" in str(e):
                    # assuming that's a base model and we just need to add bos
                    if len(messages) != 1 or messages[0]['role'] != 'user':
                        raise ValueError(
                            "The model doesn't support chat template, can't format messages which contain non-user values"
                        )
                    if hasattr(self.tokenizer, 'bos_token'):
                        messages_string = self.tokenizer.bos_token + messages[0]['content']
                    else:
                        messages_string = messages[0]['content']
            if start_assistant_response_key:
                messages_string += input_dict[start_assistant_response_key]
            return messages_string

        return messages

    def __str__(self):
        return str(self.config)


def get_config_path(config: str, config_dir: str | None = None, config_extension: str = "yaml") -> Path:
    if config_dir is None:
        config_dir = str(Path(__file__).parent.absolute() / 'config')

    if config.endswith(f".{config_extension}"):
        config_path = Path(config).absolute()
    elif config.startswith("nemo_skills"):
        config_path = Path(__file__).parents[2].absolute() / f"{config}.{config_extension}"
    else:
        config_path = Path(config_dir) / f"{config}.{config_extension}"

    return config_path


def load_config(config: str, config_dir: str | None = None) -> dict:
    """
    Reads the prompt config/template from the yaml file.

    Args:
        config (str): The location of the prompt config file.
            Can be the full path to a yaml file (if ends with .yaml) or one of the available configs.
            If configs starts with nemo_skills we will look relative to the repo root.
            If not, we will look relative to the config_dir parameter
        config_dir (str): The dir to look for the config file.

    Returns:
        The loaded dictionary.
    """
    config_path = get_config_path(config, config_dir)

    with open(config_path, "rt", encoding="utf-8") as fin:
        return yaml.safe_load(fin)


def get_prompt(
    prompt_config: str | dict,
    tokenizer: Any | None = None,
    code_tags: str | dict | None = None,
    examples_type: str | None = None,
    config_dir: str | None = None,
    code_tags_dir: str | None = None,
) -> Prompt:
    if code_tags_dir is None:
        code_tags_dir = Path(__file__).parent.absolute() / 'code_tags'

    if isinstance(prompt_config, str):
        config = load_config(prompt_config, config_dir)
    else:
        config = prompt_config

    code_tags_obj = None
    if code_tags is not None:
        if isinstance(code_tags, str):
            code_tags_dict = load_config(code_tags, code_tags_dir)
        else:
            code_tags_dict = code_tags
        code_tags_obj = CodeTags(**code_tags_dict)

    prompt = Prompt(PromptConfig(**config, code_tags=code_tags_obj), tokenizer=tokenizer)

    if examples_type is not None:
        prompt.config.few_shot_examples.examples_type = examples_type

    return prompt
