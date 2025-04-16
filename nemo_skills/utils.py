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

import glob
import inspect
import io
import logging
import os
import re
import sys
import tokenize
import typing
import fire
from dataclasses import MISSING, dataclass, fields, is_dataclass
from typing import Any, List, Optional
from pathlib import Path

from rich.logging import RichHandler
from fire import decorators as fire_decorators

# isort: off
from nemo_skills.file_utils import (
    jdump,
    jload,
    jload_chunk,
    count_newlines,
    calculate_chunk_indices,
    unroll_files
)
# isort: on


def nested_dataclass(*args, **kwargs):
    """Decorator that will recursively instantiate all nested dataclasses.

    Adapted from https://www.geeksforgeeks.org/creating-nested-dataclass-objects-in-python/.
    """

    def wrapper(check_class):
        try:
            from omegaconf.dictconfig import DictConfig

            dict_types = (dict, DictConfig)
        except ImportError:
            dict_types = (dict,)

        # passing class to investigate
        check_class = dataclass(check_class, **kwargs)
        orig_init = check_class.__init__

        def __init__(self, *, _init_nested=False, **kwargs):
            if _init_nested:
                for name, value in kwargs.items():
                    # getting field type
                    ft = check_class.__annotations__.get(name, None)

                    if is_dataclass(ft) and isinstance(value, dict_types):
                        obj = ft(**value, _init_nested=_init_nested)
                        kwargs[name] = obj
            orig_init(self, **kwargs)

        check_class.__init__ = __init__

        return check_class

    return wrapper(args[0]) if args else wrapper


def setup_logging(disable_hydra_logs: bool = True, log_level: int = logging.INFO, use_rich: bool = False):
    logger = logging.getLogger()
    logger.setLevel(log_level)

    if use_rich:
        handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            show_time=False,
        )
        for hdlr in logger.handlers[:]:
            logger.removeHandler(hdlr)
    else:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(levelname)s  %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
    logger.addHandler(handler)
    logging.getLogger("sshtunnel_requests.cache").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if disable_hydra_logs:
        # hacking the arguments to always disable hydra's output
        sys.argv.extend(
            ["hydra.run.dir=.", "hydra.output_subdir=null", "hydra/job_logging=none", "hydra/hydra_logging=none"]
        )


def init_wandb(project, name, exp_dir=None, verbose=False):
    """
    Initialize wandb if the API key is set.
    Returns true if Wandb is initialized, false otherwise.

    Args:
        project (str): Wandb project name.
        name (str): Wandb run name.
        exp_dir (str, optional): Directory for experiment logs. Defaults to None.
        verbose (bool, optional): If True, prints debug information. Defaults to False.

    Returns:
        bool: True if wandb is initialized, False otherwise.
    """
    try:
        import wandb
    except (ImportError, ModuleNotFoundError):
        if verbose: print("Wandb is not installed. Skipping wandb initialization.")
        return False

    # Check if the project or name is None, and skip initialization if so
    if project is None or name is None:
        if verbose: print("Wandb project or name not provided. Skipping wandb initialization.")
        return False

    # Determine the log directory based on the provided exp_dir
    if exp_dir is None:
        log_dir = os.path.join(os.getcwd(), "experiment_logs", "wandb")
    else:
        log_dir = os.path.join(exp_dir, "wandb", project, name)
    log_dir = os.path.abspath(log_dir)

    # Create the log directory if it does not exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Initialize wandb with the specified parameters
    try:
        wandb.init(project=project, name=name, resume='auto', reinit=True, save_code=True, dir=log_dir)
        if verbose: print("Wandb initialized.")
        return True
    except Exception as e:
        if verbose:
            print("Wandb initialization failed with the following error.")
            print(e)
        return False



def extract_comments(code: str):
    """Extract a list of comments from the given Python code."""
    comments = []
    tokens = tokenize.tokenize(io.BytesIO(code.encode()).readline)

    for token, line, *_ in tokens:
        if token is tokenize.COMMENT:
            comments.append(line.lstrip('#').strip())

    return comments


def type_to_str(type_hint):
    """Convert type hints to a more readable string."""
    origin = typing.get_origin(type_hint)
    args = typing.get_args(type_hint)

    if hasattr(type_hint, '__name__'):
        return type_hint.__name__.replace('NoneType', 'None')
    elif origin is typing.Union:
        if len(args) == 2 and type(None) in args:
            return f'Optional[{type_to_str(args[0])}]'
        else:
            return ' or '.join(type_to_str(arg) for arg in args)
    elif origin is typing.Callable:
        if args[0] is Ellipsis:
            args_str = '...'
        else:
            args_str = ', '.join(type_to_str(arg) for arg in args[:-1])
        return f'Callable[[{args_str}], {type_to_str(args[-1])}]'
    elif origin:
        inner_types = ', '.join(type_to_str(arg) for arg in args)
        origin_name = origin.__name__ if hasattr(origin, '__name__') else str(origin)
        return f'{origin_name}[{inner_types}]'
    else:
        return str(type_hint).replace('typing.', '')


def extract_comments_above_fields(dataclass_obj, prefix: str = '', level: int = 0, **kwargs):
    source_lines = inspect.getsource(dataclass_obj).split('\n')
    fields_info = {
        field.name: {
            'type': field.type,
            'default': field.default if field.default != MISSING else None,
            'default_factory': field.default_factory if field.default_factory != MISSING else None,
        }
        for field in fields(dataclass_obj)
    }
    comments, comment_cache = {}, []

    for line in source_lines:
        # skip unfinished multiline comments
        line_comment = []
        if '#' in line:
            line_comment = extract_comments(line)
        if line_comment:
            comment_cache.append(line_comment[0])
        if ':' not in line:
            continue

        field_name = line.split(':')[0].strip()
        if field_name not in fields_info:
            continue

        field_info = fields_info[field_name]
        field_name = prefix + field_name
        field_type = type_to_str(field_info['type'])
        default = field_info['default']
        default_factory = field_info['default_factory']
        if default == '???':
            default_str = ' = MISSING'
        else:
            default_str = f' = {default}'
        if default_factory:
            try:
                default_factory = default_factory()
                default_str = f' = {default_factory}'
            except:
                pass
            if is_dataclass(default_factory):
                default_str = f' = {field_type}()'

        indent = '  ' * level
        comment = f"\n{indent}".join(comment_cache)
        comment = "- " + comment if comment else ""
        comment = comment.replace('\n', f'\n{indent}  ')
        field_detail = f"{indent}\033[92m{field_name}: {field_type}{default_str}\033[0m {comment}"
        comments[field_name] = field_detail
        comment_cache = []

        # Recursively extract nested dataclasses
        if is_dataclass(field_info['type']):
            nested_comments = extract_comments_above_fields(
                field_info['type'], prefix=field_name + '.', level=level + 1
            )
            for k, v in nested_comments.items():
                comments[f"{field_name}.{k}"] = v

    return comments


def get_fields_docstring(dataclass_obj, **kwargs):
    commented_fields = extract_comments_above_fields(dataclass_obj, **kwargs)
    docstring = [content for content in commented_fields.values()]
    return '\n'.join(docstring)


def get_help_message(dataclass_obj, help_message="", **kwargs):
    heading = """
This script uses Hydra (https://hydra.cc/) for dynamic configuration management.
You can apply Hydra's command-line syntax for overriding configuration values directly.
Below are the available configuration options and their default values:
    """.strip()

    docstring = get_fields_docstring(dataclass_obj)
    # to handle {} in docstring.
    docstring = docstring.replace('{}', '{{}}')
    # to handle any dictionaries as defaults (replacing {...} with {{...}} if there is a space inside)
    docstring = re.sub(r'{([^}]+(?=\s)[^}]*)}', r'{{\1}}', docstring)
    # Might need to add some other edge-case handling
    # here, so that formatting does not complain
    docstring = docstring.format(**kwargs)

    full_help = f"{heading}\n{'-' * 75}\n{docstring}"
    if help_message:
        full_help = f"{help_message}\n\n{full_help}"

    return full_help


def python_doc_to_cmd_help(doc_class, docs_prefix="", arg_prefix=""):
    """Converts python doc to cmd help format.

    Will color the args and change the format to match what we use in cmd help.
    """
    all_args = docs_prefix
    all_args += doc_class.__doc__.split("Args:")[1].rstrip()
    # \033[92m ... \033[0m - green in terminal
    colored_args = ""
    for line in all_args.split("\n"):
        if "        " in line and " - " in line:
            # add colors
            line = line.replace("        ", "        \033[92m").replace(" - ", "\033[0m - ")
            # fixing arg format
            line = line.replace('        \033[92m', f'        \033[92m{arg_prefix}')
        # fixing indent
        line = line.replace("        ", "    ").replace("    ", "  ")
        colored_args += line + '\n'
    return colored_args[:-1]


def get_chunked_filename(chunk_id, output_filename):
    basename, ext = os.path.splitext(output_filename)
    return f"{basename}_chunk_{chunk_id}{ext}"


def chunk_data(data: List[Any], output_filename: str, chunk_id: Optional[int], num_chunks: Optional[int]):
    """
    Chunk data if chunk_id and num_chunks are provided.

    Args:
        data: List of dictionaries to be chunked.
        output_filename: Original output filename.
        chunk_id: Chunk ID (0-indexed).
        num_chunks: Number of chunks to split the data into.

    Returns:
        Tuple of chunked data and chunked output filename.
    """
    # Chunk instruction_data if chunk_id and num_chunks are provided
    if chunk_id is not None:
        chunk_id = int(chunk_id)
    if num_chunks is not None:
        num_chunks = int(num_chunks)
    if chunk_id is not None and num_chunks is not None:
        if chunk_id < 0 or chunk_id >= num_chunks:
            raise ValueError(
                f"Invalid chunk_id or num_chunks. chunk_id: {chunk_id}, num_chunks: {num_chunks}.\n"
                f"chunk_id should be in the range [0, num_chunks)."
            )

        start_idx, end_idx = calculate_chunk_indices(len(data), num_chunks, chunk_id)
        data = data[start_idx:end_idx]

        if len(data) > 0:
            logging.info(f"Processing chunk {chunk_id + 1}/{num_chunks} with {len(data)} samples.")

        # Modify output_filename to include chunk_id
        output_filename = get_chunked_filename(chunk_id, output_filename)
        logging.info(f"Chunked Output filename: {output_filename}")

    return data, output_filename


def str_ids_to_list(ids: str) -> list[int]:
    """
    Convert a string of comma or .. separated ids to a list of ids.

    Args:
        ids: Comma separated list of ids.

    Returns:
        List of ids.
    """
    if ',' in ids and '..' in ids:
        raise ValueError(
            "Invalid chunk ids format. Can be a comma separated list or a range separated by '..' but not both"
        )
    if ',' in ids:
        ids = ids.split(',')
        ids = [int(x.strip()) for x in ids if x.strip() != '']
    elif '..' in ids:
        start, end = ids.split('..')
        ids = list(range(int(start), int(end) + 1))
    else:
        try:  # could be a single number
            ids = [int(ids)]
        except ValueError:
            raise ValueError("Invalid chunk ids format. Can be a comma separated list or a range separated by '..'")
    return ids


def compute_chunk_ids(chunk_ids: list[int] | str, num_chunks: int) -> list[int] | None:
    """
    Compute chunk ids from the provided chunk ids string.

    Args:
        chunk_ids: Comma separated list of chunk ids or range separated by '..' or ','.
        num_chunks: Total number of chunks.

    Returns:
        List of chunk ids.
    """
    if num_chunks is None:
        return None

    # Parse chunk ids
    if chunk_ids is not None:
        if isinstance(chunk_ids, str):
            return str_ids_to_list(chunk_ids)
        return chunk_ids

    else:
        chunk_ids = list(range(0, num_chunks))

    # Assert that run ids are 1-indexed
    for chunk_id in chunk_ids:
        assert chunk_id < num_chunks, "Run ids should have 1-based indexing"
        assert chunk_id >= 0, "Run ids should have 1-based indexing"

    return chunk_ids


def prefill_judgement(data_point: dict) -> str | None:
    """Will automatically fill judgement if there is an exact match or the answer is None."""
    if data_point['predicted_answer'] is None or data_point['predicted_answer'] == '':
        return "Reasoning: No answer was provided.\nJudgement: No"

    if str(data_point['predicted_answer']).strip() == str(data_point['expected_answer']).strip():
        return "Reasoning: The two answers are identical.\nJudgement: Yes"

    return None

def check_no_extra_args_fire():
    """
    Check if there are any extra arguments passed to the function.

    This function inspects the command-line arguments and verifies that all
    arguments passed to the function are expected by the function's signature.
    If any extra arguments are found, it raises a ValueError.

    Raises:
        RuntimeError: If the function name is not found in the calling context.
        ValueError: If extra arguments are found that are not accepted by the function.
    """
    args = sys.argv[1:]
    # Extract the function name and its arguments from the command-line arguments
    function_name = args[0]
    function_args = args[1:]

    # Determine the calling context by inspecting the call stack
    context = {}
    caller = inspect.stack()[1]
    caller_frame = caller[0]
    caller_globals = caller_frame.f_globals
    caller_locals = caller_frame.f_locals
    context.update(caller_globals)
    context.update(caller_locals)

    # Check if the help flag is present
    if any(arg.startswith("--help") for arg in function_args) or '--help' in function_name:
        return None  # Skip the check if the help flag is present

    # Check if the function name exists in the calling context
    if function_name not in context:
        raise RuntimeError(
            f"Function {function_name} not found in the calling context when checking for unused arguments."
        )

    component = context[function_name]

    # Determine if the component is a class or a routine
    is_class = inspect.isclass(component)
    treatment = ('class' if is_class else 'routine',)

    # Get the metadata and parse function arguments
    metadata = fire_decorators.GetMetadata(component)
    fn = component.__call__ if treatment == 'callable' else component
    parse = fire.core._MakeParseFn(fn, metadata)
    (varargs, kwargs), consumed_args, remaining_args, capacity = parse(function_args)

    # Check for extra arguments that are not accepted by the function
    if remaining_args:
        raise ValueError(
            f"Extra arguments found that are not accepted by function `{function_name}`:\n"
            f"Additional arguments: {' '.join(remaining_args)}"
        )


def resolve_python_module_from_file(py_filepath: str, root_module: str = 'nemo_skills'):
    """
    Get the python module path from a python file path.
    Ex: /.../NeMo-Skills/nemo_skills/dataset/abc.py -> nemo_skills.dataset.abc

    Args:
        py_filepath: Python file path. Provided as __file__ from the calling module.

    Returns:
        str: Python module path.
    """
    module_path = Path(py_filepath).absolute()
    parts = Path(module_path).parts

    if root_module not in parts:
        raise ValueError("The provided file path is not within the nemo_codegen module.")

    stripped_module_path = Path(*parts[parts.index(root_module) :])
    striped_module = str(stripped_module_path).replace("/", ".")
    striped_module = os.path.splitext(striped_module)[0]
    return striped_module
