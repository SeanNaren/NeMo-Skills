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

import importlib
import logging
import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import nemo_skills.pipeline.utils as pipeline_utils
from nemo_skills.dataset.utils import get_dataset_module
from nemo_skills.inference.generate import GenerationTask
from nemo_skills.utils import compute_chunk_ids, get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


@dataclass
class BenchmarkArgs:
    name: str
    input_file: str
    generation_args: str
    eval_args: str
    judge_args: str
    judge_pipeline_args: dict
    requires_sandbox: bool
    generation_module: str
    num_samples: int
    num_chunks: int | None
    eval_subfolder: str
    benchmark_group: str | None = None
    score_module: str | None = None
    job_ids: list[int] = field(default_factory=list)
    remaining_jobs: list[dict] = field(default_factory=list)

    @property
    def requires_judge(self):
        return bool(self.judge_args or self.judge_pipeline_args)


def get_arg_from_module_or_dict(module, arg_name, default_value=None, override_dict=None):
    """If argument is in a dict, take from there. If not, take from the module."""
    if override_dict and arg_name in override_dict:
        return override_dict[arg_name]
    if hasattr(module, arg_name):
        return getattr(module, arg_name)
    if default_value is not None:
        return default_value
    raise ValueError(f"Argument {arg_name} not found in module {module} or override_dict.")


def get_benchmark_args_from_module(
    benchmark_module,
    benchmark,
    split,
    cluster_config,
    data_path,
    is_on_cluster,
    benchmark_group=None,
    override_dict=None,
):
    if split is None:
        split = get_arg_from_module_or_dict(benchmark_module, "EVAL_SPLIT", "test", override_dict)

    if not is_on_cluster:
        if pipeline_utils.is_mounted_filepath(cluster_config, data_path):
            input_file = f"{data_path}/{benchmark.replace('.', '/')}/{split}.jsonl"
            unmounted_input_file = pipeline_utils.get_unmounted_path(cluster_config, input_file)
            unmounted_path = str(Path(__file__).parents[3] / unmounted_input_file.replace('/nemo_run/code/', ''))
        else:
            # will be copied over in this case as it must come from extra datasets
            input_file = f"/nemo_run/code/{Path(data_path).name}/{benchmark.replace('.', '/')}/{split}.jsonl"
            unmounted_path = Path(data_path) / benchmark.replace('.', '/') / f"{split}.jsonl"
    else:
        # on cluster we will always use the mounted path
        input_file = f"{data_path}/{benchmark.replace('.', '/')}/{split}.jsonl"
        unmounted_path = pipeline_utils.get_unmounted_path(cluster_config, input_file)

    unmounted_path = str(unmounted_path)
    # checking if data file exists (can check locally as well)
    if is_on_cluster:
        if not pipeline_utils.cluster_path_exists(cluster_config, unmounted_path):
            raise ValueError(
                f"Data file {unmounted_path} does not exist on cluster. "
                "Please check the benchmark and split parameters. "
                "Did you forget to run prepare data commands?"
            )
    else:
        if not Path(unmounted_path).exists():
            raise ValueError(
                f"Data file {unmounted_path} does not exist locally. "
                "Please check the benchmark and split parameters. "
                "Did you forget to run prepare data commands?"
            )

    prompt_config = get_arg_from_module_or_dict(benchmark_module, "PROMPT_CONFIG", override_dict=override_dict)
    generation_args = get_arg_from_module_or_dict(benchmark_module, "GENERATION_ARGS", "", override_dict=override_dict)
    generation_args = f"++prompt_config={prompt_config} {generation_args}"
    requires_sandbox = get_arg_from_module_or_dict(benchmark_module, "REQUIRES_SANDBOX", False, override_dict)

    generation_module = get_arg_from_module_or_dict(
        benchmark_module, "GENERATION_MODULE", "nemo_skills.inference.generate", override_dict
    )
    # make a copy to avoid modifying the original
    judge_pipeline_args = deepcopy(
        get_arg_from_module_or_dict(benchmark_module, "JUDGE_PIPELINE_ARGS", {}, override_dict)
    )
    judge_args = get_arg_from_module_or_dict(benchmark_module, "JUDGE_ARGS", "", override_dict)
    eval_args = get_arg_from_module_or_dict(benchmark_module, "EVAL_ARGS", override_dict=override_dict)
    num_samples = get_arg_from_module_or_dict(benchmark_module, "NUM_SAMPLES", 0, override_dict)
    num_chunks = get_arg_from_module_or_dict(benchmark_module, "NUM_CHUNKS", 0, override_dict)
    if num_chunks == 0:
        num_chunks = None

    if judge_args or judge_pipeline_args:
        # setting to a tmp folder for judge and then the judged outputs will be in main eval-results folder
        eval_subfolder = "tmp-eval-results/"
    else:
        eval_subfolder = "eval-results/"

    if benchmark_group:
        eval_subfolder += f"{benchmark_group}/"
    eval_subfolder += benchmark

    return BenchmarkArgs(
        name=benchmark,
        input_file=input_file,
        generation_args=generation_args,
        eval_args=eval_args,
        judge_args=judge_args,
        judge_pipeline_args=judge_pipeline_args,
        requires_sandbox=requires_sandbox,
        generation_module=generation_module,
        num_samples=num_samples,
        num_chunks=num_chunks,
        eval_subfolder=eval_subfolder,
        benchmark_group=benchmark_group,
    )


def add_default_args(cluster_config, benchmark_or_group, split, data_dir, extra_datasets_type, extra_datasets):
    benchmark_or_group_module, data_path, is_on_cluster = get_dataset_module(
        dataset=benchmark_or_group,
        data_dir=data_dir,
        cluster_config=cluster_config,
        extra_datasets=extra_datasets,
        extra_datasets_type=extra_datasets_type,
    )

    if getattr(benchmark_or_group_module, "IS_BENCHMARK_GROUP", False):
        benchmarks_args = []
        for benchmark, override_dict in benchmark_or_group_module.BENCHMARKS.items():
            benchmark_module, data_path, is_on_cluster = get_dataset_module(
                dataset=benchmark,
                data_dir=data_dir,
                cluster_config=cluster_config,
                extra_datasets=extra_datasets,
                extra_datasets_type=extra_datasets_type,
            )
            benchmark_args = get_benchmark_args_from_module(
                benchmark_module=benchmark_module,
                benchmark=benchmark,
                benchmark_group=benchmark_or_group,
                split=split,
                cluster_config=cluster_config,
                data_path=data_path,
                is_on_cluster=is_on_cluster,
                override_dict=override_dict,
            )
            # TODO: should it be optional?
            benchmark_args.score_module = benchmark_or_group_module.SCORE_MODULE
            benchmarks_args.append(benchmark_args)
        return benchmarks_args

    # Single benchmark
    benchmark = benchmark_or_group
    return [
        get_benchmark_args_from_module(
            benchmark_module=benchmark_or_group_module,
            benchmark=benchmark,
            split=split,
            cluster_config=cluster_config,
            data_path=data_path,
            is_on_cluster=is_on_cluster,
        )
    ]


def prepare_eval_commands(
    cluster_config,
    benchmarks_or_groups,
    split,
    extra_datasets,
    num_jobs,
    starting_seed,
    output_dir,
    num_chunks,
    chunk_ids,
    rerun_done,
    server_parameters,
    extra_arguments,
    data_dir,
    extra_datasets_type,
    exclusive,
    with_sandbox,
    wandb_parameters,
    extra_eval_args,
):
    # TODO: there is a bit too much code duplication here and logic is quite dense, should try to refactor

    # TODO: should we allow setting num chunks per benchmark when not using groups? Maybe benchmark:rs_num:num_chunks?

    benchmarks_or_groups = {
        k: int(v) for k, v in [b.split(":") if ":" in b else (b, -1) for b in benchmarks_or_groups.split(",")]
    }

    extra_datasets = extra_datasets or os.environ.get("NEMO_SKILLS_EXTRA_DATASETS")

    if num_jobs is None:
        if cluster_config['executor'] == 'slurm':
            num_jobs = -1  # -1 means run all benchmarks in parallel
        else:
            # for local executor, it makes no sense to use other values
            num_jobs = 1

    benchmarks_dict = {}  # benchmark_name -> benchmark_args
    for benchmark_or_group, rs_num in benchmarks_or_groups.items():
        cur_benchmarks = add_default_args(
            cluster_config,
            benchmark_or_group,
            split,
            data_dir,
            extra_datasets_type,
            extra_datasets,
        )
        for benchmark_args in cur_benchmarks:
            benchmark = benchmark_args.name
            if benchmark in benchmarks_dict:
                raise ValueError(
                    f"Benchmark {benchmark} is specified multiple times in the benchmarks list. "
                    "Please ensure each benchmark is unique."
                )
            benchmarks_dict[benchmark] = benchmark_args
            if rs_num != -1:
                if len(cur_benchmarks) > 1:
                    raise ValueError(
                        f"Cannot specify number of samples ({rs_num}) for benchmark group {benchmark_or_group}. "
                        f"Use '{benchmark_or_group}' instead of '{benchmark_or_group}:{rs_num}'."
                    )
                benchmarks_dict[benchmark].num_samples = rs_num

            if benchmark_args.requires_sandbox and not with_sandbox:
                LOG.warning("Found benchmark (%s) which requires sandbox, enabled sandbox for it.", benchmark)

    total_evals = 0
    for benchmark, benchmark_args in benchmarks_dict.items():
        if benchmark_args.num_samples == 0:
            random_seeds = [None]
        else:
            random_seeds = list(range(starting_seed, starting_seed + benchmark_args.num_samples))

        benchmark_chunk_ids = None
        if num_chunks:
            benchmark_args.num_chunks = num_chunks
        if benchmark_args.num_chunks is not None:
            # TODO: currently using global chunk_ids but local num_chunks. That's not ideal
            benchmark_chunk_ids = compute_chunk_ids(chunk_ids, benchmark_args.num_chunks)
        if benchmark_chunk_ids is None:
            benchmark_chunk_ids = [None]

        benchmark_args.remaining_jobs = pipeline_utils.get_remaining_jobs(
            cluster_config=cluster_config,
            output_dir=f"{output_dir}/{benchmark_args.eval_subfolder}",
            random_seeds=random_seeds,
            chunk_ids=benchmark_chunk_ids,
            rerun_done=rerun_done,
        )
        for seed_idx, (seed, benchmark_chunk_ids) in enumerate(benchmark_args.remaining_jobs.items()):
            total_evals += len(benchmark_chunk_ids)

    if num_jobs < 0:
        # if num_jobs is -1, we run all benchmarks in parallel
        num_jobs = total_evals

    if num_jobs == 0:
        return benchmarks_dict, []

    evals_per_job = total_evals // num_jobs if num_jobs > 0 else total_evals
    remainder = total_evals % num_jobs
    eval_to_job_map = []
    for i in range(num_jobs):
        count = evals_per_job + (1 if i < remainder else 0)
        eval_to_job_map.extend([i] * count)

    cur_job_idx = 0
    get_random_port = pipeline_utils.should_get_random_port(
        server_parameters['server_gpus'], exclusive, server_parameters['server_type']
    )
    job_server_config, job_server_address, job_extra_arguments = pipeline_utils.configure_client(
        **server_parameters,
        extra_arguments=extra_arguments,
        get_random_port=get_random_port,
    )

    cur_eval = 0
    job_batches = []
    job_cmds = []
    job_benchmarks = set()

    for benchmark, benchmark_args in benchmarks_dict.items():
        if benchmark_args.num_samples == 0:
            random_seeds = [None]
        else:
            random_seeds = list(range(starting_seed, starting_seed + benchmark_args.num_samples))

        benchmark_output_dir = f"{output_dir}/{benchmark_args.eval_subfolder}"
        for seed_idx, (seed, benchmark_chunk_ids) in enumerate(benchmark_args.remaining_jobs.items()):
            if wandb_parameters:
                # no need for chunks as it will run after merging
                wandb_parameters['samples_file'] = pipeline_utils.get_chunked_rs_filename(
                    benchmark_output_dir,
                    random_seed=seed,
                    chunk_id=None,
                )
            for chunk_id in benchmark_chunk_ids:
                job_benchmarks.add(benchmark)

                generation_task = importlib.import_module(benchmark_args.generation_module)
                if not hasattr(generation_task, 'GENERATION_TASK_CLASS'):
                    raise ValueError(
                        f"Module {benchmark_args.generation_module} does not have a GENERATION_TASK_CLASS attribute. "
                        "Please provide a valid generation module."
                    )
                generation_task = generation_task.GENERATION_TASK_CLASS
                if (
                    generation_task.get_server_command_fn.__func__ != GenerationTask.get_server_command_fn.__func__
                    and num_jobs != total_evals
                ):
                    raise ValueError(
                        f"Class {generation_task} overrides get_server_command_fn, "
                        "which is not supported for evaluation when grouping jobs."
                    )
                full_extra_arguments = (
                    f"{generation_task.get_generation_default_args()} "
                    f"{benchmark_args.generation_args} "
                    f"{job_extra_arguments} "
                )
                cmd = pipeline_utils.get_generation_cmd(
                    input_file=benchmark_args.input_file,
                    output_dir=benchmark_output_dir,
                    extra_arguments=full_extra_arguments,
                    random_seed=seed,
                    eval_args=f"{benchmark_args.eval_args} {extra_eval_args}",
                    chunk_id=chunk_id,
                    num_chunks=benchmark_args.num_chunks,
                    script=benchmark_args.generation_module,
                    # only logging for the first seed
                    wandb_parameters=wandb_parameters if seed_idx == 0 else None,
                )
                job_cmds.append(cmd)

                if cur_eval == total_evals - 1 or cur_job_idx != eval_to_job_map[cur_eval + 1]:
                    job_needs_sandbox = any(benchmarks_dict[b].requires_sandbox for b in job_benchmarks)
                    # TODO: move to a dataclass
                    job_batches.append(
                        (
                            job_cmds,
                            job_benchmarks,
                            job_needs_sandbox,
                            job_server_config,
                            job_server_address,
                            # a check above guarantees that this is the same for all tasks in a job
                            generation_task.get_server_command_fn(),
                        )
                    )
                    job_server_config, job_server_address, job_extra_arguments = pipeline_utils.configure_client(
                        **server_parameters,
                        extra_arguments=extra_arguments,
                        get_random_port=get_random_port,
                    )
                    for job_benchmark in job_benchmarks:
                        benchmarks_dict[job_benchmark].job_ids.append(cur_job_idx)
                    cur_job_idx += 1
                    job_cmds = []
                    job_benchmarks = set()

                cur_eval += 1

    return benchmarks_dict, job_batches
