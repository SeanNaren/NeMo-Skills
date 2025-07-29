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

import logging
import os
from typing import List

import typer

from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.run_cmd import run_cmd as _run_cmd
from nemo_skills.pipeline.utils import get_cluster_config, get_env_variables
from nemo_skills.utils import get_logger_name, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def prepare_data(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs.",
    ),
    data_dir: str = typer.Option(
        None,
        help="Path to the data directory. If not specified, will use the default nemo_skills/dataset path. "
        "If a custom path is provided, you'd need to set it as NEMO_SKILLS_DATA_DIR environment variable "
        "or pass it as an argument to the `ns eval` command.",
    ),
    expname: str = typer.Option("prepare-data", help="Experiment name for data preparation"),
    partition: str = typer.Option(None, help="Slurm partition to use"),
    time_min: str = typer.Option(None, help="Time-min slurm parameter"),
    num_gpus: int | None = typer.Option(None, help="Number of GPUs to use"),
    num_nodes: int = typer.Option(1, help="Number of nodes to use"),
    mount_paths: str = typer.Option(None, help="Comma separated list of paths to mount"),
    run_after: List[str] = typer.Option(None, help="List of expnames that this job depends on before starting"),
    reuse_code: bool = typer.Option(True, help="Whether to reuse code from previous experiments"),
    reuse_code_exp: str = typer.Option(None, help="Experiment to reuse code from"),
    config_dir: str = typer.Option(None, help="Custom cluster config directory"),
    with_sandbox: bool = typer.Option(False, help="Start a sandbox container alongside"),
    log_dir: str = typer.Option(None, help="Custom location for slurm logs"),
    exclusive: bool = typer.Option(False, help="If set will add exclusive flag to the slurm job."),
    check_mounted_paths: bool = typer.Option(False, help="Check mounted paths availability"),
):
    """Prepare datasets by running python -m nemo_skills.dataset.prepare.

    Run `python -m nemo_skills.dataset.prepare --help` to see other supported arguments.
    """
    setup_logging(disable_hydra_logs=False, use_rich=True)
    extra_arguments = f'{" ".join(ctx.args)}'
    command = f"python -m nemo_skills.dataset.prepare {extra_arguments}"
    if data_dir:
        command += f" && mkdir -p {data_dir} && cp -r /nemo_run/code/nemo_skills/dataset/* {data_dir}"

    cluster_config = get_cluster_config(cluster, config_dir=config_dir)
    if cluster_config['executor'] == 'local' and not data_dir:
        # in this case we need to put the results in the current folder
        # if we use container, it will mess up permissions, so as a workaround
        # setting executore to none
        cluster_config['executor'] = 'none'

    if cluster_config['executor'] == 'slurm' and not data_dir:
        raise ValueError(
            "Data directory is required to be specified when using slurm executor. "
            "Please provide --data_dir argument."
        )

    log_dir = log_dir or data_dir

    # we already captured extra arguments
    ctx.args = []

    if data_dir:
        env_vars = get_env_variables(cluster_config)
        data_dir_env_var = env_vars.get("NEMO_SKILLS_DATA_DIR") or os.environ.get("NEMO_SKILLS_DATA_DIR")
        if data_dir_env_var != data_dir:
            LOG.warning(
                f"NEMO_SKILLS_DATA_DIR environment variable is set to {data_dir_env_var}, "
                f"but you provided --data_dir={data_dir}. "
                f"Make sure to set NEMO_SKILLS_DATA_DIR={data_dir} in your environment "
                f"or pass it as an argument to `ns eval`."
            )
        if data_dir_env_var is None:
            LOG.warning(
                f"NEMO_SKILLS_DATA_DIR environment variable is not set. "
                f"You might want to set it as NEMO_SKILLS_DATA_DIR={data_dir} "
                f"to avoid always specifying it as a parameter to `ns eval`."
            )
        # TODO: automatically add it to cluster config based on user prompt?

    return _run_cmd(
        ctx=ctx,
        cluster=cluster_config,
        command=command,
        expname=expname,
        partition=partition,
        time_min=time_min,
        num_gpus=num_gpus,
        num_nodes=num_nodes,
        mount_paths=mount_paths,
        run_after=run_after,
        reuse_code=reuse_code,
        reuse_code_exp=reuse_code_exp,
        config_dir=config_dir,
        with_sandbox=with_sandbox,
        log_dir=log_dir,
        exclusive=exclusive,
        check_mounted_paths=check_mounted_paths,
    )


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()
