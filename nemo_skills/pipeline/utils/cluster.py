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
import sys
import tarfile
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

import nemo_run as run
import yaml
from huggingface_hub import get_token
from invoke import StreamWatcher
from nemo_run.config import set_nemorun_home
from omegaconf import DictConfig

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))

# Add a module-level set to track which environment variables have been logged
_logged_required_env_vars = set()
_logged_optional_env_vars = set()


def get_timeout(cluster_config, partition):
    if 'timeouts' not in cluster_config:
        timeout = "10000:00:00:00"
    else:
        timeout = cluster_config["timeouts"][partition or cluster_config["partition"]]

        # subtracting 15 minutes to account for the time it takes to save the model
        # the format expected by nemo is days:hours:minutes:seconds
        time_diff = datetime.strptime(timeout, "%H:%M:%S") - datetime.strptime("00:15:00", "%H:%M:%S")
        timeout = (
            f'00:{time_diff.seconds // 3600:02d}:{(time_diff.seconds % 3600) // 60:02d}:{time_diff.seconds % 60:02d}'
        )
    return timeout


def get_env_variables(cluster_config):
    """
    Will get the environment variables from the cluster config and the user environment.

    The following items in the cluster config are supported:
    - `required_env_vars` - list of required environment variables
    - `env_vars` - list of optional environment variables

    WANDB_API_KEY, NVIDIA_API_KEY, AZURE_OPENAI_API_KEY, OPENAI_API_KEY, and HF_TOKEN are always added if they exist.

    Args:
        cluster_config: cluster config dictionary

    Returns:
        dict: dictionary of environment
    """
    global _logged_required_env_vars, _logged_optional_env_vars

    env_vars = {}
    # Check for user requested env variables
    required_env_vars = cluster_config.get("required_env_vars", [])
    for env_var in required_env_vars:
        env_var_name = env_var.split('=')[0].strip() if "=" in env_var else env_var

        if "=" in env_var:
            if env_var.count("=") == 1:
                env_var_name, value = env_var.split("=")
                env_var_name = env_var_name.strip()
                value = value.strip()
            else:
                raise ValueError(f"Invalid required environment variable format: {env_var}")
            env_vars[env_var_name] = value
            if env_var_name not in _logged_required_env_vars:
                LOG.info(f"Adding required environment variable {env_var_name} from config")
                _logged_required_env_vars.add(env_var_name)
        elif env_var in os.environ:
            env_vars[env_var] = os.environ[env_var]
            if env_var not in _logged_required_env_vars:
                LOG.info(f"Adding required environment variable {env_var} from environment")
                _logged_required_env_vars.add(env_var)
        else:
            raise ValueError(f"Required environment variable {env_var} not found.")

    # It is fine to have these as always optional even if they are required for some configs
    # Assume it is required, then this will override the value set above with the same
    # value, assuming it has not been updated externally between these two calls
    always_optional_env_vars = [
        "WANDB_API_KEY",
        "NVIDIA_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "OPENAI_API_KEY",
        "HF_TOKEN",
    ]
    default_factories = {
        "HF_TOKEN": lambda: str(get_token()),
    }
    # Add optional env variables
    optional_env_vars = cluster_config.get("env_vars", [])
    for env_var in optional_env_vars + always_optional_env_vars:
        env_var_name = env_var.split('=')[0].strip() if "=" in env_var else env_var

        if "=" in env_var:
            if env_var.count("=") == 1:
                env_var_name, value = env_var.split("=")
                env_var_name = env_var_name.strip()
                value = value.strip()
            else:
                raise ValueError(f"Invalid optional environment variable format: {env_var}")
            env_vars[env_var_name] = value
            if env_var_name not in _logged_optional_env_vars:
                LOG.info(f"Adding optional environment variable {env_var_name} from config")
                _logged_optional_env_vars.add(env_var_name)
        elif env_var in os.environ:
            env_vars[env_var] = os.environ[env_var]
            if env_var not in _logged_optional_env_vars:
                LOG.info(f"Adding optional environment variable {env_var} from environment")
                _logged_optional_env_vars.add(env_var)
        elif env_var in default_factories:
            env_vars[env_var] = default_factories[env_var]()
            if env_var not in _logged_optional_env_vars:
                LOG.info(f"Adding optional environment variable {env_var} from environment")
                _logged_optional_env_vars.add(env_var)
        else:
            if env_var not in _logged_optional_env_vars:
                LOG.info(f"Optional environment variable {env_var} not found in user environment; skipping.")
                _logged_optional_env_vars.add(env_var)

    return env_vars


@contextmanager
def temporary_env_update(cluster_config, updates):
    original_env_vars = cluster_config.get("env_vars", []).copy()
    updated_env_vars = original_env_vars.copy()
    for key, value in updates.items():
        updated_env_vars.append(f"{key}={value}")
        cluster_config["env_vars"] = updated_env_vars
    try:
        yield
    finally:
        cluster_config["env_vars"] = original_env_vars


def read_config(config_file):
    with open(config_file, "rt", encoding="utf-8") as fin:
        cluster_config = yaml.safe_load(fin)

    # resolve ssh tunnel config
    if "ssh_tunnel" in cluster_config:
        cluster_config = update_ssh_tunnel_config(cluster_config)

    if cluster_config['executor'] == 'slurm' and "ssh_tunnel" not in cluster_config:
        if "job_dir" not in cluster_config:
            raise ValueError("job_dir must be provided in the cluster config if ssh_tunnel is not provided.")
        set_nemorun_home(cluster_config["job_dir"])

    if 'trtllm' in cluster_config['containers']:
        # automatically setting same container for trtllm-serve
        if 'trtllm-serve' not in cluster_config['containers']:
            LOG.info("Setting trtllm-serve container to be the same as trtllm.")
            cluster_config['containers']['trtllm-serve'] = cluster_config['containers']['trtllm']

    return cluster_config


def get_cluster_config(cluster=None, config_dir=None):
    """Trying to find an appropriate cluster config.

    Will search in the following order:
    1. config_dir parameter
    2. NEMO_SKILLS_CONFIG_DIR environment variable
    3. Current folder / cluster_configs
    4. This file folder / ../../cluster_configs

    If NEMO_SKILLS_CONFIG is provided and cluster is None,
    it will be used as a full path to the config file
    and NEMO_SKILLS_CONFIG_DIR will be ignored.

    If cluster is a python object (dict-like), then we simply
    return the cluster config, under the assumption that the
    config is prepared by the user.
    """
    # if cluster is provided, we try to find it in one of the folders
    if cluster is not None:
        # check if cluster is a python object instead of a str path, pass through
        if isinstance(cluster, (dict, DictConfig)):
            return cluster

        # either using the provided config_dir or getting from env var
        config_dir = config_dir or os.environ.get("NEMO_SKILLS_CONFIG_DIR")
        if config_dir:
            return read_config(Path(config_dir) / f"{cluster}.yaml")

        # if it's not defined we are trying to find locally
        if (Path.cwd() / 'cluster_configs' / f"{cluster}.yaml").exists():
            return read_config(Path.cwd() / 'cluster_configs' / f"{cluster}.yaml")

        if (Path(__file__).parents[3] / 'cluster_configs' / f"{cluster}.yaml").exists():
            return read_config(Path(__file__).parents[3] / 'cluster_configs' / f"{cluster}.yaml")

        raise ValueError(f"Cluster config {cluster} not found in any of the supported folders.")

    config_file = os.environ.get("NEMO_SKILLS_CONFIG")
    if not config_file:
        LOG.warning(
            "Cluster config is not specified. Running locally without containers. "
            "Only a subset of features is supported and you're responsible "
            "for installing any required dependencies. "
            "It's recommended to run `ns setup` to define appropriate configs!"
        )
        # just returning empty string for any container on access
        cluster_config = {'executor': 'none', 'containers': defaultdict(str)}
        return cluster_config

    if not Path(config_file).exists():
        raise ValueError(f"Cluster config {config_file} not found.")

    cluster_config = read_config(config_file)

    return cluster_config


def update_ssh_tunnel_config(cluster_config: dict):
    """
    Update the ssh tunnel configuration in the cluster config to resolve job dir and username.
    uses the `user` information to populate `job_dir` if in config.

    Args:
        cluster_config: dict: The cluster configuration dictionary

    Returns:
        dict: The updated cluster configuration dictionary
    """
    if 'ssh_tunnel' not in cluster_config:
        return cluster_config

    resolve_map = [
        dict(key='user', default_env_key='USER'),
        dict(key='job_dir', default_env_key=None),
        dict(key='identity', default_env_key=None),
    ]

    for item in resolve_map:
        key = item['key']
        default_env_key = item['default_env_key']

        if key in cluster_config['ssh_tunnel']:
            # Resolve `user` from env if not provided
            if cluster_config['ssh_tunnel'][key] is None and default_env_key is not None:
                cluster_config['ssh_tunnel'][key] = os.environ[default_env_key]
                LOG.info(f"Resolved `{key}` to `{cluster_config['ssh_tunnel'][key]}`")

            elif isinstance(cluster_config['ssh_tunnel'][key], str) and '$' in cluster_config['ssh_tunnel'][key]:
                cluster_config['ssh_tunnel'][key] = os.path.expandvars(cluster_config['ssh_tunnel'][key])
                LOG.info(f"Resolved `{key}` to `{cluster_config['ssh_tunnel'][key]}`")

    if "$" in cluster_config['ssh_tunnel']['identity']:
        raise ValueError(
            "SSH identity cannot be resolved from environment variables. "
            "Please provide a valid path to the identity file."
        )

    return cluster_config


@lru_cache
def _get_tunnel_cached(
    job_dir: str,
    host: str,
    user: str,
    identity: str | None = None,
    shell: str | None = None,
    pre_command: str | None = None,
):
    return run.SSHTunnel(
        host=host,
        user=user,
        identity=identity,
        shell=shell,
        pre_command=pre_command,
        job_dir=job_dir,
    )


def tunnel_hash(tunnel):
    if isinstance(tunnel, run.LocalTunnel):
        return "local"
    return f"{tunnel.job_dir}:{tunnel.host}:{tunnel.user}:{tunnel.identity}:{tunnel.shell}:{tunnel.pre_command}"


def get_tunnel(cluster_config):
    if "ssh_tunnel" not in cluster_config:
        if cluster_config["executor"] == "slurm":
            LOG.info("No ssh_tunnel configuration found, assuming we are running from the cluster already.")
        return run.LocalTunnel(job_dir="")
    return _get_tunnel_cached(**cluster_config["ssh_tunnel"])


# Helper class and function to support streaming updates
class OutputWatcher(StreamWatcher):
    """Class for streaming remote tar/compression process."""

    def submit(self, stream):
        print(stream, end='\r')
        sys.stdout.flush()
        return []


def progress_callback(transferred: int, total: int) -> None:
    """Display SFTP transfer progress."""
    percent = (transferred / total) * 100
    bar = '=' * int(percent / 2) + '>'
    sys.stdout.write(
        f'\rFile Transfer Progress: [{bar:<50}] {percent:.1f}% '
        f'({transferred/1024/1024:.1f}MB/{total/1024/1024:.1f}MB)'
    )
    sys.stdout.flush()


def cluster_download_file(cluster_config: dict, remote_file: str, local_file: str):
    tunnel = get_tunnel(cluster_config)
    tunnel.get(remote_file, local_file)


def cluster_path_exists(cluster_config: dict, remote_path: str):
    tunnel = get_tunnel(cluster_config)
    result = tunnel.run(f'test -e {remote_path} && echo "Exists"', hide=True, warn=True)
    return "Exists" in result.stdout


def cluster_download_dir(
    cluster_config: dict, remote_dir: str, local_dir: str, remote_tar_dir: Optional[str] = None, verbose: bool = True
):
    """
    Downloads a directory from a remote cluster by creating a tar archive and transferring it.

    Args:
        cluster_config: dictionary with cluster configuration
        remote_dir: Path to the directory on remote server
        local_dir: Local path to save the downloaded directory
        remote_tar_dir: Optional directory for temporary tar file creation
        verbose: Print download progress
    """
    tunnel = get_tunnel(cluster_config)
    remote_dir = remote_dir.rstrip('/')
    remote_dir_parent, remote_dir_name = os.path.split(remote_dir)

    # Directory where the remote tarball is written
    remote_tar_dir = remote_tar_dir if remote_tar_dir else remote_dir_parent
    # Path of the remote tar file
    remote_tar_filename = f"{remote_dir_name}.tar.gz"

    # Remote and local tar files
    remote_tar = f"{os.path.join(remote_tar_dir, remote_tar_filename)}"
    local_tar = os.path.join(local_dir, remote_tar_filename)

    # Get the directory size
    result = tunnel.run(f'du -sb {remote_dir} | cut -f1')
    total_size = int(result.stdout.strip())

    # Check if result directory compression is streamable
    streaming_possible = False
    try:
        # Check whether the command pv is present on the remote system or not.
        # Certain systems may not have the `pv` command
        result = tunnel.run('which pv', warn=True)
        streaming_possible = result.exited == 0
    except Exception:
        streaming_possible = False

    if streaming_possible and verbose:
        # We can do streaming compression
        # Command for streaming the compression progress
        command = (
            f'cd {remote_dir_parent} && '
            f'tar --exclude="*.log" -cf - {remote_dir_name} | '
            f'pv -s {total_size} -p -t -e -b -F "Compressing Remote Directory: %b %t %p" | '
            f'gzip > {remote_tar}'
        )
        # Run the remote compression command and stream the progress
        result = tunnel.run(command, watchers=[OutputWatcher()], pty=True, hide=(not verbose))
    else:
        command = f'cd {remote_dir_parent} && tar -czf {remote_tar} {remote_dir_name}'
        result = tunnel.run(command, hide=(not verbose))

    # Get SFTP client from tunnel's session's underlying client
    sftp = tunnel.session.client.open_sftp()

    # Use SFTP's get with callback
    sftp.get(remote_tar, local_tar, callback=progress_callback if verbose else None)
    print(f"\nTransfer complete: {local_tar}")

    # Extract the tarball locally
    os.makedirs(local_dir, exist_ok=True)
    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(path=local_dir)

    # Clean up the tarball from the remote server
    tunnel.run(f'rm {remote_tar}', hide=True)

    # Clean up the local tarball
    os.remove(local_tar)


def cluster_upload(cluster_config: dict, local_file: str, remote_dir: str, verbose: bool = True):
    """
    Uploads a file to cluster.
    TODO: extend to a folder.

    Args:
        cluster_config: dictionary with cluster configuration
        local_file: Path to the local file to upload
        remote_dir: Cluster path where to save the file
        verbose: Print upload progress
    """
    tunnel = get_tunnel(cluster_config)
    sftp = tunnel.session.client.open_sftp()
    sftp.put(str(local_file), str(remote_dir), callback=progress_callback if verbose else None)
    print(f"\nTransfer complete")
