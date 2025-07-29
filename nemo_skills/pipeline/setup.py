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

import os
import subprocess
from pathlib import Path

import typer
import yaml

from nemo_skills import _containers
from nemo_skills.pipeline.app import app


def is_docker_available():
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        return True
    except subprocess.SubprocessError:
        return False


# Helper function to pull Docker containers
def pull_docker_containers(containers):
    for container_name, container_image in containers.items():
        typer.echo(f"Pulling {container_name}: {container_image}...")
        try:
            subprocess.run(["docker", "pull", container_image], check=True)
            typer.echo(f"Successfully pulled {container_image}")
        except subprocess.SubprocessError as e:
            typer.echo(f"Failed to pull {container_image}: {e}")


@app.command()
def setup():
    """Helper command to setup cluster configs."""
    typer.echo(
        "Let's set up your cluster configs! It's called 'cluster', but you need one to run things locally as well.\n"
        "The configs are just yaml files that you can later inspect and modify.\n"
        "They are mostly there to let us know which containers/mounts to use and how to orchestrate jobs on slurm."
    )

    # Get the directory for cluster configs with default as current dir / cluster_configs
    default_dir = os.path.join(os.getcwd(), "cluster_configs")
    config_dir = typer.prompt("\nWhere would you like to store your cluster configs?", default=default_dir)

    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)

    while True:
        # Ask the user if they want a local or slurm config
        config_type = typer.prompt("\nWhat type of config would you like to create? (local/slurm)").lower()

        # Ask for the name of the config file
        default_name = f"{config_type}"
        config_name = typer.prompt(
            f"\nWhat would you like to name your {config_type} config file? "
            "You'd need to use that name as --cluster argument to ns commands.",
            default=default_name,
        )

        # Check if the config file already exists
        config_file = config_dir / (config_name + ".yaml")
        if config_file.exists():
            overwrite = typer.confirm(
                f"\nThe config file {config_file} already exists. Do you want to overwrite it?",
                default=False,
            )
            if not overwrite:
                continue

        # initializing default containers
        config = {'executor': config_type, 'containers': _containers}

        mounts = typer.prompt(
            "\nWe execute all commands in docker containers, so you need to "
            f"define what to mount there to access your {config_type} data/models.\n"
            "You don't need to mount nemo-skills or your local git repo, it's always accessible with /nemo_run/code\n"
            "It's usually a good idea to define some mounts for your general workspace (to keep data/output results)\n"
            "as well as for your models (e.g. /trt_models, /hf_models).\n"
            "If you're setting up a Slurm config, make sure to use the cluster paths here.\n"
            "What mounts would you like to add? (comma separated)",
            default=f"/home/{os.getlogin()}:/workspace" if config_type == 'local' else None,
        )

        env_vars = typer.prompt(
            "\nYou can also specify any environment variables that you want to be accessible in containers.\n"
            "Can either define just the name (we take value from the current environment), "
            "or name=value to use a fixed value.\n"
            "By default we will always pass "
            "WANDB_API_KEY, NVIDIA_API_KEY, AZURE_OPENAI_API_KEY, OPENAI_API_KEY, HF_TOKEN, "
            "so you don't need to list those.\n"
            "What other environment variables would you like to define? (comma separated)",
            default="",
        )

        config['mounts'] = [m.strip() for m in mounts.split(",")]
        config['env_vars'] = [e.strip() for e in env_vars.split(",") if e.strip()]
        if not config['env_vars']:
            config.pop('env_vars')

        if config_type == 'slurm':
            ssh_access = typer.confirm(
                "\nIt's recommended to run ns commands from a local workstation (not on the cluster) "
                "and let us take care of uploading your code / scheduling jobs through ssh.\n"
                "But we also support running jobs on cluster directly (you'd need to install nemo-skills there).\n"
                "Are you planning to run ns commands from your local workstation (not on cluster)?",
                default=True,
            )
            if ssh_access:
                ssh_tunnel = {}
                ssh_tunnel['host'] = typer.prompt("\nWhat is the ssh hostname of your cluster?")
                ssh_tunnel['user'] = typer.prompt(
                    "\nWhat is your ssh username on the cluster?",
                    default=os.getlogin(),
                )
                default_key = os.path.expanduser("~/.ssh/id_rsa")
                ssh_tunnel['identity'] = typer.prompt(
                    "\nWhat is the path to your ssh private key? If you don't use key, leave empty.",
                    default=default_key if os.path.exists(default_key) else "",
                )
                if ssh_tunnel['identity'] == "":
                    ssh_tunnel.pop('identity')
                ssh_tunnel['job_dir'] = typer.prompt(
                    "\nWe need some place to keep uploaded code and experiment metadata on cluster.\n"
                    "What path do you want to use for that (you might need to clean it "
                    "up periodically if you submit many experiments)?",
                )
                config['ssh_tunnel'] = ssh_tunnel
            else:
                config['job_dir'] = typer.prompt(
                    "\nWe need some place to experiment metadata and packaged code.\n"
                    "What path do you want to use for that (you might need to clean it "
                    "up periodically if you submit many experiments)?",
                )
            config['account'] = typer.prompt("\nWhat is the slurm account you want to use?")
            config['partition'] = typer.prompt(
                "\nWhat is the default slurm partition you want to use? "
                "You can always override with --partition argument.",
            )
            config['job_name_prefix'] = ""
            timeouts = typer.prompt(
                "\nIf your cluster has a strict time limit for each job, we need to "
                "know the value to be able to save checkpoints before the job is killed.\n"
                "Specify as partition1=hh:mm:ss;partition2=hh:mm:ss\n"
                "Leave empty if you don't have time limits.",
                default="",
            )
            if timeouts:
                config["timeouts"] = {
                    partition.split("=")[0]: partition.split("=")[1] for partition in timeouts.split(";")
                }

        # Create the config file
        with open(config_file, 'wt') as fout:
            yaml.dump(config, fout, sort_keys=False, indent=4)

        typer.echo(
            f"\nCreated {config_type} config file at {config_file}.\n"
            f"The containers section was initialized with default values, but you can always change them manually.\n"
            f"You can find more information on what containers we use in "
            f"https://github.com/NVIDIA/NeMo-Skills/tree/main/dockerfiles"
        )

        if config_type == 'local':
            pull_containers = typer.confirm(
                "\nWould you like to pull all the necessary Docker containers now? "
                "This might take some time but ensures everything is ready to use.\n"
                "You can skip this step and we will pull the containers automatically when you run the first job.",
                default=True,
            )

            if pull_containers:
                if is_docker_available():
                    typer.echo("\nPulling Docker containers...")
                    pull_docker_containers(config['containers'])
                    typer.echo("All containers have been pulled!")
                else:
                    typer.echo(
                        "\nDocker does not seem to be available on your system. Please ensure Docker is installed."
                    )

        # Ask if the user wants to create another config
        create_another = typer.confirm("\nWould you like to create another cluster config?", default=False)
        if not create_another:
            break

    typer.echo(
        f"\nGreat, you're all done! It might be a good idea to define "
        f"NEMO_SKILLS_CONFIGS={config_dir}, so that configs are always found."
    )


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()
