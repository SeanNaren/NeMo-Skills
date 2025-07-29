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
import os
import subprocess
from pathlib import Path

import pytest

from tests.conftest import docker_rm


@pytest.mark.gpu
def test_vllm_reward():
    model_path = os.getenv('NEMO_SKILLS_TEST_HF_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_HF_MODEL to run this test")
    model_type = os.getenv('NEMO_SKILLS_TEST_MODEL_TYPE')
    if not model_type:
        pytest.skip("Define NEMO_SKILLS_TEST_MODEL_TYPE to run this test")
    prompt_template = 'llama3-instruct' if model_type == 'llama' else 'qwen-instruct'

    input_dir = "/nemo_run/code/tests/data"
    output_dir = f"/tmp/nemo-skills-tests/{model_type}/rm"

    docker_rm([output_dir])

    cmd = (
        f"ns generate "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type vllm "
        f"    --generation_type reward "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    --input_dir={input_dir} "
        f"    --output_dir={output_dir} "
        f"    --num_random_seeds=1 "
        f"    --preprocess_cmd='cp {input_dir}/output-rs0.test {input_dir}/output-rs0.jsonl' "
        f"    ++prompt_config=generic/math "
        f"    ++prompt_template={prompt_template} "
        f"    ++max_samples=10 "
        f"    ++skip_filled=False "
    )
    subprocess.run(cmd, shell=True, check=True)

    output_file = f"{output_dir}/output-rs0.jsonl"

    # no evaluation by default - checking just the number of lines and that there is a "reward_model_score" key
    with open(output_file) as fin:
        lines = fin.readlines()
    assert len(lines) == 10
    for line in lines:
        data = json.loads(line)
        assert 'reward_model_score' in data
