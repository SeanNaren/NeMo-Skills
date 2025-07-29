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

import os
import subprocess
from pathlib import Path

import pytest

from tests.conftest import docker_rm


@pytest.mark.gpu
def test_hf_trtllm_conversion():
    model_path = os.getenv('NEMO_SKILLS_TEST_HF_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_HF_MODEL to run this test")
    model_type = os.getenv('NEMO_SKILLS_TEST_MODEL_TYPE')
    if not model_type:
        pytest.skip("Define NEMO_SKILLS_TEST_MODEL_TYPE to run this test")
    hf_model_name = (
        'meta-llama/Meta-Llama-3.1-8B-Instruct' if model_type == 'llama' else 'Qwen/Qwen2.5-Math-7B-Instruct'
    )

    output_model = f"/tmp/nemo-skills-tests/{model_type}/conversion/hf-to-trtllm/model"
    docker_rm([output_model])

    cmd = (
        f"ns convert "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --input_model {model_path} "
        f"    --output_model {output_model} "
        f"    --convert_from hf "
        f"    --convert_to trtllm "
        f"    --model_type {model_type} "
        f"    --num_gpus 1 "
        f"    --hf_model_name {hf_model_name} "
    )

    subprocess.run(cmd, shell=True, check=True)
    assert Path(f"{output_model}").exists()


@pytest.mark.gpu
def test_hf_nemo_conversion():
    model_path = os.getenv('NEMO_SKILLS_TEST_HF_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_HF_MODEL to run this test")
    model_type = os.getenv('NEMO_SKILLS_TEST_MODEL_TYPE')
    if not model_type:
        pytest.skip("Define NEMO_SKILLS_TEST_MODEL_TYPE to run this test")
    hf_model_name = (
        'meta-llama/Meta-Llama-3.1-8B-Instruct' if model_type == 'llama' else 'Qwen/Qwen2.5-Math-7B-Instruct'
    )

    output_model = f"/tmp/nemo-skills-tests/{model_type}/conversion/hf-to-nemo/model"
    docker_rm([output_model])

    cmd = (
        f"ns convert "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --input_model {model_path} "
        f"    --output_model {output_model} "
        f"    --convert_from hf "
        f"    --convert_to nemo "
        f"    --model_type {model_type} "
        f"    --num_gpus 1 "
        f"    --hf_model_name {hf_model_name} "
        f"    --override "
    )

    subprocess.run(cmd, shell=True, check=True)
    assert Path(output_model).exists()


@pytest.mark.gpu
def test_hf_megatron_conversion():
    model_path = os.getenv('NEMO_SKILLS_TEST_HF_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_HF_MODEL to run this test")
    model_type = os.getenv('NEMO_SKILLS_TEST_MODEL_TYPE')
    if not model_type:
        pytest.skip("Define NEMO_SKILLS_TEST_MODEL_TYPE to run this test")
    if model_type != "llama":
        pytest.skip("Only llama models are supported in Megatron.")
    hf_model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'

    output_model = f"/tmp/nemo-skills-tests/{model_type}/conversion/hf-to-megatron/model"
    docker_rm([output_model])

    cmd = (
        f"ns convert "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --input_model {model_path} "
        f"    --output_model {output_model} "
        f"    --convert_from hf "
        f"    --convert_to megatron "
        f"    --model_type {model_type} "
        f"    --num_gpus 1 "
        f"    --hf_model_name {hf_model_name} "
    )

    subprocess.run(cmd, shell=True, check=True)
    assert Path(output_model).exists()


@pytest.mark.gpu
def test_nemo_hf_conversion():
    model_path = os.getenv('NEMO_SKILLS_TEST_NEMO_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_NEMO_MODEL to run this test")
    model_type = os.getenv('NEMO_SKILLS_TEST_MODEL_TYPE')
    if not model_type:
        pytest.skip("Define NEMO_SKILLS_TEST_MODEL_TYPE to run this test")
    hf_model_name = (
        'meta-llama/Meta-Llama-3.1-8B-Instruct' if model_type == 'llama' else 'Qwen/Qwen2.5-Math-7B-Instruct'
    )

    output_model = f"/tmp/nemo-skills-tests/{model_type}/conversion/nemo-to-hf/model"
    docker_rm([output_model])

    cmd = (
        f"ns convert "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --input_model {model_path} "
        f"    --output_model {output_model} "
        f"    --convert_from nemo "
        f"    --convert_to hf "
        f"    --model_type {model_type} "
        f"    --num_gpus 1 "
        f"    --hf_model_name {hf_model_name} "
    )

    subprocess.run(cmd, shell=True, check=True)
    assert Path(output_model).exists()
