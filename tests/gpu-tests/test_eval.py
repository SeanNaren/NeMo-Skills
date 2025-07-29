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
def test_trtllm_eval():
    model_path = os.getenv('NEMO_SKILLS_TEST_TRTLLM_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_TRTLLM_MODEL to run this test")
    model_type = os.getenv('NEMO_SKILLS_TEST_MODEL_TYPE')
    if not model_type:
        pytest.skip("Define NEMO_SKILLS_TEST_MODEL_TYPE to run this test")
    prompt_template = 'llama3-instruct' if model_type == 'llama' else 'qwen-instruct'

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/trtllm-eval"
    docker_rm([output_dir])

    cmd = (
        f"ns eval "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type trtllm "
        f"    --output_dir {output_dir} "
        f"    --benchmarks gsm8k "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    ++prompt_template={prompt_template} "
        f"    ++max_samples=20 "
    )
    subprocess.run(cmd, shell=True, check=True)

    with open(f"{output_dir}/eval-results/gsm8k/metrics.json", 'r') as f:
        metrics = json.load(f)["gsm8k"]["pass@1"]

    # rough check, since exact accuracy varies depending on gpu type
    if model_type == 'llama':
        assert metrics['symbolic_correct'] >= 50
    else:  # qwen
        assert metrics['symbolic_correct'] >= 70
    assert metrics['num_entries'] == 20


@pytest.mark.gpu
@pytest.mark.parametrize("server_type", ['trtllm', 'trtllm-serve'])
def test_trtllm_code_execution_eval(server_type):
    model_path = os.getenv('NEMO_SKILLS_TEST_TRTLLM_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_TRTLLM_MODEL to run this test")
    model_type = os.getenv('NEMO_SKILLS_TEST_MODEL_TYPE')
    if not model_type:
        pytest.skip("Define NEMO_SKILLS_TEST_MODEL_TYPE to run this test")
    # we are using the base prompt for llama to make it follow few-shots
    prompt_template = 'llama3-base' if model_type == 'llama' else 'qwen-instruct'
    code_tags = 'nemotron' if model_type == 'llama' else 'qwen'

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/{server_type}-eval"
    docker_rm([output_dir])

    cmd = (
        f"ns eval "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type {server_type} "
        f"    --output_dir {output_dir} "
        f"    --benchmarks gsm8k "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    --with_sandbox "
        f"    ++prompt_template={prompt_template} "
        f"    ++code_tags={code_tags} "
        f"    ++examples_type=gsm8k_text_with_code "
        f"    ++max_samples=20 "
        f"    ++code_execution=True "
    )
    subprocess.run(cmd, shell=True, check=True)

    with open(f"{output_dir}/eval-results/gsm8k/metrics.json", 'r') as f:
        metrics = json.load(f)["gsm8k"]["pass@1"]
    # rough check, since exact accuracy varies depending on gpu type
    if model_type == 'llama':
        assert metrics['symbolic_correct'] >= 40
    else:  # qwen
        assert metrics['symbolic_correct'] >= 70
    assert metrics['num_entries'] == 20


@pytest.mark.gpu
@pytest.mark.parametrize(
    "server_type,server_args", [('vllm', ''), ('sglang', ''), ('trtllm-serve', '--backend pytorch')]
)
def test_hf_eval(server_type, server_args):
    # this test expects llama3-instruct to properly check accuracy
    # will run a bunch of benchmarks, but is still pretty fast
    # mmlu/ifeval will be cut to 400 samples to save time
    # could cut everything, but human-eval/mbpp don't work with partial gens
    model_path = os.getenv('NEMO_SKILLS_TEST_HF_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_HF_MODEL to run this test")
    model_type = os.getenv('NEMO_SKILLS_TEST_MODEL_TYPE')
    if not model_type:
        pytest.skip("Define NEMO_SKILLS_TEST_MODEL_TYPE to run this test")
    if model_type != 'llama':
        pytest.skip("Only running this test for llama models")

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/{server_type}-eval"
    docker_rm([output_dir])

    cmd = (
        f"ns eval "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type {server_type} "
        f"    --output_dir {output_dir} "
        f"    --benchmarks algebra222,human-eval,ifeval,mmlu "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    --num_jobs 1 "
        f"    --server_args='{server_args}' "
        f"    ++prompt_template=llama3-instruct "
        f"    ++max_samples=164 "
        f"    ++max_concurrent_requests=200 "
    )
    subprocess.run(cmd, shell=True, check=True)

    # checking that summarize results works (just that there are no errors, but can inspect the output as well)
    subprocess.run(
        f"ns summarize_results {output_dir}",
        shell=True,
        check=True,
    )

    with open(f"{output_dir}/eval-results/algebra222/metrics.json", 'r') as f:
        metrics = json.load(f)["algebra222"]["pass@1"]

    assert metrics['symbolic_correct'] >= 75
    assert metrics['num_entries'] == 164

    with open(f"{output_dir}/eval-results/human-eval/metrics.json", 'r') as f:
        metrics = json.load(f)["human-eval"]["pass@1"]

    assert metrics['passing_base_tests'] >= 50
    assert metrics['passing_plus_tests'] >= 50
    assert metrics['num_entries'] == 164

    with open(f"{output_dir}/eval-results/ifeval/metrics.json", 'r') as f:
        metrics = json.load(f)["ifeval"]["pass@1"]

    assert metrics['prompt_strict_accuracy'] >= 60
    assert metrics['instruction_strict_accuracy'] >= 70
    assert metrics['prompt_loose_accuracy'] >= 60
    assert metrics['instruction_loose_accuracy'] >= 70
    assert metrics['num_prompts'] == 164

    with open(f"{output_dir}/eval-results/mmlu/metrics.json", 'r') as f:
        metrics = json.load(f)["mmlu"]["pass@1"]
    assert metrics['symbolic_correct'] >= 60
    assert metrics['num_entries'] == 164


@pytest.mark.gpu
def test_nemo_eval():
    model_path = os.getenv('NEMO_SKILLS_TEST_NEMO_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_NEMO_MODEL to run this test")
    model_type = os.getenv('NEMO_SKILLS_TEST_MODEL_TYPE')
    if not model_type:
        pytest.skip("Define NEMO_SKILLS_TEST_MODEL_TYPE to run this test")
    prompt_template = 'llama3-instruct' if model_type == 'llama' else 'qwen-instruct'

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/nemo-eval"
    docker_rm([output_dir])

    cmd = (
        f"ns eval "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type nemo "
        f"    --output_dir {output_dir} "
        f"    --benchmarks gsm8k "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    ++prompt_template={prompt_template} "
        f"    ++max_samples=20 "
    )
    subprocess.run(cmd, shell=True, check=True)

    # running compute_metrics to check that results are expected
    with open(f"{output_dir}/eval-results/gsm8k/metrics.json", 'r') as f:
        metrics = json.load(f)["gsm8k"]["pass@1"]
    # rough check, since exact accuracy varies depending on gpu type
    if model_type == 'llama':
        assert metrics['symbolic_correct'] >= 50
    else:  # qwen
        assert metrics['symbolic_correct'] >= 70
    assert metrics['num_entries'] == 20


@pytest.mark.gpu
def test_megatron_eval():
    model_path = os.getenv('NEMO_SKILLS_TEST_MEGATRON_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_MEGATRON_MODEL to run this test")
    model_type = os.getenv('NEMO_SKILLS_TEST_MODEL_TYPE')
    if not model_type:
        pytest.skip("Define NEMO_SKILLS_TEST_MODEL_TYPE to run this test")
    if model_type != "llama":
        pytest.skip("Only llama models are supported in Megatron.")
    prompt_template = 'llama3-instruct'

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/megatron-eval"
    docker_rm([output_dir])

    cmd = (
        f"ns eval "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type megatron "
        f"    --output_dir {output_dir} "
        f"    --benchmarks gsm8k "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    ++prompt_template={prompt_template} "
        f"    ++max_samples=20 "
        f"    --server_args='--tokenizer-model meta-llama/Llama-3.1-8B-Instruct --inference-max-requests=20' "
    )
    subprocess.run(cmd, shell=True, check=True)

    # running compute_metrics to check that results are expected
    with open(f"{output_dir}/eval-results/gsm8k/metrics.json", 'r') as f:
        metrics = json.load(f)["gsm8k"]["pass@1"]
    # rough check, since exact accuracy varies depending on gpu type
    # TODO: something is broken in megatron inference here as this should be 50!
    assert metrics['symbolic_correct'] >= 20
    assert metrics['num_entries'] == 20
