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

import hashlib
from pathlib import Path

from nemo_skills.pipeline.cli import run_cmd, wrap_arguments
from tests.conftest import docker_rm_and_mkdir


def compute_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def test_multiple_files():
    output_file = f"/tmp/nemo-skills-tests/data/processed_multifile_output.jsonl"
    docker_rm_and_mkdir(output_file)
    run_cmd(
        cluster='test-local',
        config_dir=Path(__file__).parent / 'gpu-tests',
        log_dir='/tmp/nemo-skills-tests/test_multiple_files',
        ctx=wrap_arguments(
            f"python -m nemo_skills.training.prepare_data "
            f"    ++input_files='tests/data/output-rs*.test' "
            f"    ++output_path={output_file} "
            f"    ++prompt_config=generic/math "
            f"    ++prompt_template=llama3-instruct "
            f"    ++exclude_optional_keys=false "
            f"    ++filters.remove_len_outlier_problems=false "
            f"    ++filters.drop_multi_boxed=true "
            f"    ++filters.trim_solutions=true "
            f"    ++filters.drop_incorrect_arithmetic=false "
            f"    ++filters.split_arithmetic=false "
            f"    ++filters.remove_contaminated=false "
            f"    ++num_output_samples=32 "
            f"    ++downsampling_method=fair "
            f"    ++do_shuffle=false "
        ),
    )

    expected_md5 = "7c3129affcf6f31a68775fae6f8c1742"
    output_md5 = compute_md5(output_file)

    assert (
        expected_md5 == output_md5
    ), "MD5 hashes do not match, something is wrong with nemo_skills/training/prepare_data.py"


def test_exclude_keys():
    output_file = f"/tmp/nemo-skills-tests/data/processed_compact_output.jsonl"
    docker_rm_and_mkdir(output_file)
    run_cmd(
        cluster='test-local',
        config_dir=Path(__file__).parent / 'gpu-tests',
        log_dir='/tmp/nemo-skills-tests/test_exclude_keys',
        ctx=wrap_arguments(
            f"python -m nemo_skills.training.prepare_data "
            f"    ++input_files='tests/data/output-rs*.test' "
            f"    ++output_path={output_file} "
            f"    ++prompt_config=generic/math "
            f"    ++prompt_template=llama3-instruct "
            f"    ++exclude_optional_keys=true "
            f"    ++filters.remove_len_outlier_problems=false "
            f"    ++filters.drop_multi_boxed=true "
            f"    ++filters.trim_solutions=true "
            f"    ++filters.drop_incorrect_arithmetic=false "
            f"    ++filters.split_arithmetic=false "
            f"    ++filters.remove_contaminated=false "
            f"    ++num_output_samples=32 "
            f"    ++downsampling_method=fair "
            f"    ++do_shuffle=false ",
        ),
    )

    expected_md5 = "08c9b228faa1065825b68c0c994fcdb4"
    output_md5 = compute_md5(output_file)

    assert (
        expected_md5 == output_md5
    ), "MD5 hashes do not match, something is wrong with nemo_skills/training/prepare_data.py"


def test_code_sft_data():
    output_file = f"/tmp/nemo-skills-tests/data/code_processed_output.jsonl"
    docker_rm_and_mkdir(output_file)
    run_cmd(
        cluster='test-local',
        config_dir=Path(__file__).parent / 'gpu-tests',
        log_dir='/tmp/nemo-skills-tests/test_code_sft_data',
        ctx=wrap_arguments(
            f"python -m nemo_skills.training.prepare_data "
            f"    --config-name=code_sft "
            f"    ++preprocessed_dataset_files='tests/data/code-output.test' "
            f"    ++output_path={output_file} "
            f"    ++prompt_config=generic/codegen "
            f"    ++prompt_template=llama3-instruct "
            f"    ++exclude_optional_keys=false "
            f"    ++filters.drop_incorrect_code_blocks=false "
        ),
    )

    expected_md5 = "a830a174291795cc7db0d1c3ee39de25"
    output_md5 = compute_md5(output_file)

    assert (
        expected_md5 == output_md5
    ), "MD5 hashes do not match, something is wrong with nemo_skills/training/prepare_data.py"


def test_openmathinstruct2():
    output_file = f"/tmp/nemo-skills-tests/data/openmathinstruct2-sft.jsonl"
    docker_rm_and_mkdir(output_file)
    run_cmd(
        cluster='test-local',
        config_dir=Path(__file__).parent / 'gpu-tests',
        log_dir='/tmp/nemo-skills-tests/test_openmathinstruct2',
        ctx=wrap_arguments(
            f"python -m nemo_skills.training.prepare_data "
            f"    ++preprocessed_dataset_files='tests/data/openmathinstruct2.test' "
            f"    ++output_path={output_file} "
            f"    ++prompt_template=llama3-instruct "
            f"    ++prompt_config=generic/math "
            f"    ++output_key=generated_solution "
            f"    ++filters.remove_len_outlier_problems=false "
            f"    ++filters.drop_multi_boxed=false "
            f"    ++filters.trim_prefix=false "
            f"    ++filters.trim_solutions=false "
            f"    ++filters.drop_incorrect_arithmetic=false "
            f"    ++filters.split_arithmetic=false "
            f"    ++filters.remove_contaminated=false "
        ),
    )

    expected_md5 = "981e11051436be68cdc45953888a5685"
    output_md5 = compute_md5(output_file)

    assert (
        expected_md5 == output_md5
    ), "MD5 hashes do not match, something is wrong with nemo_skills/training/prepare_data.py"


def test_aggregate_answers_fill():
    output_dir = "/tmp/nemo-skills-tests/test_majority_filling"
    run_cmd(
        cluster='test-local',
        config_dir=Path(__file__).parent / 'gpu-tests',
        log_dir='/tmp/nemo-skills-tests/test_aggregate_answers',
        ctx=wrap_arguments(
            f"python -m nemo_skills.evaluation.aggregate_answers "
            f"    ++input_dir='tests/data' "
            f"    ++input_files='output-rs*.test' "
            f"    ++mode=fill "
            f"    ++output_dir={output_dir} "
        ),
    )

    # Check md5 of one of the output files
    output_file = f"{output_dir}/output-rs0.test"
    expected_md5 = "20cd998b090603b2049f27a321cc9e27"
    output_md5 = compute_md5(output_file)

    assert (
        expected_md5 == output_md5
    ), "MD5 hashes do not match, something is wrong with nemo_skills/evaluation/aggregate_answers.py"


def test_aggregate_answers_extract():
    output_dir = "/tmp/nemo-skills-tests/test_majority_filling"
    run_cmd(
        cluster='test-local',
        config_dir=Path(__file__).parent / 'gpu-tests',
        log_dir='/tmp/nemo-skills-tests/test_aggregate_answers',
        ctx=wrap_arguments(
            f"python -m nemo_skills.evaluation.aggregate_answers "
            f"    ++input_dir='tests/data' "
            f"    ++input_files='output-rs*.test' "
            f"    ++mode=extract "
            f"    ++output_dir={output_dir} "
        ),
    )

    # Check md5 of one of the output files
    output_file = Path(output_dir) / "output-agg.jsonl"
    expected_md5 = "5f2cdfde69f5eed82c2eb9515c9e07ea"
    output_md5 = compute_md5(output_file)

    print(f"output_md5: {output_md5}")

    assert (
        expected_md5 == output_md5
    ), "MD5 hashes do not match, something is wrong with nemo_skills/evaluation/aggregate_answers.py"
