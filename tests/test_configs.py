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

import subprocess


def test_error_on_extra_params():
    """Testing that when we pass in any unsupported parameters, there is an error."""

    # top-level
    # test is not supported
    cmd = (
        "python nemo_skills/inference/generate.py "
        "    ++prompt_config=generic/math "
        "    ++output_file=./test-results/gsm8k/output.jsonl "
        "    ++input_file=./nemo_skills/dataset/gsm8k/test.jsonl "
        "    ++server.server_type=nemo "
        "    ++test=1"
    )
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        assert "got an unexpected keyword argument 'test'" in e.stderr.decode()

    # inside nested dataclass
    cmd = (
        "python nemo_skills/inference/generate.py "
        "    ++prompt_config=generic/math "
        "    ++output_file=./test-results/gsm8k/output.jsonl "
        "    ++inference.num_few_shots=0 "
        "    ++input_file=./nemo_skills/dataset/gsm8k/test.jsonl "
        "    ++server.server_type=nemo "
    )
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        assert "got an unexpected keyword argument 'num_few_shots'" in e.stderr.decode()

    # sandbox.sandbox_host is not supported
    cmd = (
        "python nemo_skills/evaluation/evaluate_results.py "
        "    ++input_files=./test-results/gsm8k/output.jsonl "
        "    ++eval_type=math "
        "    ++eval_config.sandbox.sandbox_type=local "
        "    ++eval_config.sandbox.sandbox_host=123 "
        "    ++remove_thinking=false "
    )
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        assert "got an unexpected keyword argument 'sandbox'" in e.stderr.decode()
