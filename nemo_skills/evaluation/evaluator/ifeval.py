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

import json
import logging
import subprocess
from pathlib import Path

from nemo_skills.utils import get_logger_name, unroll_files

LOG = logging.getLogger(get_logger_name(__file__))


def eval_if(cfg):
    for jsonl_file in unroll_files(cfg.input_files):
        parent_dir = Path(jsonl_file).absolute().parent
        cmd = (
            'cd /opt/benchmarks/google-research && python -m instruction_following_eval.evaluation_main '
            f'--input_data={jsonl_file} '
            f'--input_response_data={jsonl_file} '
            f'--output_dir={parent_dir} '
        )
        subprocess.run(cmd, shell=True, check=True)
        # fusing eval metrics back into the generation file
        with open(jsonl_file, "rt", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f]

        with open(parent_dir / 'eval_results_loose.jsonl', 'rt', encoding="utf-8") as f:
            eval_results = [json.loads(line) for line in f]
        for sample, eval_result in zip(samples, eval_results):
            sample['loose_eval'] = eval_result

        with open(parent_dir / 'eval_results_strict.jsonl', 'rt', encoding="utf-8") as f:
            eval_results = [json.loads(line) for line in f]
        for sample, eval_result in zip(samples, eval_results):
            sample['strict_eval'] = eval_result

        with open(jsonl_file, "wt", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        # removing metric files to avoid reusing them
        (parent_dir / 'eval_results_loose.jsonl').unlink()
        (parent_dir / 'eval_results_strict.jsonl').unlink()
