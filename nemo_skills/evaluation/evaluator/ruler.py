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
import re

from tqdm import tqdm

from nemo_skills.utils import get_logger_name, nested_dataclass, unroll_files

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class RulerEvaluatorConfig:
    parse_func: str = "default"
    match_type: str


def eval_ruler(cfg):
    def default_parse(prediction):
        prediction = prediction.strip()
        # Remove all non-printable characters
        np_pattern = re.compile(r'[\x00-\x1f]')
        pp_predict = np_pattern.sub('\n', prediction).strip()
        return pp_predict

    def string_match_all_single(preds, refs):
        """the metric function with input (predictions: [str], references: [[str]]) to compute score."""
        preds = [preds]
        refs = [refs]
        score = [
            sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref) for pred, ref in zip(preds, refs)
        ][0]
        return score

    def string_match_part_single(preds, refs):
        preds = [preds]
        refs = [refs]
        score = [
            sum([max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) for pred, ref in zip(preds, refs)])
        ][0]
        return score

    eval_config = RulerEvaluatorConfig(**cfg.eval_config)

    parse_funcs = {
        'default': default_parse,
    }
    match_type_funcs = {
        'all': string_match_all_single,
        'part': string_match_part_single,
    }

    for file in unroll_files(cfg.input_files):
        with open(file, 'rt', encoding='utf-8') as fin:
            data = [json.loads(line) for line in fin]
        with open(file, 'wt', encoding='utf-8') as fout:
            for sample in tqdm(data):
                parse_result = parse_funcs[eval_config.parse_func](sample['generation'])
                sample['is_correct'] = match_type_funcs[eval_config.match_type](
                    sample['generation'], sample['expected_answer']
                )
                sample['predicted_answer'] = parse_result
                fout.write(json.dumps(sample) + "\n")
