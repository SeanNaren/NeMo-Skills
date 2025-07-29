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
# See the License for the specific lang

import json
import logging
from typing import Union

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def read_predictions(predictions, line_idx, file_handles):
    data = []
    for file_idx, prediction in enumerate(predictions):
        try:
            prediction_dict = json.loads(prediction)
        except Exception as e:
            LOG.error(f"\n\n ***** Error reading line %s in file %s: %s", line_idx + 1, file_handles[file_idx].name, e)
            raise
        data.append(prediction_dict)

    return data


def is_correct_judgement(judgement, return_none=False) -> Union[bool, None]:
    if 'Judgement:' in judgement:
        verdict = judgement.split('Judgement:')[-1].strip()
        if verdict.lower().startswith('yes'):
            return True
        elif verdict.lower().startswith('no'):
            return False
    
    if return_none:
        return None
    else:
        return False  # improper judgement format, so have to judge as false
