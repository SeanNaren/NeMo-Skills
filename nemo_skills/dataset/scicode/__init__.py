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

# settings that define how evaluation should be done by default (all can be changed from cmdline)
DATASET_GROUP = 'code'
METRICS_TYPE = 'scicode'
# generation is a dictionary instead of string and remove_thinking is done during inference
EVAL_ARGS = "++eval_type=scicode ++remove_thinking=False"
GENERATION_ARGS = "++prompt_config=eval/scicode/default"
GENERATION_MODULE = "nemo_skills.inference.eval.scicode"
REQUIRES_SANDBOX = True
