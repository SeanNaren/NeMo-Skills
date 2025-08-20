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
DATASET_GROUP = 'math'
METRICS_TYPE = "math"
EVAL_ARGS = "++eval_type=math"
GENERATION_ARGS = "++prompt_config=generic/math"

# some answers are not possible to compare symbolically, so have to use a judge model
# setting openai judge by default, but can be overriden from command line for a locally hosted model
JUDGE_PIPELINE_ARGS = {
    "generation_type": "math_judge",
    "model": "gpt-4.1",
    "server_type": "openai",
    "server_address": "https://api.openai.com/v1",
}
