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


def compute_score(metrics: dict):
    # just an average of all metrics. Here we assume that all tasks are present.
    # if that's not the case, users shouldn't run as a group
    tasks = [
        "niah_single_1",
        "niah_single_2",
        "niah_single_3",
        "niah_multikey_1",
        "niah_multikey_2",
        "niah_multikey_3",
        "niah_multivalue",
        "niah_multiquery",
        "vt",
        "cwe",
        "fwe",
        "qa_1",
        "qa_2",
    ]
    setup = list(metrics.keys())[0].rsplit(".", 1)[0]
    metrics[setup] = {}

    for aggregation in metrics[f"{setup}.niah_single_1"]:
        metrics[setup][aggregation] = {
            "accuracy": sum(metrics[f"{setup}.{task}"][aggregation]["accuracy"] for task in tasks) / len(tasks)
        }

    return metrics
