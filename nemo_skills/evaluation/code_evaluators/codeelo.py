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
import re
import time
from math import ceil

import requests

from nemo_skills.utils import nested_dataclass, unroll_files


@nested_dataclass(kw_only=True)
class CodeEloEvaluatorConfig:
    """
    Attributes:
        authorization_key: The API key for authorization.
        test_batch_size: Number of different entries to group, submit, and evaluate at once.
        language: The source type for submission (e.g., "python.pypy3-64").
        polling_interval_seconds: Seconds to wait between checking submission status.
        submission_url: The URL for the submission API endpoint.
        reports_url: The URL for the submission reports API endpoint.
    """
    dataset: str = "codeelo"
    authorization_key: str = ""  # Replace with your authorization key
    test_batch_size: int = 5
    language: str = "python"  # set to python or cpp, or a valid source type below
    polling_interval_seconds: int = 5
    submission_url: str = "https://codeforces.com/api/v2/submissions"
    reports_url: str = "https://codeforces.com/api/v2/submissionReports"

    # set of all valid source types
    _ALLOWED_SOURCE_TYPES = {
        "pas.dpr", "pas.fpc", "php.5", "python.2", "csharp.mono", "haskell.ghc", "perl.5",
        "ocaml", "scala", "d", "python.3", "go", "v8.3", "java8", "python.pypy2", "python.pypy3",
        "c.gcc11", "pas.pascalabc", "cpp.g++17", "v8.nodejs", "csharp.dotnet-core", "ruby.3",
        "python.pypy3-64", "rust.2021", "csharp.dotnet-sdk-6", "kotlin17", "java21", "kotlin19",
        "cpp.gcc13-64-winlibs-g++20", "cpp.gcc14-64-msys2-g++23"
    }

    def __post_init__(self):
        alias_map = {
            "python": "python.pypy3-64",
            "cpp": "cpp.g++17",
        }
        if self.language in alias_map:
            self.language = alias_map[self.language]

        # Perform the validation check on the (potentially updated) source_type
        if self.language not in self._ALLOWED_SOURCE_TYPES:
            raise ValueError(
                f"Invalid source_type: '{self.language}'. "
                f"It must be one of {sorted(list(self._ALLOWED_SOURCE_TYPES))}"
            )


def eval_codeelo(cfg):
    """
    Evaluates code generation solutions by submitting them to the CodeElo API in batches.
    """
    eval_config = CodeEloEvaluatorConfig(_init_nested=True, **cfg.eval_config)

    assert eval_config.authorization_key, "Please provide your authorization key."

    submission_headers = {
        "Authorization": f"Bearer {eval_config.authorization_key}",
        "Content-Type": "application/json",
    }
    reports_headers = {"Authorization": f"Bearer {eval_config.authorization_key}"}

    for jsonl_file in unroll_files(cfg.input_files):
        with open(jsonl_file) as f:
            samples = [json.loads(line) for line in f]

        if not samples:
            print(f"Warning: No samples found in {jsonl_file}.")
            continue

        output_path = jsonl_file.replace('.jsonl', '_eval_results.jsonl')

        with open(output_path, 'w') as f_out:
            batch_size = eval_config.test_batch_size
            num_batches = ceil(len(samples) / batch_size)

            for i in range(0, len(samples), batch_size):
                batch_entries = samples[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                print(f"--- Processing Batch {batch_num}/{num_batches} ---")

                submission_tracking = []

                # 1. VALIDATE AND SUBMIT THE ENTIRE BATCH
                for entry in batch_entries:
                    full_generation = entry['generation']
                    get_output = re.findall(r'```(?:[a-zA-Z]+\n)?(.*?)```', full_generation, re.DOTALL)

                    if not get_output:
                        print(f"  SKIPPING entry for contest {entry['cf_contest_id']}: Code block was not found.")
                        result = {**entry, "is_correct": False, "api_response": {"error": "code block was not found"}}
                        f_out.write(json.dumps(result) + '\n')
                        continue

                    completion = get_output[0]
                    completion = re.sub(r'^\s*(?:[Pp]ython\s*\n|[Jj]ava\s*\n|[Cc]\+\+\s*\n|[Cc]\s*\n)', '',
                                        completion, flags=re.IGNORECASE)
                    if "def main" in completion and not re.search(r'if __name__ == [\'"]__main__[\'"]:|main\(\)',
                                                                  completion):
                        completion += "\n\nmain()"
                    if "threading.Thread(target=main).start()" in completion:
                        completion = completion.replace("threading.Thread(target=main).start()", "main()")

                    submission_payload = {
                        "contestId": entry['cf_contest_id'], "problemIndex": entry['cf_index'],
                        "sourceType": eval_config.language, "sourceText": completion,
                    }
                    try:
                        print(
                            f"  Submitting solution for contest {entry['cf_contest_id']}, problem {entry['cf_index']}...")
                        response_submission = requests.post(eval_config.submission_url, headers=submission_headers,
                                                            json=submission_payload)
                        response_submission.raise_for_status()
                        submission_result = response_submission.json()
                        napi_id = str(submission_result["napiSubmissionId"])
                        submission_tracking.append({'entry': entry, 'napi_id': napi_id})
                    except requests.RequestException as e:
                        print(f"  ERROR submitting entry: {e}. It will be marked as incorrect.")
                        result = {**entry, "is_correct": False, "api_response": {"error": str(e)}}
                        f_out.write(json.dumps(result) + '\n')

                if not submission_tracking:
                    print("No submissions to poll in this batch. Moving to next.")
                    continue

                batch_napi_ids = [sub['napi_id'] for sub in submission_tracking]
                napi_ids_str = ",".join(batch_napi_ids)
                print(f"\n  Polling for {len(batch_napi_ids)} submissions in batch {batch_num}...")

                while True:
                    try:
                        response_reports = requests.get(eval_config.reports_url, headers=reports_headers,
                                                        params={"napiSubmissionIds": napi_ids_str})
                        response_reports.raise_for_status()
                        reports_result = response_reports.json()

                        finished_reports = [r for r in reports_result.get("napiSubmissionReports", []) if
                                            r.get("status") == "COMPLETED"]

                        if len(finished_reports) == len(batch_napi_ids):
                            final_reports = reports_result
                            print(f"  All {len(batch_napi_ids)} submissions in batch have been judged.")
                            break
                        else:
                            print(
                                f"  Still waiting for results... ({len(finished_reports)}/{len(batch_napi_ids)} judged)")
                    except requests.RequestException as e:
                        print(f"  Error fetching reports: {e}")
                        final_reports = {"error": "Failed to fetch final batch report."}
                        break
                    time.sleep(eval_config.polling_interval_seconds)

                results_map = {str(r['napiSubmissionId']): r for r in final_reports.get("napiSubmissionReports", [])}

                for sub in submission_tracking:
                    entry = sub['entry']
                    napi_id = sub['napi_id']
                    report_for_entry = results_map.get(napi_id)
                    is_correct = False
                    if report_for_entry and report_for_entry.get("verdict") == "OK":
                        is_correct = True
                    result = {
                        **entry,
                        "is_correct": is_correct,
                        "api_response": report_for_entry or {"error": "Report not found for this submission."},
                    }
                    f_out.write(json.dumps(result) + '\n')

        print(f"\nEvaluation results for {jsonl_file} saved to {output_path}")
