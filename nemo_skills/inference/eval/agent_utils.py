import json
import re


def sanitize_message(msg: dict) -> dict:
    """Ensure tool_call arguments in assistant messages are valid JSON.

    Models sometimes generate invalid JSON escape sequences in tool call
    arguments (e.g. \\q, \\0 from C++ code). When these messages are sent
    back to the API, the server fails to parse the nested JSON.
    """
    tool_calls = msg.get("tool_calls")
    if msg.get("role") != "assistant" or not tool_calls:
        return msg
    for tc in tool_calls:
        func = tc.get("function") or {}
        args_str = func.get("arguments")
        if not isinstance(args_str, str) or not args_str:
            continue
        try:
            json.loads(args_str)
        except json.JSONDecodeError:
            # Fix invalid escape sequences: replace \X (where X is not a
            # valid JSON escape char) with \\X so the backslash is literal.
            fixed = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", args_str)
            try:
                parsed = json.loads(fixed)
                func["arguments"] = json.dumps(parsed)
            except json.JSONDecodeError:
                # Cannot recover — replace with empty args so the turn
                # is still representable as valid JSON.
                func["arguments"] = "{}"
    return msg


def extract_cpp(text: str | None) -> str | None:
    """Extract the last C++ code block from markdown-formatted text."""
    if not text:
        return None
    matches = re.findall(r"```(?:cpp|c\+\+)\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else None


def normalize_scores(test_case_results: dict) -> dict:
    """Normalize evaluator outputs to a common shape.

    ICPC-style results have a flat dict with ``outputs`` and ``score``.
    IOI-style results have a dict of subtask dicts.
    """
    if (
        isinstance(test_case_results, dict)
        and "outputs" in test_case_results
        and "score" in test_case_results
        and isinstance(test_case_results.get("outputs"), list)
    ):
        return {"overall": {"score": float(test_case_results["score"]), "outputs": test_case_results["outputs"]}}
    return {
        k: {"score": float(v.get("score", 0.0)), "outputs": list(v.get("outputs", []))}
        for k, v in test_case_results.items()
    }


def calculate_avg_score(normalized_results: dict) -> float:
    """Calculate average pass rate across all test outputs."""
    all_outputs = []
    for subtask_data in normalized_results.values():
        all_outputs.extend(subtask_data.get("outputs", []))
    if not all_outputs:
        return 0.0
    total = len(all_outputs)
    passed = sum(1.0 if float(o.get("score", 0.0)) == 1.0 else 0.0 for o in all_outputs)
    return float(passed / total)


def filter_test_outputs(test_case_results: dict, max_limit: int) -> dict:
    """Remove passed tests and truncate stdout/stderr in outputs."""

    def truncate(val):
        if isinstance(val, str) and len(val) > max_limit:
            return val[:max_limit] + "...<truncated>"
        return val

    def _filter_outputs(outputs):
        return [
            {
                k: truncate(v) if k in ("run_stdout", "run_stderr", "compile_stdout", "compile_stderr") else v
                for k, v in o.items()
            }
            for o in outputs
            if float(o.get("score", 0.0)) != 1.0
        ]

    # ICPC-style
    if (
        isinstance(test_case_results, dict)
        and "outputs" in test_case_results
        and "score" in test_case_results
        and isinstance(test_case_results.get("outputs"), list)
    ):
        return {**test_case_results, "outputs": _filter_outputs(test_case_results["outputs"])}
    # IOI-style
    return {k: {**v, "outputs": _filter_outputs(v.get("outputs", []))} for k, v in test_case_results.items()}


def parse_max_time(max_time_str: str | None) -> float | None:
    """Parse ``hh:mm:ss`` string into total seconds."""
    if not max_time_str:
        return None
    parts = max_time_str.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid max_time format: {max_time_str}. Expected hh:mm:ss")
    hours, minutes, seconds = map(int, parts)
    return hours * 3600 + minutes * 60 + seconds


def process_submission_result(
    eval_result: dict,
    data_point: dict,
    *,
    explicit_feedback: bool,
    avg_score: bool,
    max_limit_in_test_output: int,
) -> dict:
    """Process an evaluator result into a tool output dict.

    Handles IOI subtask filtering (only shows/checks the target subtask)
    and ICPC full-result mode.

    Returns a dict with keys:
        tool_output (str): JSON string to return to the agent
        success (bool): whether the target subtask (or all tests for ICPC) passed
        target_score (float): score achieved on the target subtask
        target_max_score (float): maximum possible score for the target subtask
    """
    test_case_results = eval_result.get("test_case_results", {})

    is_ioi = "ioi_id" in data_point
    if is_ioi and "subtask" in data_point:
        subtask_name = data_point["subtask"]
        if subtask_name in test_case_results:
            test_case_results = {subtask_name: test_case_results[subtask_name]}

    normalized = normalize_scores(test_case_results)

    if explicit_feedback:
        tool_out_dict = {
            **eval_result,
            "test_case_results": filter_test_outputs(test_case_results, max_limit_in_test_output),
        }
    else:
        if is_ioi and "subtask_score" in data_point:
            max_score = data_point["subtask_score"]
            subtask_scores = {k: f"{v['score']}/{max_score}" for k, v in normalized.items()}
        else:
            subtask_scores = {k: v["score"] for k, v in normalized.items()}
        tool_out_dict = {"subtask_scores": subtask_scores}

    avg_score_val = calculate_avg_score(normalized)
    if avg_score:
        tool_out_dict["avg_score"] = avg_score_val

    if is_ioi and "subtask_score" in data_point:
        max_score_f = float(data_point["subtask_score"])
        success = bool(normalized) and all(float(v["score"]) == max_score_f for v in normalized.values())
    else:
        success = bool(normalized) and all(float(v["score"]) == 1.0 for v in normalized.values())

    tool_out_dict["success"] = success

    return {
        "tool_output": json.dumps(tool_out_dict),
        "success": success,
        "target_score": avg_score_val,
        "target_max_score": 1.0,
    }
