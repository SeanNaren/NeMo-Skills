import json

input_path = "/mnt/ssd/htamoyan/output_eval/artsiv/qwen3_235b_swe_bench_lite_t4_v17/eval-results/swe-bench-lite/output.jsonl"
output_path = "/mnt/ssd/htamoyan/output_eval/artsiv/qwen3_235b_swe_bench_lite_t4_v17/eval-results/swe-bench-lite/output_failed.jsonl"

def to_float(x):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.strip())
        except ValueError:
            return None
    return None

def precisions_from_eval_status(obj):
    """Return list of precision floats found at eval_status[*].file_level.precision."""
    out = []
    es = obj.get("eval_status")
    if isinstance(es, list):
        for entry in es:
            if isinstance(entry, dict):
                fl = entry.get("file_level")
                if isinstance(fl, dict) and "precision" in fl:
                    p = to_float(fl["precision"])
                    if p is not None:
                        out.append(p)
    elif isinstance(es, dict):
        fl = es.get("file_level")
        if isinstance(fl, dict) and "precision" in fl:
            p = to_float(fl["precision"])
            if p is not None:
                out.append(p)
    return out

total = 0
with_precisions = 0
failed = 0
first_fails = []

with open(input_path, "r", encoding="utf-8") as infile, \
     open(output_path, "w", encoding="utf-8") as outfile:
    for ln, line in enumerate(infile, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # Skip malformed lines
            continue

        total += 1
        precs = precisions_from_eval_status(obj)
        if precs:
            with_precisions += 1

        # If ANY precision is not exactly 1.0, mark as failed
        if any(p != 1.0 for p in precs):
            outfile.write(line + "\n")
            failed += 1
            if len(first_fails) < 10:
                first_fails.append((ln, precs))

print(f"Total lines parsed: {total}")
print(f"Lines with eval_status[*].file_level.precision: {with_precisions}")
print(f"Failed lines written: {failed} -> {output_path}")
if first_fails:
    print("First few failing lines (line_number, precisions):")
    for item in first_fails:
        print(item)