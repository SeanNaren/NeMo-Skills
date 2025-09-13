#!/bin/bash

python swerank_embed.py --dataset princeton-nlp/SWE-bench_Lite --top_k 5 --output_path swebench_lite_5.json
python swerank_embed.py --dataset princeton-nlp/SWE-bench_Verified --top_k 5 --output_path swebench_verified_5.json
python swerank_embed.py --dataset princeton-nlp/SWE-bench_Lite --top_k 10 --output_path swebench_lite_10.json
python swerank_embed.py --dataset princeton-nlp/SWE-bench_Verified --top_k 10 --output_path swebench_verified_10.json
