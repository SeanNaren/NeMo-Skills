#!/bin/bash

RUN_ID="v3_t1_temp099"

MODEL_SIZE_NUM=235
MODEL_SIZE="${MODEL_SIZE_NUM}b"
MODEL_PATH=/hf_models/sharded_0.7.0_sglang/Qwen3-235B-A22B-Thinking-2507-tp16/
# OUTPUT_DIR="/lustre/fsw/portfolios/llmservice/users/htamoyan/locagent/output_eval/artsiv/tamohannes/qwen3_${MODEL_SIZE_NUM}b_swe_bench_lite_${RUN_ID}"
OUTPUT_DIR="/lustre/fsw/portfolios/llmservice/users/htamoyan/locagent/output_eval/artsiv/tamohannes/${RUN_ID}"
REPO_STRUCTURES_DIR=/lustre/fsw/portfolios/llmservice/users/wasiuddina/data/swe_datasets/SWE-bench_Lite/repo_structures/
DATASET_DIR="/lustre/fsw/portfolios/llmservice/users/htamoyan/datasets/"

NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 ns eval \
    --cluster cw-dfw \
    --benchmarks swe-bench-lite \
    --data_dir $DATASET_DIR \
    --expname "${RUN_ID}" \
    --model "${MODEL_PATH}" \
    --server_type "sglang" \
    --num_chunks 50 \
    --server_nodes 2 \
    --server_gpus 8 \
    --server_args "--load-format sharded_state --tensor-parallel-size 16" \
    --dependent_jobs=1 \
    --output_dir "$OUTPUT_DIR" \
    --mount_paths "${REPO_STRUCTURES_DIR}:/repos/" \
    ++skip_filled=True \
    ++prompt_template="qwen-instruct" \
    ++prompt_config="eval/artsiv/system" \
    ++inference.temperature=0.99 \
    ++inference.tokens_to_generate=81920 \
    ++max_seq_length=262144 \
    ++mount_directory=/repos/ \
    ++multi_turn_key=turns \
    ++max_concurrent_requests=512

echo "- Experiment: ${RUN_ID}"
