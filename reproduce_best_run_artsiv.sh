#!/bin/bash

MODEL_SIZE_NUM=235
RUN_ID="try10_1_reproduce_best_artsiv"
MODEL_SIZE="${MODEL_SIZE_NUM}b"
MODEL_PATH=/hf_models/sharded_0.7.0_sglang/Qwen3-235B-A22B-Thinking-2507-tp16/
OUTPUT_DIR="/lustre/fsw/portfolios/llmservice/users/htamoyan/artsiv/output_eval/qwen3_${MODEL_SIZE_NUM}b_swe_bench_lite_${RUN_ID}"
REPO_STRUCTURES_DIR=/lustre/fsw/portfolios/llmservice/users/wasiuddina/data/swe_datasets/SWE-bench_Lite/repo_structures/
DATASET_DIR="/lustre/fsw/portfolios/llmservice/users/htamoyan/datasets/"

echo "============================================"
echo "REPRODUCING BEST RUN WITH ARTSIV"
echo "Target Results:"
echo "- File F1: 83.99%"
echo "- File Precision: 83.06%" 
echo "- File Recall: 86.0%"
echo "- Success Rate: 99.67%"
echo "============================================"

# EXACT configuration from the successful run that achieved 83.99% F1
# Now using ARTSIV instead of LOCAGENT
NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 ns eval \
    --cluster cw-dfw \
    --benchmarks swe-bench-lite \
    --data_dir $DATASET_DIR \
    --expname "eval_qwen_${MODEL_SIZE_NUM}b_swe_bench_lite_artsiv" \
    --model "${MODEL_PATH}" \
    --server_type "sglang" \
    --num_chunks 50 \
    --server_nodes 2 \
    --server_gpus 8 \
    --server_args "--load-format sharded_state --tensor-parallel-size 16" \
    --output_dir "$OUTPUT_DIR" \
    --mount_paths "${REPO_STRUCTURES_DIR}:/repos/" \
    ++skip_filled=True \
    ++prompt_template="qwen-instruct" \
    ++prompt_config="eval/artsiv/system" \
    ++inference.temperature=0.7 \
    ++inference.tokens_to_generate=81920 \
    ++max_seq_length=262144 \
    ++mount_directory=/repos/ \
    ++multi_turn_key=turns \
    ++total_steps=20 \
    ++truncation_strategy="bookend" \
    ++enable_loop_detection=True \
    ++enable_enhanced_context=True \
    ++context_safety_margin=0.9 \
    ++use_tiktoken=True \
    ++enable_final_turn_prompt=True \
    ++final_turn_instruction_type="aligned" \
    ++final_turn_threshold=1.0 \
    ++enable_response_length_management=True \
    ++max_response_tokens=60000 \
    ++max_thinking_tokens=70000 \
    ++max_final_turn_tokens=40000 \
    ++response_length_retry_limit=20000 \
    ++enable_response_truncation=True \
    ++inject_length_warnings=True \
    ++max_allowed_generation_tokens=75000 \
    ++generation_buffer_tokens=5000 \
    ++max_concurrent_requests=512

echo ""
echo "✅ ARTSIV: Command executed with EXACT configuration from best run!"
echo "🎯 Using new ARTSIV names throughout the pipeline"
