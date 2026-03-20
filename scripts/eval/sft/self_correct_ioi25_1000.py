from nemo_skills.pipeline.cli import wrap_arguments
from nemo_skills.pipeline.eval import eval

model_name = "ccc_800k_ocr3_subsample_585k_qwen3_32b_lr5e_05_ep3_gbs512_lr5e-05_ep3_gbs512"
dataset = "ioi25"
solutions = 1000
steps = 10
num_runs = 1
start_run = 0

# Base directory for all runs
output_base = f"/workspace/baselines/self-correct/ioi25/1000_gen_{model_name}_self_correct_with_memory_{solutions}_2_solutions_all_{steps}_steps"
# Directory containing evaluation datasets
data_dir = "/workspace/data/eval_datasets/"

cluster = "eos"

for i in range(start_run, start_run + num_runs):
    print(f"Running {i}th experiment")
    run_output_dir = f"{output_base}/runs/{i+1}"
    eval(
        ctx=wrap_arguments(
            "++skip_filled=True "
            "++prompt_config=generic/default "
            "++inference.temperature=1.0 "
            "++inference.top_p=1.0 "
            "++inference.top_k=-1 "
            "++inference.min_p=0 "
            "++inference.repetition_penalty=1.0 "
            "++inference.tokens_to_generate=120000 " 
            "++chat_template_kwargs.thinking=true "
            "++time_limit=03:50:00 "
            "++save_every_step=False "
            f"++total_steps={steps} "
            "++max_concurrent_requests=512 "
            "++show_k_solutions=2 "
            "++only_sample_tests=False "
            "++per_step_evaluate=False "
            f"++eval_config.test_file=/workspace/data/eval_datasets/{dataset}/test_metadata.json "
        ),
        cluster=cluster,
        with_sandbox=True,
        expname=f"self_correct",
        generation_module="nemo_skills.inference.eval.self_correct_with_memory",
        model=f"/msamadi/hf_models/{model_name}",
        server_type="vllm",
        server_gpus=4 if cluster == "hsg" else 8,
        server_nodes=1,
        num_jobs=10,
        dependent_jobs=20,
        benchmarks=f"{dataset}:{solutions}",
        data_dir=data_dir,
        output_dir=run_output_dir,
        server_args = "--async-scheduling --max-num-seqs=1024",
        split="test",
    )
