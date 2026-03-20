import os

from nemo_skills.pipeline.cli import wrap_arguments
from nemo_skills.pipeline.eval import SingleNodeMode, eval

os.environ["NVIDIA_API_KEY"] = "sk-fX8ok2ChQOgI1SNqekeY3Q"

model_name = "azure/anthropic/claude-opus-4-6"
dataset = "icpc25"
solutions = 24
steps = 10
num_runs = 1
start_run = 0

# Base directory for all runs
output_base = f"/workspace/baselines/self-correct/icpc25/{model_name.split('/')[-1]}_self_correct_with_memory_{solutions}_2_solutions_all_{steps}_steps_pt2"
# Directory containing evaluation datasets
data_dir = "/workspace/data/eval_datasets/"

cluster = "cw-dfw"
test_file = f"{data_dir}/{dataset}/test_metadata.json"

for i in range(start_run, start_run + num_runs):
    print(f"Running {i}th experiment")
    run_output_dir = f"{output_base}/runs/{i + 1}"
    eval(
        ctx=wrap_arguments(
            "++skip_filled=True "
            "++prompt_config=generic/default "
            "++inference.endpoint_type=chat "
            "++inference.tokens_to_generate=128000 "
            "++inference.top_k=-1 "
            "++inference.temperature=0.0 "
            "++time_limit=03:50:00 "
            f"++total_steps={steps} "
            "++max_concurrent_requests=2 "
            "++inference.reasoning_effort=max "
            "++show_k_solutions=2 "
            "++only_sample_tests=True "
            "++per_step_evaluate=True "
            f"++eval_config.test_file={test_file} "
        ),
        cluster=cluster,
        with_sandbox=True,
        expname=f"{dataset}_eval_run_module_{i}",
        generation_module="nemo_skills.inference.eval.self_correct_with_memory",
        model=model_name,
        server_type="anthropic",
        server_address="https://inference-api.nvidia.com",
        single_node_mode=SingleNodeMode.sequential,
        dependent_jobs=10,
        benchmarks=f"{dataset}:{solutions}",
        data_dir=data_dir,
        output_dir=run_output_dir,
        installation_command="pip install anthropic -U",
        partition="cpu_short",
        split="test",
    )
