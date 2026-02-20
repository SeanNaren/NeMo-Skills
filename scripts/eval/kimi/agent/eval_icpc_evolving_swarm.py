from nemo_skills.pipeline.cli import wrap_arguments
from nemo_skills.pipeline.eval import eval

model_name = "Kimi-K2.5"
server_nodes = 2
data_dir = "/workspace/data/eval_datasets/"

# cw-dfw, hsg, lax
cluster = "hsg"

output_dir = f"/workspace/baselines/icpc25/agent/{model_name}/evolving_swarm_v1/"
gpus = 4 if cluster == "hsg" else 8
eval(
    ctx=wrap_arguments(
        "++skip_filled=True "
        "++inference.temperature=0.0 "
        "++inference.tokens_to_generate=220000 "
        "++inference_subagent.temperature=1.0 "
        "++inference_subagent.top_p=0.95 "
        "++inference_subagent.tokens_to_generate=220000 "
        "++max_concurrent_requests=12 "
        "++inference.endpoint_type=chat "
        "++inference_subagent.endpoint_type=chat "
        "++max_steps=100 "
        "++max_subagent_steps=50 "
        "++max_time=03:45:00 "
        "++eval_config.test_file=/workspace/data/eval_datasets/icpc25/test_metadata.json "
    ),
    cluster=cluster,
    with_sandbox=True,
    expname="evolving_swarm_kimi_agent_icpc25",
    generation_module="nemo_skills.inference.eval.evolving_swarm",
    model=f"/hf_models/{model_name}",
    server_type='sglang',
    server_gpus=gpus,
    server_nodes=server_nodes,
    dependent_jobs=5,
    benchmarks="icpc25:0",
    data_dir=data_dir,
    output_dir=f"{output_dir}/",
    server_args=f"--tp {gpus * server_nodes} --max-total-tokens 262144 --tool-call-parser kimi_k2 --reasoning-parser kimi_k2",
)
