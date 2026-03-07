from nemo_skills.pipeline.cli import wrap_arguments
from nemo_skills.pipeline.eval import eval

model_name = "Kimi-K2.5"
server_nodes = 2
cluster = "lax"
data_dir = "/workspace/data/eval_datasets/"
output_dir = f"/workspace/generations/comparison/icpc25/kimi_swarm_reflection_sample_only_10steps/"
gpus = 8

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
        "++max_steps=10 "
        "++max_subagent_steps=10 "
        "++sample_only=True "
        "++max_time=03:45:00 "
        "++eval_config.test_file=/workspace/data/eval_datasets/icpc25/test_metadata.json "
    ),
    cluster=cluster,
    with_sandbox=True,
    expname="kimi_swarm_reflection_sample_only_icpc25",
    generation_module="nemo_skills.inference.eval.swarm_with_reflection",
    model=f"/hf_models/{model_name}",
    server_type='sglang',
    server_gpus=gpus,
    server_nodes=server_nodes,
    benchmarks="icpc25:50",
    data_dir=data_dir,
    output_dir=f"{output_dir}/",
    server_args=f"--tp {gpus * server_nodes} --max-total-tokens 262144 --tool-call-parser kimi_k2 --reasoning-parser kimi_k2",
    split="test",
)
