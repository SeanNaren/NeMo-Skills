from nemo_skills.pipeline.cli import wrap_arguments
from nemo_skills.pipeline.eval import eval

model_names = ["DeepSeek-V3.2", "DeepSeek-V3.2-Speciale"]
server_nodes = 2
cluster = "lax"
data_dir = "/workspace/data/eval_datasets/"
output_dir = "/workspace/generations/comparison/icpc25/ds_reasoning_agent_sample_only_10steps/"
gpus = 8
default_server_args = f"--ep-size {gpus * server_nodes} --dp {gpus * server_nodes} --enable-dp-attention --mem-fraction-static=0.8 --tool-call-parser deepseekv32"

eval(
    ctx=wrap_arguments(
        "++skip_filled=True "
        "++prompt_config=eval/ioi/agent/agent_tools_solver "
        "++inference.temperature=1.0 "
        "++inference.top_p=0.95 "
        "++inference.tokens_to_generate=120000 "
        "++max_concurrent_requests=1024 "
        "++inference.endpoint_type=chat "
        "++chat_template_kwargs.thinking=true "
        "++use_client_parsing=False "
        "++avg_score=True "
        "++max_steps=10 "
        "++sample_only=True "
        "++explicit_feedback=False "
        "++max_time=03:45:00 "
        "++eval_config.test_file=/workspace/data/eval_datasets/icpc25/test_metadata.json "
    ),
    benchmarks="icpc25:50",
    data_dir=data_dir,
    cluster=cluster,
    with_sandbox=True,
    expname="ds_reasoning_agent_sample_only_icpc25",
    generation_module="nemo_skills.inference.eval.reasoning_agent_v2",
    model=[f"/hf_models/{model_name}" for model_name in model_names],
    server_type='sglang',
    server_gpus=gpus,
    server_nodes=server_nodes,
    output_dir=f"{output_dir}/",
    installation_command="pip install bfcl-eval func-timeout -U",
    server_args=default_server_args,
    split="test",
)
