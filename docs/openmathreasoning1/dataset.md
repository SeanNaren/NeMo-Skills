# Dataset construction

OpenMathReasoning-1 dataset consists of mathematical problems collected from [AoPS community forums](https://artofproblemsolving.com/community). Below we describe the pipeline used to create this dataset. All relevant scripts are available in
[recipes/openmathreasoning](https://github.com/NVIDIA/NeMo-Skills/tree/main/recipes/openmathreasoning) folder.

If you don't have a slurm cluster with a large number of GPUs,
you can still try out all the steps of our pipeline by using [Nvidia NIM models](https://build.nvidia.com/). We include
a 10-sample subset of the raw data in [configs/example-data.txt](https://github.com/NVIDIA/NeMo-Skills/tree/main/recipes/openmathreasoning/configs/example-data.txt) and you can
switch to that data and NIM models by adding `--mode demo` to all the pipeline commands. We also use different models
in this "demo" mode to make it faster, but you can change [configs/demo.yaml](https://github.com/NVIDIA/NeMo-Skills/tree/main/recipes/openmathreasoning/configs/demo.yaml) to pick
any other models supported in https://build.nvidia.com. Make sure to define `NVIDIA_API_KEY` environment variable for this to work
(and ignore scraping and model preparation steps as they are not needed when using NIM models).

Finally, please make sure to go through the
[getting started documentation](../basics/index.md) to make sure you understand how the below commands
work and avoid running into errors.


## Data scraping

There is a great open-source [AoPS-Instruct repository](https://github.com/dsl-lab/aops) where you can find the scripts to scrape
the data. There is also a [DeepStudentLlama/AoPS-Instruct HF dataset](https://huggingface.co/datasets/DeepStudentLlama/AoPS-Instruct) where the raw forum data can be found.
While we didn't use that repository/dataset in our work directly, it should produce a similar output to our internal scripts.

To download and preprocess raw data you can run

```bash
python scripts/prepare_raw_data.py
```

This script will rename certain columns in the original dataset to align with our scripts, combine forum discussions into
a single string, remove quotes and truncate the discussions that are longer than 24000 tokens. The prepared data will be
saved as `raw_aops_data.jsonl`.

The output file should have ~550k rows, so all of the following commands will take a very long time and require a big
number of GPUs if you want to run them on full data. If you just want to try out the full pipeline, we recommend to subsample
the dataset by e.g. running

```bash
mv raw_aops_data.jsonl raw_aops_data_full.jsonl
head -n 1000 raw_aops_data_full.jsonl > raw_aops_data.jsonl
```

## Model conversion

Here are the steps to download/convert all models that we used to create this dataset.

Download the models by running this on cluster from the path that is mounted as `/hf_models` in your cluster config.
```
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen2.5-32B-Instruct --local-dir Qwen2.5-32B-Instruct
huggingface-cli download Qwen/QwQ-32B --local-dir QwQ-32B
huggingface-cli download deepseek-ai/DeepSeek-R1 --local-dir DeepSeek-R1
```

Convert the model to TensorRT-LLM with the following (you can skip this and make corresponding changes in the pipeline
scripts to run with `server_type=vllm` or `server_type=sglang`)

```bash
ns convert \
   --cluster=slurm \
   --input_model=/hf_models/Qwen2.5-32B-Instruct \
   --output_model=/trt_models/qwen2.5-32b-instruct \
   --convert_from=hf \
   --convert_to=trtllm \
   --num_gpus=8 \
   --model_type=qwen \
   --hf_model_name=Qwen/Qwen2.5-32B-Instruct \
   --max_input_len 24580 \
   --max_seq_len 32768

ns convert \
   --cluster=slurm \
   --input_model=/hf_models/QwQ-32B \
   --output_model=/trt_models/qwq-32b \
   --convert_from=hf \
   --convert_to=trtllm \
   --num_gpus=8 \
   --model_type=qwen \
   --hf_model_name=Qwen/QwQ-32B \
   --max_input_len 24580 \
   --max_seq_len 32768
```

At the time of our experiments serving DeepSeek-R1 model with sglang was faster than with TensorRT-LLM, so
we do not convert that model, but instead prepare a sharded checkpoint that is much faster to load.

```python
from nemo_skills.pipeline.cli import run_cmd, wrap_arguments
from nemo_skills.pipeline.utils import get_ray_server_cmd

cmd = get_ray_server_cmd(
    "python3 nemo_skills/conversion/save_sharded_state.py "
    "    --model=/hf_models/DeepSeek-R1 "
    "    --output=/hf_models/DeepSeek-R1-tp16 "
    "    --tensor-parallel-size=16 "
    "    --max_model_len=8192 "
    "    --trust-remote-code "
    "    --enforce-eager "
)

run_cmd(
    ctx=wrap_arguments(cmd),
    cluster="slurm",
    num_gpus=8,
    num_nodes=2,
    # we are using vllm's script to shard but the model can be served with sglang
    container="vllm",
    log_dir="/hf_models/DeepSeek-R1-tp16",
)
```

## Problem generation pipeline

[Problem generation pipeline](https://github.com/NVIDIA/NeMo-Skills/tree/main/recipes/openmathreasoning/pipelines/problem_generation.py)
consists of the following stages:

1. [Extract all problems](https://github.com/NVIDIA/NeMo-Skills/tree/main/recipes/openmathreasoning/prompts/extract-problems.yaml)
   from the first forum post (`extract_problems` stage).
2. Classify whether each problem belongs to one of the following categories:
   [proof question](https://github.com/NVIDIA/NeMo-Skills/tree/main/recipes/openmathreasoning/prompts/classify-if-proof.yaml),
   [binary question](https://github.com/NVIDIA/NeMo-Skills/tree/main/recipes/openmathreasoning/prompts/classify-if-binary.yaml),
   [multiple-choice question](https://github.com/NVIDIA/NeMo-Skills/tree/main/recipes/openmathreasoning/prompts/classify-if-mcq.yaml),
   [invalid question](https://github.com/NVIDIA/NeMo-Skills/tree/main/recipes/openmathreasoning/prompts/classify-if-invalid.yaml)
   (`classify_problems` stage).
3. [Extract answers](https://github.com/NVIDIA/NeMo-Skills/tree/main/recipes/openmathreasoning/prompts/extract-answers.yaml)
   from the forum discussions (`extract_answers` stage).
4. [Convert proof questions](https://github.com/NVIDIA/NeMo-Skills/tree/main/recipes/openmathreasoning/prompts/convert-proofs.yaml)
   to answer questions (`convert_proofs` stage).
5. Remove all binary/multiple-choice/invalid problems and merge remaining problems with converted proofs (`merge_data` stage).
6. [Decontaminate](../pipelines/decontamination.md) the resulting questions with popular math benchmarks (`decontaminate` stage).

You can run the full pipeline with

```
python recipes/openmathreasoning/pipelines/problem_generation.py
```

You can specify a subset of stages using `--stages` argument, e.g. `--stages extract_problems` or `--stages classify_problems,extract_answers`.

If you want to run using [Nvidia NIM models](https://build.nvidia.com/models) on 10 example questions, add `--mode demo`.


## CoT solution generation pipeline

[Solution generation pipeline](https://github.com/NVIDIA/NeMo-Skills/tree/main/recipes/openmathreasoning/pipelines/solution_generation.py)
consists of the following stages:

1. [Generate solutions](../pipelines/generation.md) for each of the prepared problems (`generate_solutions` stage).
2. [Fill majority answer](https://github.com/NVIDIA/NeMo-Skills/tree/main/nemo_skills/evaluation/aggregate_answers.py)
   for all problems where ground-truth answer is not known (`fill_majority_answer` stage).
3. [Judge answers using an LLM](../pipelines/llm-as-a-judge.md). Only the final answer is compared to the ground-truth (or majority)
   answer, not the full solution (`judge_answers` stage).
4. TODO: generate a new summary
5. Filter out all incorrect solutions and prepare the data for SFT (`prepare_for_sft` stage).


You can run the full pipeline using [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) as solution generation model with

```
python recipes/openmathreasoning/pipelines/solution_generation.py --mode full-qwq
```

You can specify a subset of stages using `--stages` argument and can switch between QwQ and R1 models using `--mode full-qwq` or `--mode full-r1`.

If you want to run using [Nvidia NIM models](https://build.nvidia.com/models) on 10 example questions, add `--mode demo`.

## TIR solution generation pipeline

Coming soon!

## GenSelect pipeline

Coming soon!
