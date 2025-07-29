# How to build all necessary dockerfiles

Some dockerfiles are directly included in this folder and for some others the instructions to build them are below.
To build one of the existing dockerfiles use a command like this

```
docker build -t igitman/nemo-skills-nemo:0.6.1 -f dockerfiles/Dockerfile.nemo .
```
It might take a long time for some of the images.

## Building trtllm image

Follow instructions in [TensorRT-LLM documentation](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html#option-1-build-tensorrt-llm-in-one-step).

Our current container is built from `v0.19.0` code version.
