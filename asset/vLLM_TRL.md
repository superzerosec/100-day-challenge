# vLLM and TRL Integration

This document will guide you through the process of using vLLM with TRL for faster generation in online methods like GRPO and Online DPO. We first summarize a tl;dr on how to use vLLM with TRL, and then we will go into the details of how it works under the hood.

Transformer Reinforcement Learning (TRL) is a full stack library where we provide a set of tools to train transformer language models with methods like Supervised Fine-Tuning (SFT), Group Relative Policy Optimization (GRPO), Direct Preference Optimization (DPO), Reward Modeling, and more. The library is integrated with 🤗 transformers.

## Installation

```bash
uv add "trl[vllm]"
```

# Usage
Then run the server on specific GPUs.
```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
uv run trl vllm-serve --model Qwen/Qwen3-1.7B --tensor-parallel-size 1 --data-parallel-size 1
```
Once the server is running, you can use it to generate completions for training. In the example below, we are using the different supported trainers using the vLLM server for generation. The --tensor-parallel-size and --data-parallel-size arguments control how the model and data are sharded across GPUs.

In this example, we are sharding two copies of the model across 4 GPUs. Increasing data parallelism increases throughput, while increasing tensor parallelism allows for serving larger models. Then, run the training script on different GPUs (e.g., GPUs 4-7) by passing use_vllm=True in the training arguments as follows:

Sample of a simple train.py script:

```python
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

trainer = GRPOTrainer(
    model="Qwen/Qwen3-1.7B",
    args=GRPOConfig(use_vllm=True),
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
)

trainer.train()
```
And the train command on separate GPUs from the server:
```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
uv run accelerate launch train.py
```

# Reference

[vLLM TRL](https://huggingface.co/docs/trl/v0.26.2/en/vllm_integration)  
[Slide](https://cas-bridge.xethub.hf.co/xet-bridge-us/6761b247a5ee32556ce733ca/25796df92f85e22fd9b31bf0d558c913638dddd8e2e949035dd5aacb09a891ac?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=cas%2F20260201%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260201T090710Z&X-Amz-Expires=3600&X-Amz-Signature=84ce6e1b0528608da56268b8452d1e6ba895f0ea2c05fd0af6a2baf89144ff7e&X-Amz-SignedHeaders=host&X-Xet-Cas-Uid=624970fe8abe179b008cebf0&response-content-disposition=inline%3B+filename*%3DUTF-8%27%27Fine%252520tuning%252520with%252520TRL%252520%252528Oct%25252025%252529.pdf%3B+filename%3D%22Fine%2520tuning%2520with%2520TRL%2520%2528Oct%252025%2529.pdf%22%3B&response-content-type=application%2Fpdf&x-id=GetObject&Expires=1769940430&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2OTk0MDQzMH19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2FzLWJyaWRnZS54ZXRodWIuaGYuY28veGV0LWJyaWRnZS11cy82NzYxYjI0N2E1ZWUzMjU1NmNlNzMzY2EvMjU3OTZkZjkyZjg1ZTIyZmQ5YjMxYmYwZDU1OGM5MTM2MzhkZGRkOGUyZTk0OTAzNWRkNWFhY2IwOWE4OTFhYyoifV19&Signature=sF5GhTeVcuyyvtfJJjkcHo9cZuYYNlhBH8a9bN7wkrMiK9g5VpobHfPwui76uqfVepjE2-s3pQDnZXKAjbwdbvuJtiCLgkblMTvp2O%7EjH66AIMeEop0SgiLAjcT4UBKBnisOAL0OEtVdrDzi0UhHegfytDAjW6rut7mweh0xCXfw5hWmmdILGzTlIaz8ovW4Hwx0ssjjxpPL9TP-xTcTKPAYZes7vsqbHnpar6pSDfmweOYDbsRbjh3D9YjHukxfMMGu73W83q7z%7E84wOwbwdvRs9c1YPETcKvwuMb-pk2fTO8Hma4zB-GYb4onltECiFfhwszsO1rpjAQb-UbB3Zw__&Key-Pair-Id=K2L8F4GPSG1IFC)