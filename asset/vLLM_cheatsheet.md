# vLLM Cheatsheet
vLLM is a high-performance library for running large language models locally on CPU or GPU. It is a fast, flexible, and easy-to-use library for running large language models locally.
## Install
To get started with vLLM, first install the package and its dependencies. Follow the steps below to quickly set up vLLM in environment.

```bash
uv add pip
uv run python -m pip install vllm
```
# Usage
Run the model in the terminal window in this sequence:
1. Serve a model
2. Chat with the model
## Serve a Model
Serve the model in the terminal window.
```bash
uv run vllm serve Qwen/Qwen3-1.7B
```
## Chat
Chat with the model in the another terminal window.
```bash
uv run vllm chat --model Qwen/Qwen3-1.7B
```
## Serve Model on Specific GPU
Check the GPU device order.
```bash
nvidia-smi -L
```
Set the GPU device order and serve the model on specific GPUs.
```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
uv run vllm serve Qwen/Qwen3-1.7B --gpu-memory-utilization 0.70 --max-model-len 8192
```
Then chat with the model on specific GPUs.
```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
uv run vllm chat --model Qwen/Qwen3-1.7B
```
## Test Chat with the Model
Test chat with the model in the terminal window.
```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dev" \
  -d '{
    "model": "Qwen/Qwen3-1.7B",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "hi"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

Test chat with the model in the terminal window.
```bash
 curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dev" \
  -d '{
    "model": "Qwen/Qwen3-1.7B",
    "messages": [                                             
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Which one is bigger, 9.11 or 9.9? think carefully."}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }' | jq -r '.choices[].message.content'
```
# Advanced Usage
## Serve Multiple Models
Each model should be served on a different port.  

For example, to serve meta-llama/Llama-3.2-1B-Instruct on port 9001 and Qwen/Qwen3-7B on port 9002.  

Serve meta-llama/Llama-3.2-1B-Instruct on port 9001 and using GPU 0.  
```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
uv run vllm serve meta-llama/Llama-3.2-1B-Instruct --gpu-memory-utilization 0.70 --port 9001
```
Serve Qwen/Qwen3-1.7B on port 9002 and using GPU 1.
```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
uv run vllm serve Qwen/Qwen3-1.7B --gpu-memory-utilization 0.70 --port 9002
```
Then chat with the models on specific URLs.  

Start chat with meta-llama/Llama-3.2-1B-Instruct on port 9001 and using GPU 0.
```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
uv run vllm chat --model meta-llama/Llama-3.2-1B-Instruct --url http://localhost:9001/v1
```
Start chat with Qwen/Qwen3-1.7B on port 9002 and using GPU 1.
```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
uv run vllm chat --model Qwen/Qwen3-1.7B --url http://localhost:9002/v1
```
## Data Parallel Deployment
vLLM supports Data Parallel deployment, where model weights are replicated across separate instances/GPUs to process independent batches of requests. This will work with both dense and MoE models.  

For MoE models, particularly those like DeepSeek that employ MLA (Multi-head Latent Attention), it can be advantageous to use data parallel for the attention layers and expert or tensor parallel (EP or TP) for the expert layers.  

This will run DP=2, TP=2 on a single 4-GPU node.
```bash
uv run vllm serve Qwen/Qwen3-1.7B --gpu-memory-utilization 0.70 --max-model-len 8192 --data-parallel-size 2 --tensor-parallel-size 2
```
Start chat with the model.
```bash
uv run vllm chat --model Qwen/Qwen3-1.7B --quick "Which one is bigger, 9.11 or 9.9? think carefully."
```
## Inference with vLLM
Inference with vLLM in a Python script `inference_with_vllm.py`.
```python
import os
import torch
from vllm import LLM, SamplingParams

def main() -> None:
    # Configure multi-GPU: use tensor parallelism to split model across GPUs
    num_gpus = torch.cuda.device_count() or 1
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "Which one is bigger, 9.11 or 9.9? think carefully."
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    llm = LLM(
        model="Qwen/Qwen3-1.7B",
        tensor_parallel_size=min(num_gpus, 2),
        data_parallel_size=1,
        gpu_memory_utilization=0.70,
        max_model_len=8192,
    )

    # Use chat-style API: each prompt becomes a single-user-message conversation
    conversations = [
        [{"role": "You are a helpful assistant.", "content": prompt}]
        for prompt in prompts
    ]

    outputs = llm.chat(messages=conversations, sampling_params=sampling_params, use_tqdm=False)
    for prompt, output in zip(prompts, outputs):
        print(f"Prompt: {prompt!r}, Generated text: {output.outputs[0].text!r}")


if __name__ == "__main__":
    main()
```
Run the script.
```bash
uv run python inference_with_vllm.py
```
# Reference
[vLLM](https://github.com/vllm-project/vllm)  
[vLLM Documentation](https://docs.vllm.ai/en/latest/)  
[Data Parallel Deployment](https://github.com/vllm-project/vllm/blob/main/docs/serving/data_parallel_deployment.md)