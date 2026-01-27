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
## Inference using vLLM
Serve the model.
```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
uv run vllm serve Qwen/Qwen3-1.7B --gpu-memory-utilization 0.70 --max-model-len 8192 --attention-backend FLASHINFER --dtype auto --api-key token-abc123
```
Inference using vLLM in a Python script `vLLM_Inference.py`.
```python
from openai import OpenAI

def main():
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
    )

    completion = client.chat.completions.create(
        model="Qwen/Qwen3-1.7B",
        messages = [
            {"role": "system", "content": "You are Llama, an AI assistant created by Meta."},
            {"role": "user", "content": [{"type": "text", "text": "which one is bigger, 9.11 or 9.9? think carefully."}]},
        ],
    )

    print(completion.choices[0].message.content)


if __name__ == "__main__":
    main()
```
Run the script.
```bash
uv run python vLLM_Inference.py
```
## CPU Offloading
The following commands demonstrate how to run a large language model (DeepHat/DeepHat-V1-7B) on a machine that may have limited GPU memory by offloading part of the model to the CPU. The `--cpu-offload-gb 20` option specifies that up to 20GB of system RAM will be used for model weights that cannot fit into GPU memory, and `--max-model-len 8192` sets the maximum sequence length for your inputs and outputs.  
Start the vLLM server with CPU offloading.
```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
uv run vllm serve DeepHat/DeepHat-V1-7B --max-model-len 8192 --cpu-offload-gb 20 
```
Once the server is up, you can quickly interact with the model using the `vllm chat` interface. The example below asks the model for instructions on checking for malware detections within Elastic Security.
```bash
uv run vllm chat --model DeepHat/DeepHat-V1-7B --quick "How to check malware detection in Elastic Security?"
```
## Quantization
This section demonstrates how to run a quantized version of the Qwen3-1.7B model with vLLM, specifically using FP8 quantization for faster and more memory-efficient inference on supported hardware. It includes both serving and interactive chat commands.  
Serve the FP8 Quantized Model.  
First, set the environment variables to specify which GPUs to use, then launch the vLLM server with FP8 quantization enabled:
```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
uv run vllm serve Qwen/Qwen3-1.7B \
    --gpu-memory-utilization 0.70 \
    --max-model-len 8192 \
    --attention-backend FLASHINFER \
    --quantization fp8
```
- `--quantization fp8` tells vLLM to load the model weights in FP8 format, greatly decreasing memory requirements and potentially increasing inference speed, depending on your hardware.
- Adjust `--gpu-memory-utilization` and `--max-model-len` as needed for your workload and GPU size.

Chat with the Quantized Model.  
Once the server is running, you can send a prompt to the model using the `vllm chat` tool.
```bash
uv run vllm chat --model Qwen/Qwen3-1.7B --quick "Which one is bigger, 9.11 or 9.9? think carefully."
```
This command quickly sends the prompt to your running quantized model and outputs the response.
# Reference
[vLLM](https://github.com/vllm-project/vllm)  
[vLLM Documentation](https://docs.vllm.ai/en/latest/)  
[Data Parallel Deployment](https://github.com/vllm-project/vllm/blob/main/docs/serving/data_parallel_deployment.md)  
[Quantization](https://docs.vllm.ai/en/latest/features/quantization/)  