# vLLM Cheatsheet
## Install
```bash
uv add pip
uv run python -m pip install vllm
```
# Usage
Run the model in the terminal window in this sequence:
1. Serve a model
2. Chat with the model
## Serve a Model
```bash
uv run vllm serve Qwen/Qwen3-1.7B
```
## Chat
```bash
uv run vllm chat --model Qwen/Qwen3-1.7B
```
## Serve Model on Specific GPU
Checking the GPU device order
```bash
nvidia-smi -L
```
Set the GPU device order and serve the model
```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
uv run vllm serve Qwen/Qwen3-1.7B --gpu-memory-utilization 0.70 --max-model-len 8192
```
Then chat with the model
```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
uv run vllm chat --model Qwen/Qwen3-1.7B
```
## Test Chat with the Model
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

This will run DP=2, TP=2 on a single 4-GPU node:
```bash
uv run vllm serve Qwen/Qwen3-1.7B --gpu-memory-utilization 0.70 --max-model-len 8192 --data-parallel-size 2 --tensor-parallel-size 2
```
Start chat with the model
```bash
uv run vllm chat --model Qwen/Qwen3-1.7B --quick "Which one is bigger, 9.11 or 9.9? think carefully."
```
# Reference
[vLLM](https://github.com/vllm-project/vllm)  
[vLLM Documentation](https://docs.vllm.ai/en/latest/)  
[Data Parallel Deployment](https://github.com/vllm-project/vllm/blob/main/docs/serving/data_parallel_deployment.md)