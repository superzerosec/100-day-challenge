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
## Serve a model
```bash
uv run vllm serve Qwen/Qwen3-1.7B
```
## Chat
```bash
uv run vllm chat --model Qwen/Qwen3-1.7B
```
## Serve with specific GPU
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
## To Test the Model
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
For example, to serve Qwen/Qwen3-1.7B on port 9001 and Qwen/Qwen3-7B on port 9002.  
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
Then chat with the models on specific urls.  
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
# Reference
[vLLM](https://github.com/vllm-project/vllm)  
[vLLM Documentation](https://docs.vllm.ai/en/latest/)