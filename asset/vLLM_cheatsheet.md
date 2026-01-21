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
# Reference
[vLLM](https://github.com/vllm-project/vllm)  
[vLLM Documentation](https://docs.vllm.ai/en/latest/)