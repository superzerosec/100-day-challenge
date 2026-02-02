# vLLM and Ollama + OpenCode
vLLM is a high-performance library for running large language models locally on CPU or GPU. Ollama is a lightweight, fast, and easy-to-use model server for running large language models locally. OpenCode is an open source agent that helps you write code in your terminal, IDE, or desktop.

# Installation
This section will guide you through installing the core components—vLLM, Ollama, and OpenCode—on your local system. 

- **vLLM**: High-performance local backend for running large language models with GPU acceleration and an OpenAI-compatible API.
- **Ollama**: User-friendly tool to run open-source language models locally.
- **OpenCode**: A developer-focused agent that connects to vLLM or Ollama, bringing AI coding assistance into your terminal, IDE, or desktop tools.  

Below you'll find step-by-step installation instructions for each component, including environment setup commands for vLLM (covering CPU architectures and CUDA versions), Homebrew-based installation for Ollama and OpenCode, and a simple install script for OpenCode on Linux.  

By following these instructions, you'll prepare your environment to serve and interact with LLMs locally using OpenCode, leveraging either vLLM or Ollama as your model backend.

## Install vLLM
```bash
uv add pip
# Install vLLM with a specific CUDA version (e.g., 13.0).
export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')
export CUDA_VERSION=130 # or other
export CPU_ARCH=$(uname -m) # x86_64 or aarch64
uv run python -m pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu${CUDA_VERSION}-cp38-abi3-manylinux_2_35_${CPU_ARCH}.whl --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}
```

## Install Ollama
```bash
brew install ollama
```
## Install OpenCode
The easiest way to install OpenCode is through the install script.

### Using curl on linux
```bash
curl -fsSL https://opencode.ai/install | bash
```

### Using Homebrew on macOS
```bash
brew install anomalyco/tap/opencode
```

# Usage
This section provides a quick overview and practical instructions for using vLLM, Ollama, and OpenCode together for local large language model serving and agent experiences.

- **vLLM**: a fast inference library for hosting models on your local GPU or CPU with an OpenAI-compatible API.
- **Ollama**: an easy-to-use runtime to run popular open source language models locally.
- **OpenCode**: a developer agent (by Anomaly) that helps you interact with LLMs in your terminal or tools, and works with both Ollama and vLLM as local backends.

The guides below show you:
- How to install each tool.
- How to start OpenCode in the terminal.
- How to configure OpenCode to use models from Ollama or vLLM by editing a simple JSON settings file.
- How to launch models with vLLM for local, accelerated inference.

With this setup, you can interact with advanced LLMs directly through the OpenCode agent using either Ollama or vLLM as the language model backend—all running locally on your machine.

## Terminal Window
```bash
opencode
```

## Models Configuration
```bash
/models
```

# Configuration Ollama for OpenCode
Get list of models from Ollama.
```bash
ollama list
ollama run llama3.3:latest
```
Edit the `~/.config/opencode/opencode.json` file to add your preferred models.
```bash
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama (local)",
      "options": {
        "baseURL": "http://localhost:11434/v1"
      },
      "models": {
        "llama3.3:latest": {
          "name": "llama3.3:latest"
        }
      }
    }
  }
}
```
# Configuration vLLM for OpenCode
This section provides guidance on how to configure OpenCode to use vLLM as a backend model provider. 
It explains how to point OpenCode to your locally running vLLM server by editing the `~/.config/opencode/opencode.json` file.  

This allows you to interact with models served by vLLM (such as Gemma or Qwen) via OpenCode using an OpenAI-compatible API endpoint, enabling advanced customization and efficient inference on your local hardware.

## Serve the Google Gemma 3 1B IT Model
Serve the Google Gemma 3 1B IT model on port 9000.
```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
uv run vllm serve google/gemma-3-1b-it \
  --port 9000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 20000 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code
```

## Serve the Qwen 3 4B Model
Serve the Qwen 3 4B model on port 9001.
```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
uv run vllm serve Qwen/Qwen3-4B \
  --port 9001 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 20000 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --attention-backend FLASHINFER \
  --trust-remote-code
```
Edit the `~/.config/opencode/opencode.json` file to add your preferred models.
```bash
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "vllm-gemma": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://localhost:9000/v1" },
      "models": {
        "google/gemma-3-1b-it": { "limit": { "context": 20000, "output": 4096 } }
      }
    },
    "vllm-qwen": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://localhost:9001/v1" },
      "models": {
        "Qwen/Qwen3-4B": { "limit": { "context": 20000, "output": 4096 } }
      }
    }
  }
}
```
## Run the OpenCode Agent
Run the OpenCode agent in the terminal.
```bash
opencode -m vllm-gemma/google/gemma-3-1b-it
```
or
```bash
opencode -m vllm-qwen/Qwen/Qwen3-4B
```
or in `opencode` terminal window, select the model from the list.
```bash
/models
```
# Reference
[OpenCode](https://opencode.ai/)
