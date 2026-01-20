# vLLM and Ollama + OpenCode
vLLM is a high-performance library for running large language models locally on CPU or GPU. Ollama is a lightweight, fast, and easy-to-use model server for running large language models locally. OpenCode is an open source agent that helps you write code in your terminal, IDE, or desktop.

# Install
## vLLM
```bash
uv add pip
# Install vLLM with a specific CUDA version (e.g., 13.0).
export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')
export CUDA_VERSION=130 # or other
export CPU_ARCH=$(uname -m) # x86_64 or aarch64
uv run python -m pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu${CUDA_VERSION}-cp38-abi3-manylinux_2_35_${CPU_ARCH}.whl --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}
```

## Ollama
```bash
brew install ollama
```
## OpenCode
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
## Terminal window
```bash
opencode
```

## Models config
```bash
/models
```

### Configuration Ollama
Get list of models from Ollama
```bash
ollama list
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

# Reference
[OpenCode](https://opencode.ai/)
