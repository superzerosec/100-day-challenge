# DeepHat with vLLM

## Introduction

Deep Hat is trained on real-world attack patterns, operator-inspired tasks, and a continuous flow of high-signal security data to support deep analysis, exploit exploration, adversary emulation, and multi-step offensive reasoning. Deep Hat works inside Kindo, giving you private execution, long-context understanding, and the ability to automate both offensive and defensive workflows across your environment. It is designed for sensitive, technical work that commercial models cannot support.

# Usage

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
uv run vllm serve DeepHat/DeepHat-V1-7B --gpu-memory-utilization 0.70 --max-model-len 8192
```

```bash
uv run vllm chat --model DeepHat/DeepHat-V1-7B
```