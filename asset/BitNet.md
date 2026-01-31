# BitNet
bitnet.cpp is the official inference framework for 1-bit LLMs (e.g., BitNet b1.58). It offers a suite of optimized kernels, that support fast and lossless inference of 1.58-bit models on CPU and GPU (NPU support will coming next).

## Installation
```bash
git clone --recursive https://github.com/microsoft/BitNet.git
cd BitNet
uv add pip
uv run python -m pip install -r requirements.txt
uv run hf download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T
uv run setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s
```
# Usage

Run inference with the quantized model.
```bash
uv run python run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -p "You are a helpful assistant" -cnv
```
# Reference
[BitNet](https://github.com/microsoft/BitNet)