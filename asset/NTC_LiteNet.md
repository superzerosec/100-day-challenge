# LiteNet
LiteNet provides a complete, end-to-end pipeline for training, optimizing, and deploying a neural network for Network Traffic Classification (NTC) model. The pipeline includes feature selection with SHAP, semi-structured sparse pruning, quantization (FP16/INT8), and conversion to a TensorRT engine for high-performance inference.

## Installation
```bash
git clone https://github.com/afifhaziq/LiteNet.git
cd LiteNet
uv sync
```
# Usage
## Download the Dataset
```bash
mkdir -p dataset
uvx hf download Afifhaziq/MalayaNetwork_GT \
    --repo-type dataset \
    --local-dir ./dataset \
    --include "PCAP/*"
mv dataset/PCAP/ dataset/Malaya_GT
```
## Process the Dataset
```bash
# Download GoByte
wget https://github.com/afifhaziq/GoByte/releases/latest/download/gobyte-linux-amd64
chmod +x gobyte-linux-amd64
sudo mv gobyte-linux-amd64 /usr/local/bin/gobyte
# Process the Dataset
gobyte --dataset dataset/Malaya_GT --format numpy --length 1500 --streaming --output malaya_gt.npy
```
## Train the Model
```bash
uv run python main.py
```
# Reference
[LiteNet](https://github.com/afifhaziq/LiteNet)