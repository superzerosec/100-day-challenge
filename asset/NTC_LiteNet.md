# LiteNet
LiteNet provides a complete, end-to-end pipeline for training, optimizing, and deploying a neural network for Network Traffic Classification (NTC) model. The pipeline includes feature selection with SHAP, semi-structured sparse pruning, quantization (FP16/INT8), and conversion to a TensorRT engine for high-performance inference.

## Installation
```bash
git clone https://github.com/afifhaziq/LiteNet.git
cd LiteNet
uv sync
```
# Preparing the Dataset
This section, **Preparing the Dataset**, provides instructions for setting up the data required to train and evaluate the LiteNet model. It typically covers the steps needed to obtain, organise, and preprocess network traffic datasets, which may include downloading datasets, converting file formats, or splitting data into training and testing sets. Proper preparation of the dataset ensures that the model receives the correct input and can achieve optimal performance during training and inference.
## MalayaNetwork_GT Dataset
MalayaNetwork_GT is a private network traffic dataset captured from several real network environments and multiple time windows to intentionally introduce non-IID characteristics.
### Download
```bash
mkdir -p dataset
uvx hf download Afifhaziq/MalayaNetwork_GT \
    --repo-type dataset \
    --local-dir ./dataset \
    --include "PCAP/*"
mv dataset/PCAP/ dataset/Malaya_GT
```
### Preprocess
```bash
# Download GoByte
wget https://github.com/afifhaziq/GoByte/releases/latest/download/gobyte-linux-amd64
chmod +x gobyte-linux-amd64
sudo mv gobyte-linux-amd64 /usr/local/bin/gobyte
# Process the Dataset
gobyte --dataset dataset/Malaya_GT --format numpy --streaming --length 50 --output malaya_50
```
### Spliting the Dataset
Split the dataset into 80:10:10 ratio for training, testing, and validation.
```python
import numpy as np
import torch

dataset = np.load("output/malaya_50_data.npy", allow_pickle=True)
dataset = torch.from_numpy(dataset.astype(np.float32))

# Spliting dataset into 80:10:10 ratio
total_len = len(dataset)
train_end = int(total_len * 0.8)
val_end = train_end + int(total_len * 0.1)

train = dataset[:train_end, :]
val = dataset[train_end:val_end, :]
test = dataset[val_end:, :]
total_len = len(train) + len(test) + len(val)

print(train.shape, test.shape, val.shape, total_len)
```
# Usage
## Train the Model
```bash
uv run python main.py
```
# Reference
[LiteNet](https://github.com/afifhaziq/LiteNet)