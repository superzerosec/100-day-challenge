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
## Preprocess
Download GoByte and process the dataset using GoByte.
```bash
# Download GoByte
wget https://github.com/afifhaziq/GoByte/releases/latest/download/gobyte-linux-amd64
chmod +x gobyte-linux-amd64
sudo mv gobyte-linux-amd64 /usr/local/bin/gobyte
```
Process the dataset.
```bash
gobyte --dataset dataset/Malaya_GT --format csv --streaming --length 50 --output malaya_50.csv
```
### Importing Libraries and Loading the Dataset
```python
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader, Dataset

dataset = pd.read_csv("output/malaya_50.csv")
```
### Randomising Stratified Sampling
```python
indices = np.arange(len(dataset))
labels = dataset["Class"] # Use the "Class" column from the dataframe for stratification

train_indices, test_indices = train_test_split(
    indices, 
    test_size=0.2, 
    stratify=labels, 
    random_state=42
)
```
### Splitting
Split the dataset into 80:20 ratio for training and testing.
```python
class DataFrameDataset(Dataset):
    """Dataset that indexes DataFrame by row and returns tensors for DataLoader collate."""
    def __init__(self, df, feature_cols=None, label_col="Class"):
        self.df = df
        self.label_col = label_col
        self.feature_cols = feature_cols or [c for c in df.columns if c != label_col]
        self.classes = sorted(df[label_col].unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.tensor(row[self.feature_cols].values.astype(np.float32))
        y = self.class_to_idx[row[self.label_col]]
        return {"x": x, "y": y}


# Convert the numpy array of indices to a plain list of ints
train_indices = train_indices.astype(int).tolist()
test_indices = test_indices.astype(int).tolist()

# Wrap DataFrame in a Dataset (returns tensors), then apply Subset
base_dataset = DataFrameDataset(dataset)
train_dataset = Subset(base_dataset, train_indices)
test_dataset = Subset(base_dataset, test_indices)

# Loaders now batch correctly (default_collate stacks the dict of tensors)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

# Usage
## Train the Model
```bash
uv run python main.py
```
# Reference
[LiteNet](https://github.com/afifhaziq/LiteNet)