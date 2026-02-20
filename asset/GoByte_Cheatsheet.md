# GoByte
GoByte is a fast PCAP/PCAPNG parser built in Go for preprocessing network traffic data for deep learning models. It extracts packet bytes and exports them in multiple formats (CSV, Parquet, or NumPy).

## Installation
Linux/macOS Setup.
```bash
# Download the binary (replace with your machine platform)
wget https://github.com/afifhaziq/GoByte/releases/latest/download/gobyte-linux-amd64

# Make it executable
chmod +x gobyte-linux-amd64

# Move to PATH (optional)
sudo mv gobyte-linux-amd64 /usr/local/bin/gobyte

# Verify installation
gobyte --help
```

# Usage

Download the MalayaNetwork_GT dataset (13.5 GB, 10 traffic classes).
```bash
uvx hf download Afifhaziq/MalayaNetwork_GT \
    --repo-type dataset \
    --local-dir ./dataset \
    --include "PCAP/*"
```
Process the entire dataset with GoByte (streaming mode for large datasets).
```bash
gobyte --dataset dataset/PCAP --format numpy --length 1500 --streaming --output dataset.npy
```

# Reference
[GoByte](https://github.com/afifhaziq/GoByte)