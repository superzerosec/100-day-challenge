# MH-Net
Revolutionizing Encrypted Traffic Classification with MH-Net: A Multi-View Heterogeneous Graph Model.
## Installation
```bash
git clone https://github.com/ViktorAxelsen/MH-Net.git
cd MH-Net
uv init
uv add pip
uv add scipy scikit-learn torch torchvision torchaudio dql
```
# Usage
```bash
# ISCX-VPN
python pcap2npy.py --dataset iscx-vpn
# ISCX-NonVPN
python pcap2npy.py --dataset iscx-nonvpn
```

# Reference
[Code](https://github.com/ViktorAxelsen/MH-Net)  
[Paper](https://arxiv.org/abs/2501.03279)