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
# Dataset
Prepare the dataset for training and testing.
## ISCX-VPN
[ISCX-VPN](https://www.unb.ca/cic/datasets/vpn.html) is a dataset consists of VPN and non-VPN traffic.
```bash
mkdir -p dataset/iscx/{vpn,nonvpn}
cd dataset/iscx/
wget --spider -r -l inf -nd -np -R "index.html*" -o /dev/stdout http://cicresearch.ca/CICDataset/ISCX-VPN-NonVPN-2016/Dataset/PCAPs/ | grep '^--' | awk '{print $3}' >> all_urls_vpn_nonvpn.txt
cat all_urls_vpn_nonvpn.txt | grep '.zip' | sort | uniq >> download_urls_vpn_nonvpn.txt
while read url; do axel -n 100 $url; done < download_urls_vpn_nonvpn.txt
find . -name "NonVPN-PCAPs-*.zip" -type f -exec unzip {} -d nonvpn \;
find . -name "VPN-PCAP*.zip" -type f -exec unzip {} -d vpn \;
cd ../../
```

## ISCX-Tor
[ISCX-Tor](https://www.unb.ca/cic/datasets/tor.html) is a dataset consists of Tor and non-Tor traffic.
```bash
mkdir -p dataset/iscx/{tor,nontor}
cd dataset/iscx/
wget --spider -r -l inf -nd -np -R "index.html*" -o /dev/stdout http://cicresearch.ca/CICDataset/ISCX-Tor-NonTor-2017/Dataset/PCAPs/ | grep '^--' | awk '{print $3}' >> all_urls_tor_nontor.txt
cat all_urls_tor_nontor.txt | grep -E '.zip|.xz' | sort | uniq >> download_urls_tor_nontor.txt
while read url; do axel -n 100 $url; done < download_urls_tor_nontor.txt
unzip Tor.zip -d tor
tar -xvf NonTor.tar.xz -C nontor
cd ../../
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