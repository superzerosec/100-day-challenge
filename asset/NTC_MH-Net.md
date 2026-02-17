# MH-Net
Revolutionizing Encrypted Traffic Classification with MH-Net: A Multi-View Heterogeneous Graph Model.
## Installation
```bash
git clone https://github.com/ViktorAxelsen/MH-Net.git
cd MH-Net
uv init
uv add pip
uv add scapy scikit-learn torch torchvision torchaudio dql
sudo apt install axel
```
# Dataset
Prepare the dataset for training and testing.
## ISCX-VPN
### Download the Dataset
[ISCX-VPN](https://www.unb.ca/cic/datasets/vpn.html) is a dataset consists of VPN and non-VPN traffic.
```bash
mkdir -p dataset/iscx/{vpn,nonvpn}/{train_test,train_test_4bit}
mkdir -p model/{iscx_vpn,iscx_nonvpn,iscx_tor,iscx_nontor}
cd dataset/iscx/
wget --spider -r -l inf -nd -np -R "index.html*" -o /dev/stdout http://cicresearch.ca/CICDataset/ISCX-VPN-NonVPN-2016/Dataset/PCAPs/ | grep '^--' | awk '{print $3}' >> all_urls_vpn_nonvpn.txt
cat all_urls_vpn_nonvpn.txt | grep '.zip' | sort | uniq >> download_urls_vpn_nonvpn.txt
while read url; do axel -n 100 $url; done < download_urls_vpn_nonvpn.txt
```

### Extract the Dataset
VPN Dataset.
```bash
find . -name "VPN-PCAP*.zip" -type f -exec unzip {} -d vpn \;
mkdir -p vpn/{Chat,Email,File,P2P,Streaming,VoIP}
find vpn/ -maxdepth 1 -regextype posix-extended -regex ".*(chat).*" -exec mv {} vpn/Chat \;
find vpn/ -maxdepth 1 -regextype posix-extended -regex ".*(email).*" -exec mv {} vpn/Email \;
find vpn/ -maxdepth 1 -regextype posix-extended -regex ".*(ftp|files).*" -exec mv {} vpn/File \;
find vpn/ -maxdepth 1 -regextype posix-extended -regex ".*(bittorrent).*" -exec mv {} vpn/P2P \;
find vpn/ -maxdepth 1 -regextype posix-extended -regex ".*(netflix|spotify|vimeo|youtube).*" -exec mv {} vpn/Streaming \;
find vpn/ -maxdepth 1 -regextype posix-extended -regex ".*(audio|voip).*" -exec mv {} vpn/VoIP \;
```
NonVPN Dataset.
```bash
find . -name "NonVPN-PCAPs-*.zip" -type f -exec unzip {} -d nonvpn \;
mkdir -p nonvpn/{Chat,Email,File,Streaming,Video,VoIP}
find nonvpn/ -maxdepth 1 -regextype posix-extended -regex ".*(chat).*" -exec mv {} nonvpn/Chat \;
find nonvpn/ -maxdepth 1 -regextype posix-extended -regex ".*(email).*" -exec mv {} nonvpn/Email \;
find nonvpn/ -maxdepth 1 -regextype posix-extended -regex ".*(ftp|file|scp).*" -exec mv {} nonvpn/File \;
find nonvpn/ -maxdepth 1 -regextype posix-extended -regex ".*(netflix|spotify|vimeo|youtube).*" -exec mv {} nonvpn/Streaming \;
find nonvpn/ -maxdepth 1 -regextype posix-extended -regex ".*(video).*" -exec mv {} nonvpn/Video \;
find nonvpn/ -maxdepth 1 -regextype posix-extended -regex ".*(audio|voip).*" -exec mv {} nonvpn/VoIP \;
cd ../../
```

## ISCX-Tor
### Download the Dataset
[ISCX-Tor](https://www.unb.ca/cic/datasets/tor.html) is a dataset consists of Tor and non-Tor traffic.
```bash
mkdir -p dataset/iscx/{tor,nontor}/{train_test,4bit}
cd dataset/iscx/
wget --spider -r -l inf -nd -np -R "index.html*" -o /dev/stdout http://cicresearch.ca/CICDataset/ISCX-Tor-NonTor-2017/Dataset/PCAPs/ | grep '^--' | awk '{print $3}' >> all_urls_tor_nontor.txt
cat all_urls_tor_nontor.txt | grep -E '.zip|.xz' | sort | uniq >> download_urls_tor_nontor.txt
while read url; do axel -n 100 $url; done < download_urls_tor_nontor.txt
```

### Extract the Dataset
Tor Dataset.
```bash
unzip Tor.zip -d tor
mv tor/Tor/* tor/
rm -rf tor/Tor
mkdir -p tor/{Audio-Streaming,Browsing,Chat,File,Mail,P2P,Video-Streaming,VoIP}
find tor/ -maxdepth 1 -regextype posix-extended -type f -regex ".*(AUDIO|spotify).*" -exec mv {} tor/Audio-Streaming \;
find tor/ -maxdepth 1 -regextype posix-extended -type f -regex ".*(BROWSING|torGoogle).*" -exec mv {} tor/Browsing \;
find tor/ -maxdepth 1 -regextype posix-extended -type f -regex ".*(CHAT|torTwitter).*" -exec mv {} tor/Chat \;
find tor/ -maxdepth 1 -regextype posix-extended -type f -regex ".*(FILE).*" -exec mv {} tor/File \;
find tor/ -maxdepth 1 -regextype posix-extended -type f -regex ".*(MAIL).*" -exec mv {} tor/Mail \;
find tor/ -maxdepth 1 -regextype posix-extended -type f -regex ".*(P2P|Torrent|p2p).*" -exec mv {} tor/P2P \;
find tor/ -maxdepth 1 -regextype posix-extended -type f -regex ".*(VIDEO|Youtube|Vimeo).*" -exec mv {} tor/Video-Streaming \;
find tor/ -maxdepth 1 -regextype posix-extended -type f -regex ".*(VOIP|torFacebook).*" -exec mv {} tor/VoIP \;
```
NonTor Dataset.
```bash
tar -xvf NonTor.tar.xz -C nontor
mv nontor/NonTor/* nontor/
rm -rf nontor/NonTor
mkdir -p nontor/{Audio-Streaming,Browsing,Chat,File,Mail,P2P,Video-Streaming,VoIP}
find nontor/ -maxdepth 1 -regextype posix-extended -type f -iregex ".*(spotify).*" -exec mv {} nontor/Audio-Streaming \;
find nontor/ -maxdepth 1 -regextype posix-extended -type f -iregex ".*(browsing|ssl).*" -exec mv {} nontor/Browsing \;
find nontor/ -maxdepth 1 -regextype posix-extended -type f -iregex ".*(chat).*" -exec mv {} nontor/Chat \;
find nontor/ -maxdepth 1 -regextype posix-extended -type f -iregex ".*(file|transfer).*" -exec mv {} nontor/File \;
find nontor/ -maxdepth 1 -regextype posix-extended -type f -iregex ".*(imap|pop).*" -exec mv {} nontor/Mail \;
find nontor/ -maxdepth 1 -regextype posix-extended -type f -iregex ".*(p2p).*" -exec mv {} nontor/P2P \;
find nontor/ -maxdepth 1 -regextype posix-extended -type f -iregex ".*(youtube|vimeo).*" -exec mv {} nontor/Video-Streaming \;
find nontor/ -maxdepth 1 -regextype posix-extended -type f -iregex ".*(audio|voice).*" -exec mv {} nontor/VoIP \;
cd ../../
```

# Usage
## Convert PCAP to NPY
```bash
# ISCX-VPN
uv run pcap2npy.py --dataset iscx-vpn
# ISCX-NonVPN
uv run pcap2npy.py --dataset iscx-nonvpn
# ISCX-Tor
uv run pcap2npy.py --dataset iscx-tor
# ISCX-NonTor
uv run pcap2npy.py --dataset iscx-nontor
```

## Preprocess the Dataset
```bash
# ISCX-VPN
uv run preprocess.py --dataset iscx-vpn
# ISCX-NonVPN
uv run preprocess.py --dataset iscx-nonvpn
# ISCX-Tor
uv run preprocess.py --dataset iscx-tor
# ISCX-NonTor
uv run preprocess.py --dataset iscx-nontor
```
# Reference
[Code](https://github.com/ViktorAxelsen/MH-Net)  
[Paper](https://arxiv.org/abs/2501.03279)