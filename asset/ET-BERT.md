# ET-BERT
ET-BERT is a method for learning datagram contextual relationships from encrypted traffic, which could be directly applied to different encrypted traffic scenarios and accurately identify classes of traffic. First, ET-BERT employs multi-layer attention in large scale unlabelled traffic to learn both inter-datagram contextual and inter-traffic transport relationships. Second, ET-BERT could be applied to a specific scenario to identify traffic types by fine-tuning the labeled encrypted traffic on a small scale.

## Installation
```bash
git clone https://github.com/linwhitehat/ET-BERT.git
cd ET-BERT
uv init
uv add pip
uv add -r requirements.txt
```
# Usage
Download pre-trained model.
```bash
wget -O models/pre-trained_model.bin https://drive.google.com/file/d/1r1yE34dU2W8zSqx1FkB8gCWri4DQWVtE/view?usp=sharing
```
Download the dataset [here](https://drive.google.com/file/d/1oFaF0Hri6lraA5YuOt9JqOFBeqVjvwWN/view?usp=drive_link).
```bash
uv run fine-tuning/run_classifier.py --pretrained_model_path models/pre-trained_model.bin \
    --vocab_path models/encryptd_vocab.txt \
    --train_path datasets/cstnet-tls1.3/packet/train_dataset.tsv \
    --dev_path datasets/cstnet-tls1.3/packet/valid_dataset.tsv \
    --test_path datasets/cstnet-tls1.3/packet/test_dataset.tsv \
    --epochs_num 10 --batch_size 32 --embedding word_pos_seg \
    --encoder transformer --mask fully_visible \
    --seq_length 128 --learning_rate 2e-5
```
# Reference
[ET-BERT](https://github.com/linwhitehat/ET-BERT)