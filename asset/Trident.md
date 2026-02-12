# Trident
Trident is a novel framework to detect unknown attack traffic in a fine-grained and incremental manner. It can realize known class classification, fine-grained unknown class detection, and incremental model updates (including sample increments and class increments). At a high level, Trident consists of three tightly coupled components named tSieve, tScissors, and tMagnifier. They are responsible for profiling traffic, determining outlier thresholds, and clustering respectively.

## Installation
```bash
git clone https://github.com/Secbrain/Trident
cd Trident
uv init
uv add pip
uv add scipy numpy pandas matplotlib scikit-learn torch tqdm
```

# Usage
```bash
uv run code/main_process.py
```
# Reference
[Code](https://github.com/Secbrain/Trident)  
[Paper](https://dl.acm.org/doi/10.1145/3589334.3645407)