# VulnLLM-R-7B
VulnLLM-R is the first specialized reasoning Large Language Model designed specifically for software vulnerability detection.

Unlike traditional static analysis tools (like CodeQL) or small LLMs that rely on simple pattern matching, VulnLLM-R is trained to reason step-by-step about data flow, control flow, and security context. It mimics the thought process of a human security auditor to identify complex logic vulnerabilities with high accuracy.

- **Reasoning-Based Detection**: Does not just classify code; it generates a "Chain-of-Thought" to analyze why a vulnerability exists.
- **Superior Accuracy**: Outperforms commercial giants (like Claude-3.7-Sonnet, o3-mini) and industry-standard tools (CodeQL, AFL++) on key benchmarks.
- **Efficiency**: Achieves SOTA performance with only 7B parameters, making it 30x smaller and significantly faster than general-purpose reasoning models.
- **Broad Coverage**: Trained and tested on C, C++, Python, and Java (zero-shot generalization).

## Installation
```bash
uv add pip
uv run python -m pip install transformers
uv run python -m pip install torch
```

# DVCP
Vulnarable code from [Damn Vulnerable C Program (DVCP)](https://github.com/hardik05/Damn_Vulnerable_C_Program/tree/master). Add vulnerable code to `asset/HuggingFace_VulnLLM-R-7B.py`.
```bash
uv run asset/HuggingFace_VulnLLM-R-7B.py
```
Answer.
```markdown
<think>
Okay, let's try to figure out the vulnerabilities in this code. Starting with the ProcessImage function, which reads image data into a struct. The struct includes a char array data[10], so any input larger than 10 bytes here would overflow the buffer. The fread uses sizeof(struct Image), which includes the entire struct, including the fixed-size data array. If the file's data exceeds 10 bytes, this causes a stack-based buffer overflow (CWE-121).

Next, examining the memcpy operations: buff1 is allocated based on img.width + img.height. Since width and height are integers, very large values could lead to excessive memory allocation, but the primary issue remains the initial struct read. The code checks for division by zero (img.width / img.height), but this isn't a direct vulnerability.

For heap overflows (CWE-122), the memcpy into buff1 uses sizeof(img.data), which is 11 bytes (including the null terminator). However, the struct's data array is only 10 bytes, so reading 11 bytes from the file overflows this buffer. This is a clear stack overflow, not heap-based.

The code also has a use-after-free scenario when buff1 is freed and then accessed via buff3, but this doesn't directly relate to the listed CWEs. Double-free issues aren't present here, and the stack overflow (CWE-121) is the most critical vulnerability due to the struct's fixed-size data array being overwritten by untrusted input.
</think>
## Step 1: Analyze the struct's data array
The struct Image has a char array `data[10]`, which can hold up to 10 bytes plus a null terminator. When reading the image data via `fread(&img, sizeof(img), 1, fp)`, the entire struct is read, including the `data` array. If the file contains more than 10 bytes in the `data` field, this results in a stack-based buffer overflow (CWE-121) as the excess data overwrites adjacent stack memory.

## Step 2: Check other potential vulnerabilities
- **Heap overflow (CWE-122)**: The `memcpy` into `buff1` uses `sizeof(img.data)` (11 bytes), but `buff1` is allocated based on `img.width + img.height`. While large allocations could cause resource exhaustion, the immediate overflow occurs during 
```
# Reference
[VulnLLM-R-7B](https://huggingface.co/UCSB-SURFI/VulnLLM-R-7B)  
[Damn Vulnerable C Program (DVCP)](https://github.com/hardik05/Damn_Vulnerable_C_Program/tree/master)