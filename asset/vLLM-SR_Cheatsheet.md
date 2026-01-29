# vLLM-SR Cheatsheet
Semantic Router is an intelligent routing layer that dynamically selects the most suitable language model for each query based on multiple signals extracted from the request.  

vLLM-SR is a library for building and running LLM-based systems. It is designed to be easy to use and understand, and to be able to run on a variety of hardware.
## Installation

```bash
uv run pip install vllm-sr
```
# Usage
Initialize the configuration file.
```bash
uv run vllm-sr init
```
Serve the model.
```bash
uv run vllm-sr serve
```
# Reference
[vLLM-SR](https://vllm-semantic-router.com/docs/intro)