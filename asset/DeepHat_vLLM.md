# DeepHat with vLLM

## Introduction

Deep Hat is trained on real-world attack patterns, operator-inspired tasks, and a continuous flow of high-signal security data to support deep analysis, exploit exploration, adversary emulation, and multi-step offensive reasoning. Deep Hat works inside Kindo, giving you private execution, long-context understanding, and the ability to automate both offensive and defensive workflows across your environment. It is designed for sensitive, technical work that commercial models cannot support.

# Usage
## Serving DeepHat with vLLM

To serve the DeepHat model using vLLM, you will first need to have the DeepHat model repository or reference available locally or from your configured model hub.

You can start a vLLM server using the `uv` runner and the `vllm serve` command. Replace `DeepHat/DeepHat-V1-7B` with the correct model path if needed, and adjust the parameters to fit your hardware (such as offload, GPU, or CPU settings):

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
uv run vllm serve DeepHat/DeepHat-V1-7B --cpu-offload-gb 20 --max-model-len 8192
```
- `--cpu-offload-gb 20` allows a portion of the model to reside in system RAM, making it possible to serve larger models on GPU or even with limited GPU resources.
- `--max-model-len 8192` sets the maximum sequence length for your input and output tokens (context window). Adjust as necessary for your use case.

By default, vLLM will launch an OpenAI-compatible API server at `http://localhost:8000/v1`.  
Once the server is running, you can interact with DeepHat using the `vllm chat` CLI tool.
```bash
uv run vllm chat --model DeepHat/DeepHat-V1-7B
```
Prompt.
```bash
Check CrowdStrike Falcon for high-confidence critical alerts. If a malware detection is found, execute automated containment immediately:
- Isolate the infected endpoint in CrowdStrike Falcon
- Block any malicious C2 IP addresses in Cisco Firepower Management Center
- Disable the compromised user account in Microsoft Entra ID
- After each action, verify it succeeded. Log all actions with timestamps. Create a Jira ticket in the Kindo space documenting all containment actions taken.
- Present output as a containment execution log showing each action, system used, verification status, and timestamps
```
# Reference
[DeepHat](https://www.deephat.ai/)  
[HuggingFace](https://huggingface.co/DeepHat/DeepHat-V1-7B)