# Crossbow Chearsheet
World's first fully autonomous AI security engineer that finds and exploits vulnerabilities, performs SOC operations, forensics, and threat intelligence.

## Installation
Install npm.
```bash
sudo apt install npm
```
Install Crossbow.
```bash
uv add pip
uv run python -m pip install crossbow-agent
```
Set API key.
```bash
export OPENAI_API_KEY=your-key-here
# or ANTHROPIC_API_KEY or GEMINI_API_KEY
```
# Usage
Start Crossbow:
```bash
crossbow
```
Chat with Crossbow.
```bash
Find vulnerabilities in https://www.canva.com
```

## Reference
[Crossbow](https://github.com/harishsg993010/crossbow-agent)