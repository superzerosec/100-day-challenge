# HuggingFace Cheatsheet

## Installation
```bash
uv add pip
uv run python -m pip install huggingface_hub
```
## Get Token from HuggingFace
Create token [web](https://huggingface.co/) -> icon profile -> setting -> access token

To enable your application to "make call inference provider" or any related permissions with Hugging Face, you need to set the appropriate permissions on your Hugging Face account. This is typically done by generating an **Access Token** with "Read" or "Inference" permissions.

Follow these steps:

1. **Go to your Hugging Face account settings:**  
   Visit [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

2. **Create a new access token:**  
   Click on "New token".

3. **Set permissions:**  
   - Name your token (e.g., `token01`).
   - Under **Role**, choose either `Read` or, if available, `Inference`.
   - For most cases, `Read` is sufficient to call public inference endpoints; some providers may have a specific `Inference` permission.

4. **Copy your access token:**  
   Save it somewhere secure. You will need it to authenticate your application with Hugging Face.

   ```bash
   export HF_TOKEN=your_access_token_here
   ```

![screenshot](./image/image01.png)

Now, application can call Hugging Face inference providers using your token with the correct permissions.

## Config Environment Variables
Add into `.bashrc` file.
```bash
export HF_TOKEN=your_access_token_here
export SHARED_DIR=custom_shared_dir_here
export HF_HOME="${SHARED_DIR}/.cache/huggingface"
```

# Usage

You can quickly generate text from a Hugging Face model using the `transformers` library with just a few lines of code.  
Below is an example that loads an LLM from Hugging Face, sets up a conversational instruction, and runs inference locally.  
Make sure your environment is authenticated with your `HF_TOKEN` if accessing private models.

```python
import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Load tokenizer and model (from local cache or Hugging Face Hub)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "hi"},
]

# Generate text response
outputs = pipeline(
    messages,
    max_new_tokens=256,
    pad_token_id = pipeline.tokenizer.eos_token_id,
)

# Print the assistant's response
print(outputs[0]["generated_text"][-1]["content"])
```

- Replace `model_id` with the model you wish to use.
- For basic text completion, you can also use the pipeline directly with a prompt string.
- Make sure to set appropriate model/device/dtype settings for your hardware.

# Reference
[Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
