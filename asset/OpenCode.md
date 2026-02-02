# OpenCode
OpenCode is an open source AI coding agent. It’s available as a terminal-based interface, desktop app, or IDE extension.
## Installation
OpenCode empowers you to chat with AI about your codebase, automate repetitive coding tasks, and build software faster using advanced AI models. It supports multiple code completion engines and lets you bring your own OpenAI, Gemini, or Ollama model keys.

Once installed, OpenCode provides:
- **Terminal UI:** Code chat and command-line interfaces.
- **Desktop & IDE integrations:** Native apps and extensions for a seamless workflow.
- **Customizable models:** Use your preferred local or cloud LLMs.

Install OpenCode with the following command.
```bash
curl -fsSL https://opencode.ai/install | bash
```
To enable OpenCode to "make call inference provider" with Hugging Face, you need to set the appropriate permissions on your Hugging Face account or organization. This is typically done by generating an **Access Token** with "Read" or "Inference" permissions.

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
   Save it somewhere secure. You will need it to authenticate OpenCode with Hugging Face.

5. **Use the token in OpenCode:**  
   Add the token to your environment variables or OpenCode configuration as instructed in the OpenCode documentation. For example:
   ```bash
   export HF_TOKEN=your_access_token_here
   ```

Now, OpenCode can call Hugging Face inference providers using your token with the correct permissions.


![OpenCode screenshot](./image/image01.png)


# Usage
To get started with OpenCode, simply launch it from your terminal:

```bash
opencode
```
or 
```bash
opencode /path/to/project/directory
```
This will start an interactive coding session in your terminal, where you can chat with the AI about your code and automate tasks.  
Once you’re in the TUI, you can prompt it with a message.
```bash
Give me a quick summary of the codebase.
```
## Basic Usage

- **Chat with the AI:**  
  Type your programming questions or code requests directly into the prompt. The AI will respond and help generate, refactor, or explain code.

- **Run commands:**  
  You can use slash commands to control OpenCode's behavior or interact with your environment.

## Example - Listing Available Models

OpenCode supports multiple code completion engines. You can see which models are available by using the `/models` command inside the OpenCode terminal:

```bash
/models
```

This will display a list of installed and available models. You can then select your preferred model using:

```bash
/model <model_name>
```

For example, to select `gpt-3.5-turbo`, you might run:

```bash
/model gpt-3.5-turbo
```

## Customizing Configuration

You can control which AI models and providers OpenCode uses by configuring your environment variables or settings file. For example:

- **Set provider credentials:**  
  ```bash
  export HF_TOKEN=your_huggingface_token
  export OPENAI_API_KEY=your_openai_key
  ```
- **Edit config file:**  
  Customize model preferences, provider endpoints, and default behaviors in the OpenCode configuration file, usually located at `~/.config/opencode/config.yaml` or similar, depending on your operating system.

# Reference
[OpenCode documentation](https://opencode.ai/docs)
