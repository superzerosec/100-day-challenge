from vllm import LLM, SamplingParams

def main():
    # Initialize the vLLM model
    llm = LLM(model="Qwen/Qwen3-1.7B")

    # Prepare the prompt (simulate a chat format)
    system_message = "You are Llama, an AI assistant created by Meta."
    user_message = "which one is bigger, 9.11 or 9.9? think carefully."
    prompt = f"{system_message}\nUser: {user_message}\nAssistant:"

    # Set sampling parameters similar to the original (if desired)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

    # Generate output using vLLM
    outputs = llm.generate([prompt], sampling_params)
    print(outputs[0].outputs[0].text)

if __name__ == "__main__":
    main()