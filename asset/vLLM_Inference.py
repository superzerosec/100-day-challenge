from openai import OpenAI

def main():
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
    )

    completion = client.chat.completions.create(
        model="Qwen/Qwen3-1.7B",
        messages = [
            {"role": "system", "content": "You are Llama, an AI assistant created by Meta."},
            {"role": "user", "content": [{"type": "text", "text": "which one is bigger, 9.11 or 9.9? think carefully."}]},
        ],
    )

    print(completion.choices[0].message.content)


if __name__ == "__main__":
    main()