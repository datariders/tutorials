"""
hello.py — the smallest possible LiteLLM example.

LiteLLM gives you one function, `completion()`, that works the same way
no matter which LLM provider is answering. Set the API key for whichever
provider you want to try (see .env.example) and run:

    python hello.py

Try changing MODEL below to a different provider's model name and
run it again — nothing else in this file needs to change.
"""

import os
from dotenv import load_dotenv
from litellm import completion

load_dotenv()

# Swap this for any supported model — e.g. "gpt-5.4", "gemini/gemini-2.5-pro"
MODEL = os.getenv("LITELLM_MODEL", "claude-sonnet-5")


def main():
    response = completion(
        model=MODEL,
        messages=[{"role": "user", "content": "Say hello in exactly five words."}],
    )

    message = response["choices"][0]["message"]["content"]
    usage = response["usage"]

    print(f"Model:  {MODEL}")
    print(f"Reply:  {message.strip()}")
    print(f"Tokens: {usage['total_tokens']}")


if __name__ == "__main__":
    main()
