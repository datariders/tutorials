"""
hello_fallback.py — same call, but with an automatic fallback chain.

If the primary model errors out (rate limit, outage, missing key),
LiteLLM retries the next model in `fallbacks` for you — no extra
retry logic needed in your own code.

    python hello_fallback.py
"""

from dotenv import load_dotenv
from litellm import completion

load_dotenv()

PRIMARY = "gpt-5.4"
FALLBACKS = ["claude-sonnet-5", "gemini/gemini-2.5-pro"]


def main():
    response = completion(
        model=PRIMARY,
        messages=[{"role": "user", "content": "Say hello in exactly five words."}],
        fallbacks=FALLBACKS,
    )

    message = response["choices"][0]["message"]["content"]
    model_used = response.get("model", PRIMARY)

    print(f"Model used: {model_used}")
    print(f"Reply:      {message.strip()}")


if __name__ == "__main__":
    main()
