"""
OpenRouter sample project
===========================
OpenRouter speaks the OpenAI chat-completions schema, so the OpenAI SDK
works as a drop-in client — point base_url at OpenRouter and use one
OpenRouter API key to reach every provider it supports.

Env vars needed:
    OPENROUTER_API_KEY

Install:
    pip install -r requirements.txt
"""

import os
from openai import OpenAI

PROMPT = "Explain what an LLM gateway is, in two sentences."
PRIMARY_MODEL = "openai/gpt-4o-mini"
FALLBACK_MODEL = "anthropic/claude-sonnet-4"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    default_headers={
        "HTTP-Referer": "https://example.com",  # optional, for OpenRouter leaderboards
        "X-Title": "LLM Gateway Sample",         # optional
    },
)


def ask(prompt: str) -> None:
    messages = [{"role": "user", "content": prompt}]
    try:
        response = client.chat.completions.create(model=PRIMARY_MODEL, messages=messages)
        model_used = PRIMARY_MODEL
    except Exception as primary_error:  # noqa: BLE001
        print(f"[openrouter] primary model failed ({primary_error}); falling back...")
        response = client.chat.completions.create(model=FALLBACK_MODEL, messages=messages)
        model_used = FALLBACK_MODEL

    print(f"[openrouter] answered by: {model_used}")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    ask(PROMPT)

# Note: OpenRouter can also do this fallback server-side in a single call by
# passing an ordered "models" list in extra_body, e.g.:
#   client.chat.completions.create(
#       model=PRIMARY_MODEL,
#       messages=messages,
#       extra_body={"models": [PRIMARY_MODEL, FALLBACK_MODEL]},
#   )
# OpenRouter then tries each model in order until one succeeds.
