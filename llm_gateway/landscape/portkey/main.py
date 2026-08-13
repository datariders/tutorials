"""
Portkey sample project
========================
Portkey's Python SDK is built on top of the OpenAI SDK. You route to a
provider via a "virtual key" / provider slug configured in your Portkey
dashboard, so the same `chat.completions.create` call works everywhere.

Env vars needed:
    PORTKEY_API_KEY
    PORTKEY_OPENAI_VIRTUAL_KEY     (virtual key for your OpenAI account)
    PORTKEY_ANTHROPIC_VIRTUAL_KEY  (virtual key for your Anthropic account)

Install:
    pip install -r requirements.txt
"""

import os
from portkey_ai import Portkey

PROMPT = "Explain what an LLM gateway is, in two sentences."
PRIMARY_MODEL = "gpt-4o-mini"
FALLBACK_MODEL = "claude-sonnet-5"

PRIMARY_VIRTUAL_KEY = os.environ["PORTKEY_OPENAI_VIRTUAL_KEY"]
FALLBACK_VIRTUAL_KEY = os.environ["PORTKEY_ANTHROPIC_VIRTUAL_KEY"]


def ask(prompt: str) -> None:
    messages = [{"role": "user", "content": prompt}]

    primary_client = Portkey(
        api_key=os.environ["PORTKEY_API_KEY"],
        virtual_key=PRIMARY_VIRTUAL_KEY,
    )
    try:
        response = primary_client.chat.completions.create(model=PRIMARY_MODEL, messages=messages)
        model_used = PRIMARY_MODEL
    except Exception as primary_error:  # noqa: BLE001
        print(f"[portkey] primary model failed ({primary_error}); falling back...")
        fallback_client = Portkey(
            api_key=os.environ["PORTKEY_API_KEY"],
            virtual_key=FALLBACK_VIRTUAL_KEY,
        )
        response = fallback_client.chat.completions.create(model=FALLBACK_MODEL, messages=messages)
        model_used = FALLBACK_MODEL

    print(f"[portkey] answered by: {model_used}")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    ask(PROMPT)

# Note: Portkey can also express this fallback declaratively via a Config
# object (routing strategy "fallback" over an ordered list of targets),
# defined once in the Portkey dashboard and referenced by config ID, so
# application code stays a single unconditional chat.completions.create call.
