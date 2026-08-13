"""
Bifrost sample project
========================
Bifrost (open-source, Go, by Maxim AI) exposes a 100%-OpenAI-compatible
API, so the standard OpenAI Python SDK is the client — you just point
`base_url` at your local Bifrost gateway instead of api.openai.com.

Run Bifrost first (zero config needed):
    npx -y @maximhq/bifrost
    # or: docker run -p 8080:8080 maximhq/bifrost

Provider API keys are configured inside Bifrost itself (via its web UI
at http://localhost:8080 or a config file) rather than in this script.

Install:
    pip install -r requirements.txt
"""

from openai import OpenAI

PROMPT = "Explain what an LLM gateway is, in two sentences."
PRIMARY_MODEL = "openai/gpt-4o-mini"       # provider-prefixed model routing
FALLBACK_MODEL = "anthropic/claude-sonnet-4"

client = OpenAI(
    base_url="http://localhost:8080/openai",
    api_key="dummy-key",  # actual provider keys live in Bifrost's config, not here
)


def ask(prompt: str) -> None:
    messages = [{"role": "user", "content": prompt}]
    try:
        response = client.chat.completions.create(model=PRIMARY_MODEL, messages=messages)
        model_used = PRIMARY_MODEL
    except Exception as primary_error:  # noqa: BLE001
        print(f"[bifrost] primary model failed ({primary_error}); falling back...")
        response = client.chat.completions.create(model=FALLBACK_MODEL, messages=messages)
        model_used = FALLBACK_MODEL

    print(f"[bifrost] answered by: {model_used}")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    ask(PROMPT)

# Note: Bifrost can also do this fallback server-side (no try/except needed)
# by configuring a fallback chain in its gateway config, so every client,
# in any language, gets automatic failover for free.
