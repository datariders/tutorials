"""
LiteLLM sample project
=======================
Unified call across providers via the LiteLLM Python SDK, with
built-in automatic fallback.

Env vars needed:
    OPENAI_API_KEY
    ANTHROPIC_API_KEY

Install:
    pip install -r requirements.txt
"""

from dotenv import load_dotenv
load_dotenv()

from litellm import completion


PROMPT = "Explain what an LLM gateway is, in two sentences."
PRIMARY_MODEL = "gpt-4o-mini"
FALLBACK_MODEL = "claude-sonnet-5"


def ask(prompt: str) -> None:
    response = completion(
        model=PRIMARY_MODEL,
        messages=[{"role": "user", "content": prompt}],
        fallbacks=[FALLBACK_MODEL],  # LiteLLM handles the fallback internally
    )
    answer = response.choices[0].message.content
    model_used = response.model
    print(f"[litellm] answered by: {model_used}")
    print(answer)


if __name__ == "__main__":
    ask(PROMPT)
