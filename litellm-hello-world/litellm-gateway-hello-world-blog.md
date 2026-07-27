# One Gateway, Every LLM: A Hello World with LiteLLM

Every LLM provider ships its own SDK, its own request format, and its own way of handling errors. Switch from OpenAI to Anthropic to Gemini and you're rewriting integration code each time — different client, different response shape, different retry logic.

[LiteLLM](https://www.litellm.ai/) fixes that by acting as a **gateway**: one OpenAI-compatible function that talks to 100+ providers underneath. Your code doesn't need to know or care which model actually answered.

This post walks through the smallest possible example — a true "hello world" — and the full code is in the companion repo:

**👉 [github.com/\<your-username\>/litellm-hello-world](https://github.com/)**

## The idea in one function

Normally, calling OpenAI and calling Anthropic are two different SDKs with two different shapes. LiteLLM collapses that into one call:

```python
from litellm import completion

response = completion(
    model="claude-sonnet-5",
    messages=[{"role": "user", "content": "Say hello in exactly five words."}],
)
```

Change `model` to `"gpt-5.4"` or `"gemini/gemini-2.5-pro"` and nothing else in your code changes. That's the entire pitch of a gateway.

## Setting it up

Clone the repo and install the one dependency:

```bash
git clone https://github.com/<your-username>/litellm-hello-world.git
cd litellm-hello-world
pip install -r requirements.txt
cp .env.example .env
```

Drop in a key for whichever provider you want to try — you only need one:

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Running it

```bash
python hello.py
```

```
Model:  claude-sonnet-5
Reply:  Hello there, wonderful human!
Tokens: 24
```

Swap providers without touching the code:

```bash
LITELLM_MODEL=gpt-5.4 python hello.py
```

## Bonus: automatic fallback

The repo also includes `hello_fallback.py`, which shows the feature that makes a gateway more than a convenience. Pass a list of backup models, and if the primary one errors — rate limit, outage, bad key — LiteLLM retries the next one for you:

```python
response = completion(
    model="gpt-5.4",
    messages=[{"role": "user", "content": "Say hello in exactly five words."}],
    fallbacks=["claude-sonnet-5", "gemini/gemini-2.5-pro"],
)
```

No try/except ladder, no manual retry loop — just a list.

## What this doesn't cover (on purpose)

This hello world uses the LiteLLM **Python SDK**, which is the fastest way to feel the value. It's a single process, no server, nothing to deploy.

The other half of LiteLLM is the **Proxy Server** — a self-hosted service you run behind your whole team, with virtual keys, per-team budgets, load balancing, and centralized cost logging. That's the right move once more than one person or service needs shared, governed access to models. It's a natural follow-up to this post, but deliberately out of scope here: the goal today was showing the core idea in the smallest possible amount of code.

## Try it

Full source, README, and both scripts are in the repo: **[litellm-hello-world](https://github.com/)**. Fork it, swap in your own prompt, and try wiring a third provider into the fallback chain.
