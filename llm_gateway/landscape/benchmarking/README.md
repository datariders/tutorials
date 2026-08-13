# LLM Gateway sample projects

Four small, parallel projects — one per gateway — that all do the exact
same thing, so you can compare the developer experience directly:

> Ask an LLM to "Explain what an LLM gateway is, in two sentences,"
> using a primary model, with automatic fallback to a secondary model
> if the primary fails.

| Folder | Gateway | Client style |
|---|---|---|
| `litellm/` | LiteLLM | `litellm` Python SDK |
| `bifrost/` | Bifrost (Maxim AI) | OpenAI SDK pointed at a local Bifrost gateway |
| `portkey/` | Portkey | `portkey_ai` Python SDK |
| `openrouter/` | OpenRouter | OpenAI SDK pointed at OpenRouter |

Each folder is self-contained: its own `main.py` and `requirements.txt`.
The code in each is intentionally structured the same way:

1. Build a gateway client (this is the only truly gateway-specific part).
2. Define `PRIMARY_MODEL` and `FALLBACK_MODEL`.
3. Call `ask(prompt)`, which tries the primary model and falls back
   automatically on failure.
4. Print the response and which model actually answered.

## Setup common to all four

```bash
cd llm_gateway_samples/<gateway>
pip install -r requirements.txt
```

Each script reads its API key(s) from environment variables — see the
top of each `main.py` for exactly which ones. None of them require you
to sign up for all providers; each is runnable with just an OpenAI key
and (optionally) an Anthropic key, except OpenRouter/Portkey which use
their own single gateway API key to reach both.

## Benchmark: compare all four side by side

`benchmark.py` sends the identical prompt through all four gateways and
prints one table with latency, which model answered, and a response
snippet. Any gateway without its required env var set (or, for Bifrost,
without the local gateway running) is reported as `SKIPPED` or `FAILED`
instead of stopping the run.

```bash
pip install -r requirements-benchmark.txt
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export PORTKEY_API_KEY=...
export PORTKEY_OPENAI_VIRTUAL_KEY=...
export OPENROUTER_API_KEY=...
# optional: run Bifrost locally first -> npx -y @maximhq/bifrost
python benchmark.py
```

Example output shape:

```
Prompt: "Explain what an LLM gateway is, in two sentences."

Gateway    | Status | Latency (s) | Model used      | Response
-----------+--------+-------------+-----------------+----------------------------------------
LiteLLM    | OK     | 0.82        | gpt-4o-mini     | An LLM gateway is a unified layer...
Bifrost    | OK     | 0.65        | openai/gpt-4o-mini | An LLM gateway sits between...
Portkey    | OK     | 0.91        | gpt-4o-mini     | An LLM gateway routes requests...
OpenRouter | OK     | 0.77        | openai/gpt-4o-mini | An LLM gateway is infrastructure...
```
