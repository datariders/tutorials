# litellm-hello-world

The smallest possible example of [LiteLLM](https://www.litellm.ai/): one function, `completion()`, that talks to 100+ LLM providers through the same interface.

Companion repo for the blog post *"One Gateway, Every LLM: A Hello World with LiteLLM."*

## What's here

- `hello.py` — a single call to a model, printing the reply and token usage
- `hello_fallback.py` — the same call, but with an automatic fallback chain across providers
- `.env.example` — template for the API keys you'll need

## Setup

```bash
git clone https://github.com/<your-username>/litellm-hello-world.git
cd litellm-hello-world
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add the key for at least one provider (OpenAI, Anthropic, or Gemini).

## Run it

```bash
python hello.py
```

```
Model:  claude-sonnet-5
Reply:  # LiteLLM

LiteLLM is an open-source Python library that provides a unified interface for calling over 100 different Large Language Model (LLM) APIs, including OpenAI, Anthropic, Cohere, Azure, Hugging Face, and many others. It standardizes inputs and outputs to match the OpenAI API format, so developers can switch between providers without rewriting their code. The library supports both a Python SDK for direct integration and a proxy server that can be deployed to manage multiple LLM calls centrally. Key features include load balancing, fallback logic (automatically retrying with a different model if one fails), and cost tracking across providers. It also supports streaming responses, function calling, and async operations for production-grade applications. LiteLLM is particularly useful for teams that want to avoid vendor lock-in or need to route requests across multiple LLM providers based on cost, latency, or availability. The proxy server component can act as a gateway, adding features like rate limiting, budget management, and logging/observability integrations (e.g., with tools like Langfuse or Helicone). It's widely used in LLM application development for simplifying multi-provider orchestration and reducing complexity when experimenting with or scaling across different models.
Tokens: 420
```

Change the model without touching any code:

```bash
LITELLM_MODEL=gpt-5.4 python hello.py
```

Try the fallback example:

```bash
python hello_fallback.py
```

## Why this matters

Every provider has its own SDK, request shape, and error format. LiteLLM normalizes all of that behind one OpenAI-compatible call, so switching models — or falling back to a backup when one is down — is a one-line change instead of a rewrite.

This repo uses the LiteLLM **Python SDK**, the fastest way to try it out. For production use across a team, LiteLLM also ships a self-hostable **Proxy Server** with virtual keys, budgets, and load balancing — a good next step once a single script isn't enough.

## Learn more

- [LiteLLM docs](https://docs.litellm.ai/)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)

## License

MIT — see [LICENSE](LICENSE).
