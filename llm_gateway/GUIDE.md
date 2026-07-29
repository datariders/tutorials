# Step-by-Step Guide: Building an LLM Gateway Demo with LiteLLM

This guide walks you from an empty folder to a working LLM gateway that
demonstrates multi-provider routing, fallbacks/load balancing, caching, cost
tracking, and rate limits. Each step says *what to do* and *what you should
see*, so it doubles as a demo script when you present it.

Estimated time: ~15 minutes for the core demo, ~30 with the optional
Redis/Postgres infrastructure.

---

## What is an LLM gateway?

An LLM gateway (a.k.a. AI gateway or model router) is a single service that
sits between your application and one or more model providers. Your app speaks
one API — here, the OpenAI-compatible API — and the gateway centralizes the
cross-cutting concerns: which provider to call, what to do when one fails, how
to cache and cap spend, and how to observe it all. LiteLLM Proxy is a popular
open-source implementation.

The key idea to show your audience: **your application code never changes.**
It keeps calling `client.chat.completions.create(...)`. Everything else is
configuration on the gateway.

---

## Prerequisites

- **Python 3.11 or 3.12** (NOT 3.13+/3.14 — newer versions break LiteLLM's
  `uvloop` dependency with an `ImportError` on startup)
- At least **one** provider API key (OpenAI *or* Anthropic is enough)
- Optional: Docker, if you want the Redis + Postgres "full" demo
- The project files from this repo (`config.yaml`, `demos/`, etc.)

---

## Step 1 — Set up the project and install LiteLLM

Create a virtual environment on **Python 3.12**. If you have
[`uv`](https://docs.astral.sh/uv/) installed, use it — it reliably provisions a
correct 3.12 (this avoids a common macOS pitfall where a `uv`-managed or
Homebrew Python produces a broken stdlib `venv`):

```bash
cd litellm-gateway-demo
uv venv --python 3.12 .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

Prefer plain `venv`? It works **only if** `python3.12` is a normal CPython
install (e.g. `/opt/homebrew/bin/python3.12`):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt   # use `python -m pip`, not bare pip3
```

**You should see:** LiteLLM and the OpenAI SDK install successfully. Confirm
the CLI is available:

```bash
litellm --version
```

---

## Step 2 — Add your keys

Copy the template and fill in what you have:

```bash
cp .env.example .env
```

Edit `.env`:

- Set **at least one** of `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`.
- Set `LITELLM_MASTER_KEY` to any `sk-...` string. This is the key your *app*
  uses to talk to the gateway — it never sees the real provider keys.

> Why this matters for the demo: separating the master key from provider keys
> is the security story of a gateway. App teams get a master (or virtual) key;
> only the gateway holds provider credentials.

---

## Step 3 — Understand the config (the heart of the demo)

Open `config.yaml`. It has four sections worth narrating:

1. **`model_list`** — the routing table. Notice that `smart-chat` appears
   **twice**, backed by OpenAI *and* Anthropic. Two entries sharing a
   `model_name` form a load-balancing group. Other aliases (`gpt-4o-mini`,
   `claude-haiku`) route to exactly one provider.

   > **Model names go stale.** Providers retire models over time. If a call
   > returns `404` / `not_found_error`, the model ID in `config.yaml` no longer
   > exists for your key. List what your key can use and update the config:
   > `curl https://api.anthropic.com/v1/models -H "x-api-key: $ANTHROPIC_API_KEY" -H "anthropic-version: 2023-06-01"`

2. **`router_settings`** — `fallbacks` say "if `premium-chat` fails, use
   `smart-chat`." `num_retries`, `cooldown_time`, and `routing_strategy`
   control retry and balancing behavior.

3. **`litellm_settings`** — `cache: true` enables response caching. It's set to
   `type: local` (in-memory) so the demo needs no extra infra; switch to
   `type: redis` (with a Redis host) for shared caching. To ship logs to an
   observability tool, add `success_callback`/`failure_callback` here (e.g.
   `["datadog"]`) — omitted by default so the demo runs with no extra accounts.

4. **`general_settings`** — `master_key` for auth, plus commented-out
   `database_url`/`max_budget` you'll enable in the optional Step 8.

---

## Step 4 — Start the gateway

In one terminal, load your env vars and launch:

```bash
set -a; source .env; set +a                  # macOS/Linux — loads keys safely
litellm --config config.yaml --port 4000 --detailed_debug
```

> **Why `set -a; source .env`** and not `export $(grep ... | xargs)`? The
> `xargs` trick silently corrupts values containing `/`, `+`, or `=` — common
> in API keys — leaving them empty. That produces confusing `not_found` /
> auth errors from providers. `set -a; source .env; set +a` loads them intact.
>
> **The gateway terminal is the one that needs the provider keys** — it's the
> process that calls OpenAI/Anthropic. The demo scripts (next steps) only talk
> to `localhost:4000`. Verify the keys are present here first:
> `echo "OpenAI:${#OPENAI_API_KEY} Anthropic:${#ANTHROPIC_API_KEY}"` (both > 0).

**You should see:** startup logs ending with the server listening on
`http://0.0.0.0:4000`. Sanity-check it:

```bash
curl http://localhost:4000/health/liveliness      # -> "I'm alive!"
curl http://localhost:4000/v1/models -H "Authorization: Bearer $LITELLM_MASTER_KEY"
```

The `/v1/models` call lists the aliases from your `model_list`.

Leave this terminal running. Open a **second** terminal (same venv,
`source .venv/bin/activate`) for the demo scripts.

---

## Step 5 — Demo A: Multi-provider routing

```bash
cd demos
python 01_routing.py
```

**What it does:** sends the *same* question three times with only the `model`
string changing — `gpt-4o-mini`, `claude-haiku`, then `smart-chat`.

**You should see:** a table where the "served by" column shows OpenAI for one
row, Anthropic for another, and either provider for `smart-chat`. Same code,
different backends — that's routing.

> Talking point: your app didn't import any provider SDK. It only changed a
> string. Swapping providers is now a config change, not a code change.

---

## Step 6 — Demo B: Fallbacks & load balancing

```bash
python 02_fallbacks.py
```

**What it does:** first fires six `smart-chat` calls and tallies which provider
served each (load balancing). Then it requests `premium-chat` with
`mock_testing_fallbacks` set, which forces the primary model to fail so you can
watch the gateway fall back to `smart-chat`.

**You should see:** a distribution split across both providers, then a line
confirming the `premium-chat` request succeeded *via a fallback model*.

> Talking point: in production this is your resilience layer — a provider
> outage or rate-limit doesn't take your app down.

---

## Step 7 — Demo C: Caching

```bash
python 03_caching.py
```

**What it does:** sends an identical, deterministic (`temperature=0`) request
twice.

**You should see:** the second call is dramatically faster and reports
`cache_hit=True`. The script prints the speedup factor.

> Talking point: for repeated or templated prompts, caching cuts both latency
> and spend. Out of the box this cache is in-memory (single process); Step 8
> makes it shared via Redis.

---

## Step 8 (optional) — Persistent spend, admin UI, Redis cache

This unlocks cross-process caching, persisted spend logs, the admin UI, and
budget-enforcing virtual keys.

```bash
docker compose up -d          # starts Redis + Postgres
```

Then in `.env`, uncomment/set:

```bash
REDIS_HOST=localhost
REDIS_PORT=6379
DATABASE_URL=postgresql://llmproxy:pass@localhost:5432/litellm
```

Restart the gateway (Ctrl-C, re-export env, relaunch as in Step 4). On boot
with a database, LiteLLM runs migrations automatically.

**You should see:** the admin UI at `http://localhost:4000/ui` (log in with the
master key). Caching now works across processes.

---

## Step 9 — Demo D: Cost & observability

```bash
python 04_cost_and_usage.py
```

**What it does:** makes several calls and prints per-call prompt/completion
tokens and computed dollar cost, plus a running total.

**You should see:** a spend table. With Step 8's database enabled, the same
data is queryable at `/spend/logs` and visualized in the admin UI over time.

> Talking point: centralized cost visibility per model, per key, per team is
> one of the strongest reasons orgs adopt a gateway.

---

## Step 10 — Demo E: Virtual keys, budgets & rate limits

*(Requires the Postgres database from Step 8.)*

```bash
python 05_virtual_keys_and_budgets.py
```

**What it does:** calls the gateway's admin API to mint a **virtual key**
scoped to specific models, with a daily budget and a per-minute rate limit.

**You should see:** a freshly generated `sk-...` key printed with its scope.
Hand that key to a team/app instead of the master key; the gateway enforces its
budget and rate limit automatically. If you didn't configure a database, the
script tells you cleanly and exits.

> Talking point: this is multi-tenancy and governance — every consumer gets its
> own guardrails without ever touching provider credentials.

---

## Suggested demo narrative (5-minute version)

1. Show `config.yaml` — "one file describes the whole gateway."
2. Run `01_routing.py` — "same code, different providers behind one API."
3. Run `02_fallbacks.py` — "a provider fails; the app doesn't."
4. Run `03_caching.py` — "repeat requests are near-instant and free."
5. Run `04_cost_and_usage.py` — "and here's exactly what it cost."
6. (If time) Open `/ui` and run `05_...py` — "per-team keys with budgets."

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ImportError: cannot import name 'BaseDefaultEventLoopPolicy'` on startup | You're on Python 3.13+/3.14. Recreate the venv on **3.12** (Step 1). |
| `ModuleNotFoundError: No module named 'encodings'` / broken venv | Your `python3.12` is a `uv`/standalone build; stdlib `venv` breaks it. Use `uv venv --python 3.12 .venv` instead. |
| `externally-managed-environment` from pip | You're hitting Homebrew's pip, not the venv's. Activate the venv and use `python -m pip`, or install with `uv pip`. |
| `not_found_error` / `404` for a model | The model ID in `config.yaml` is retired for your key. List available models (see Step 3 note) and update it. |
| Provider `AuthenticationError` / key shows 0 chars | Keys not loaded in the **gateway** terminal. Use `set -a; source .env; set +a` (not the `xargs` trick), then relaunch the gateway there. |
| `ValueError: Either 'host' or 'url' must be specified for redis` | `cache_params.type: redis` needs a running Redis + `REDIS_HOST`. Use `type: local` for the no-infra demo. |
| `Connection refused` in demos | The gateway (Step 4) isn't running or isn't on port 4000. |
| `AuthenticationError` from the gateway itself | `LITELLM_MASTER_KEY` in `.env` must match what the demo client sends. |
| No cache speedup | Ensure `cache: true` in config and `temperature=0` so requests are identical. |
| Can't mint virtual keys | Needs `DATABASE_URL` (Step 8). |

---

## Where to go next

- Add real observability: wire the callbacks to Datadog, OpenTelemetry, or
  another logging/tracing backend.
- Add guardrails: PII redaction, prompt-injection checks, allowed-model lists.
- Put it behind your own auth/ingress and deploy the container to your cloud.
- Extend `model_list` with Bedrock, Vertex, Azure OpenAI, or local models
  (Ollama) — all behind the same unified API.
