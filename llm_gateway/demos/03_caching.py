"""Demo 3 — Response caching.

Send the SAME request twice. The first call hits the provider; the second is
served from the gateway cache (configured in config.yaml -> litellm_settings).

We detect a cache hit two ways:
  * wall-clock latency drops dramatically on the second call
  * the response carries a cache-hit marker in its hidden params / headers

Run:  python demos/03_caching.py
"""

import time

from rich.console import Console

from _client import get_client

console = Console()
client = get_client()

PROMPT = "Give me a fun fact about the ocean. Keep it identical every time."


def timed_call(label: str):
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=60,
        # Deterministic so the cache key is stable across calls.
        temperature=0,
    )
    elapsed = time.perf_counter() - start
    # LiteLLM exposes a cache-hit flag in the response's hidden params.
    cache_hit = getattr(resp, "_hidden_params", {}).get("cache_hit")
    console.print(
        f"  {label}: [cyan]{elapsed:.3f}s[/cyan]  "
        f"cache_hit=[green]{cache_hit}[/green]"
    )
    return elapsed


def main() -> None:
    console.rule("[bold]Caching identical requests[/bold]")
    first = timed_call("call #1 (cold)")
    second = timed_call("call #2 (warm)")
    if second < first:
        console.print(
            f"\nSecond call was [green]{first / max(second, 1e-6):.1f}x[/green] "
            "faster — served from cache."
        )
    else:
        console.print(
            "\n[yellow]No speedup detected — is caching enabled and running?[/yellow]"
        )


if __name__ == "__main__":
    main()
