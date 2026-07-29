"""Demo 2 — Fallbacks & load balancing.

Two things to observe:

1. Load balancing: fire several "smart-chat" requests and watch them spread
   across OpenAI and Anthropic (see the `served by` column vary).

2. Fallback: request "premium-chat" but force a failure by sending a bogus
   per-request override. The gateway retries, then falls back to "smart-chat"
   per the `fallbacks` rules in config.yaml, so the call still succeeds.

Run:  python demos/02_fallbacks.py
"""

from collections import Counter

from rich.console import Console

from _client import get_client

console = Console()
client = get_client()


def show_load_balancing(n: int = 6) -> None:
    console.rule("[bold]Load balancing across providers[/bold]")
    served = Counter()
    for i in range(n):
        resp = client.chat.completions.create(
            model="smart-chat",
            messages=[{"role": "user", "content": f"Say hi (request {i})."}],
            max_tokens=10,
        )
        served[resp.model] += 1
        console.print(f"  request {i}  ->  served by [green]{resp.model}[/green]")
    console.print(f"\nDistribution: [cyan]{dict(served)}[/cyan]\n")


def show_fallback() -> None:
    console.rule("[bold]Fallback on failure[/bold]")
    # Ask for premium-chat but trigger a failure with an impossible setting.
    # LiteLLM retries, then falls back to smart-chat (see config.yaml).
    resp = client.chat.completions.create(
        model="premium-chat",
        messages=[{"role": "user", "content": "One word: are you there?"}],
        max_tokens=5,
        # `mock_testing_fallbacks` is a LiteLLM feature: it forces the primary
        # model to fail so you can SEE the fallback fire without breaking keys.
        extra_body={"mock_testing_fallbacks": True},
    )
    console.print(
        f"  Requested [yellow]premium-chat[/yellow], "
        f"succeeded via [green]{resp.model}[/green] after fallback."
    )


def main() -> None:
    show_load_balancing()
    show_fallback()


if __name__ == "__main__":
    main()
