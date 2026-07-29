"""Demo 4 — Cost tracking & observability.

Every response through the gateway carries token usage, and LiteLLM computes
a dollar cost per call. This script makes a few calls and prints a running
spend table — the same data the gateway persists to its /spend endpoints and
admin UI when a database is configured.

Run:  python demos/04_cost_and_usage.py
"""

from rich.console import Console
from rich.table import Table

from _client import get_client

console = Console()
client = get_client()

PROMPTS = [
    ("gpt-4o-mini", "Summarize the plot of Hamlet in two sentences."),
    ("claude-haiku", "List three uses for a paperclip."),
    ("gpt-4o-mini", "Translate 'good morning' into French, Spanish, and Japanese."),
]


def main() -> None:
    table = Table(title="Per-call usage & cost (as tracked by the gateway)")
    table.add_column("Model", style="cyan")
    table.add_column("Prompt tok", justify="right")
    table.add_column("Compl tok", justify="right")
    table.add_column("Cost (USD)", justify="right", style="green")

    total_cost = 0.0
    for model, prompt in PROMPTS:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
        )
        usage = resp.usage
        # LiteLLM injects the computed response cost into hidden params.
        cost = getattr(resp, "_hidden_params", {}).get("response_cost") or 0.0
        total_cost += cost
        table.add_row(
            resp.model,
            str(usage.prompt_tokens),
            str(usage.completion_tokens),
            f"${cost:.6f}",
        )

    console.print(table)
    console.print(f"\nTotal spend this run: [bold green]${total_cost:.6f}[/bold green]")
    console.print(
        "\nTip: with a DATABASE_URL set, browse aggregate spend at "
        "[cyan]http://localhost:4000/ui[/cyan] (admin UI) or the "
        "[cyan]/spend/logs[/cyan] endpoint."
    )


if __name__ == "__main__":
    main()
