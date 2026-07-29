"""Demo 1 — Multi-provider routing through ONE unified API.

Ask the exact same question three times, changing only the `model` string.
- "gpt-4o-mini"  -> routed to OpenAI
- "claude-haiku" -> routed to Anthropic
- "smart-chat"   -> load-balanced across BOTH providers

Your application code never imports the OpenAI or Anthropic SDKs directly and
never sees a provider key. The gateway handles provider selection.

Run:  python demos/01_routing.py
"""

from rich.console import Console
from rich.table import Table

from _client import get_client

console = Console()
client = get_client()

QUESTION = "In one short sentence, what is an LLM gateway?"
MODELS = ["gpt-4o-mini", "claude-haiku", "smart-chat"]


def main() -> None:
    table = Table(title="Same request, different upstream provider")
    table.add_column("Requested model", style="cyan")
    table.add_column("Served by (response model)", style="green")
    table.add_column("Answer", style="white")

    for model in MODELS:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": QUESTION}],
                max_tokens=60,
            )
            served_by = resp.model  # what the gateway actually called upstream
            answer = resp.choices[0].message.content.strip()
            table.add_row(model, served_by, answer)
        except Exception as e:
            # Don't let one unavailable provider abort the whole demo.
            table.add_row(model, "[red]error[/red]", f"[red]{type(e).__name__}[/red]")

    console.print(table)


if __name__ == "__main__":
    main()
