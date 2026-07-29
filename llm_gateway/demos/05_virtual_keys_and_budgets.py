"""Demo 5 — Virtual keys, budgets & rate limits (admin API).

A gateway lets you mint scoped "virtual keys" for each team/app, each with its
own budget and rate limit — without ever sharing provider keys. This uses the
gateway's admin REST API (requires the master key).

NOTE: minting keys with budgets requires a DATABASE_URL configured on the
proxy. If you didn't set one up, this script will tell you so and exit cleanly.

Run:  python demos/05_virtual_keys_and_budgets.py
"""

import os

import requests
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:4000")
MASTER_KEY = os.environ.get("LITELLM_MASTER_KEY", "sk-demo-master-1234")
HEADERS = {"Authorization": f"Bearer {MASTER_KEY}"}


def main() -> None:
    console.rule("[bold]Minting a budget-limited virtual key[/bold]")
    payload = {
        "models": ["smart-chat", "gpt-4o-mini"],  # scope: only these models
        "max_budget": 0.10,                        # USD spend cap for this key
        "budget_duration": "1d",                   # resets daily
        "rpm_limit": 10,                           # requests/min for this key
        "metadata": {"team": "demo-team"},
    }
    resp = requests.post(
        f"{GATEWAY_URL}/key/generate", headers=HEADERS, json=payload, timeout=30
    )

    if resp.status_code != 200:
        console.print(
            f"[yellow]Could not mint a key ({resp.status_code}). "
            "This feature needs a DATABASE_URL on the proxy.[/yellow]"
        )
        console.print(resp.text)
        return

    data = resp.json()
    key = data.get("key")
    console.print(f"  New virtual key: [green]{key}[/green]")
    console.print(
        f"  Scope: models={payload['models']}, "
        f"budget=${payload['max_budget']}/{payload['budget_duration']}, "
        f"rpm={payload['rpm_limit']}"
    )
    console.print(
        "\nHand this key to a team/app instead of the master key. The gateway "
        "enforces its budget + rate limit automatically."
    )


if __name__ == "__main__":
    main()
