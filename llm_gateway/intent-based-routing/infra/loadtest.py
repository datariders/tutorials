"""
Fires a mix of query types at the agent concurrently so attendees can watch
routing, caching, and rate limiting happen live.

Usage:
    python loadtest.py --agent-url http://localhost:8000 --requests 30 --concurrency 5
"""
import argparse
import asyncio
import random

import httpx

QUERIES = [
    "Write a Python function that reverses a linked list",
    "Summarize the plot of a mystery novel in three sentences",
    "Write a short poem about the ocean",
    "What causes rainbows?",
    "Fix this SQL query: SELCT * FROM users",
    "Brainstorm five taglines for a coffee shop",
]


async def fire(client: httpx.AsyncClient, url: str, query: str) -> None:
    try:
        resp = await client.post(f"{url}/chat", json={"query": query}, timeout=30.0)
        data = resp.json() if resp.status_code == 200 else {}
        print(
            f"[{resp.status_code}] intent={data.get('intent')} "
            f"routed_to={data.get('routed_to')} cache_hit={data.get('cache_hit')} "
            f"latency_ms={data.get('latency_ms')} :: {query[:40]}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc} :: {query[:40]}")


async def main(agent_url: str, requests: int, concurrency: int) -> None:
    sem = asyncio.Semaphore(concurrency)

    async def bound_fire(client, query):
        async with sem:
            await fire(client, agent_url, query)

    async with httpx.AsyncClient() as client:
        tasks = [
            bound_fire(client, random.choice(QUERIES)) for _ in range(requests)
        ]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(main(args.agent_url, args.requests, args.concurrency))
