"""
Benchmark: LiteLLM vs Bifrost vs Portkey vs OpenRouter
=========================================================
Runs the exact same prompt through all four gateways and prints a
side-by-side table of latency, which model actually answered, and a
snippet of the response. Any gateway that isn't configured (missing
env var, server not running, request failed) is reported as SKIPPED
or FAILED rather than crashing the whole run.

Env vars used (set only the ones for gateways you want to test):
    OPENAI_API_KEY                    (for LiteLLM direct calls)
    ANTHROPIC_API_KEY                 (for LiteLLM fallback)
    PORTKEY_API_KEY
    PORTKEY_OPENAI_VIRTUAL_KEY
    PORTKEY_ANTHROPIC_VIRTUAL_KEY
    OPENROUTER_API_KEY
    # Bifrost needs no env var here — provider keys are configured
    # inside the running Bifrost gateway itself (http://localhost:8080)

Install:
    pip install -r requirements-benchmark.txt

Run:
    python benchmark.py
"""

import os
import time
from dataclasses import dataclass
from typing import Optional

PROMPT = "Explain what an LLM gateway is, in two sentences."


@dataclass
class Result:
    gateway: str
    status: str          # "OK", "FAILED", "SKIPPED"
    latency_s: Optional[float] = None
    model_used: Optional[str] = None
    answer: Optional[str] = None
    detail: Optional[str] = None  # error message or skip reason


def _snippet(text: str, length: int = 80) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= length else text[: length - 3] + "..."


def run_litellm() -> Result:
    name = "LiteLLM"
    if not os.environ.get("OPENAI_API_KEY"):
        return Result(name, "SKIPPED", detail="OPENAI_API_KEY not set")
    try:
        from litellm import completion
        start = time.perf_counter()
        response = completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": PROMPT}],
            fallbacks=["claude-sonnet-5"],
        )
        latency = time.perf_counter() - start
        return Result(name, "OK", latency, response.model, response.choices[0].message.content)
    except Exception as e:  # noqa: BLE001
        return Result(name, "FAILED", detail=str(e))


def run_bifrost() -> Result:
    name = "Bifrost"
    try:
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:8080/openai", api_key="dummy-key")
        primary, fallback = "openai/gpt-4o-mini", "anthropic/claude-sonnet-4"
        start = time.perf_counter()
        model_used = primary
        try:
            response = client.chat.completions.create(
                model=primary, messages=[{"role": "user", "content": PROMPT}]
            )
        except Exception:
            model_used = fallback
            response = client.chat.completions.create(
                model=fallback, messages=[{"role": "user", "content": PROMPT}]
            )
        latency = time.perf_counter() - start
        return Result(name, "OK", latency, model_used, response.choices[0].message.content)
    except Exception as e:  # noqa: BLE001
        return Result(name, "FAILED", detail=f"is Bifrost running on :8080? ({e})")


def run_portkey() -> Result:
    name = "Portkey"
    required = ["PORTKEY_API_KEY", "PORTKEY_OPENAI_VIRTUAL_KEY"]
    if any(not os.environ.get(v) for v in required):
        return Result(name, "SKIPPED", detail=f"{required} not fully set")
    try:
        from portkey_ai import Portkey
        client = Portkey(
            api_key=os.environ["PORTKEY_API_KEY"],
            virtual_key=os.environ["PORTKEY_OPENAI_VIRTUAL_KEY"],
        )
        start = time.perf_counter()
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": PROMPT}]
        )
        latency = time.perf_counter() - start
        return Result(name, "OK", latency, "gpt-4o-mini", response.choices[0].message.content)
    except Exception as e:  # noqa: BLE001
        return Result(name, "FAILED", detail=str(e))


def run_openrouter() -> Result:
    name = "OpenRouter"
    if not os.environ.get("OPENROUTER_API_KEY"):
        return Result(name, "SKIPPED", detail="OPENROUTER_API_KEY not set")
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        model = "openai/gpt-4o-mini"
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": PROMPT}]
        )
        latency = time.perf_counter() - start
        return Result(name, "OK", latency, model, response.choices[0].message.content)
    except Exception as e:  # noqa: BLE001
        return Result(name, "FAILED", detail=str(e))


def print_table(results: list[Result]) -> None:
    headers = ["Gateway", "Status", "Latency (s)", "Model used", "Response"]
    rows = []
    for r in results:
        rows.append([
            r.gateway,
            r.status,
            f"{r.latency_s:.2f}" if r.latency_s is not None else "-",
            r.model_used or "-",
            _snippet(r.answer) if r.answer else (r.detail or "-"),
        ])

    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    line = "-+-".join("-" * w for w in widths)

    def fmt(cols):
        return " | ".join(c.ljust(w) for c, w in zip(cols, widths))

    print(fmt(headers))
    print(line)
    for row in rows:
        print(fmt(row))


if __name__ == "__main__":
    print(f'Prompt: "{PROMPT}"\n')
    all_results = [run_litellm(), run_bifrost(), run_portkey(), run_openrouter()]
    print_table(all_results)
