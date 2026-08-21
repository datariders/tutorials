"""
Intent-routing agent.

Receives an end-user query, classifies its intent, and forwards it to the
Bifrost gateway with a model string that selects the backend provider
(Amazon Bedrock or the native Anthropic API).

Run locally:
    uvicorn app:app --reload --port 8000
"""
import os
import re
import time

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BIFROST_URL = os.environ.get("BIFROST_URL", "http://localhost:8080")
VIRTUAL_KEY = os.environ.get("BIFROST_VIRTUAL_KEY", "workshop-team-a")

app = FastAPI(title="Intent-routing agent")


class ChatRequest(BaseModel):
    query: str


# --- Intent classification -------------------------------------------------
#
# Keyword-based classification keeps the workshop fast and free to run. In a
# real system you'd likely swap this for a small classifier model, or a cheap
# LLM call through the gateway itself (e.g. model="bedrock/anthropic.claude-haiku...").
#
# Each intent maps to a Bifrost model string. The provider prefix
# ("bedrock/" vs "anthropic/") is what tells Bifrost which backend to use.

CODE_PATTERNS = re.compile(
    r"\b(function|class|bug|error|traceback|refactor|regex|sql|python|"
    r"javascript|api|code|script|algorithm|compile)\b",
    re.IGNORECASE,
)
SUMMARIZE_PATTERNS = re.compile(
    r"\b(summarize|summary|tl;?dr|shorten|condense|key points)\b",
    re.IGNORECASE,
)
CREATIVE_PATTERNS = re.compile(
    r"\b(story|poem|write a|brainstorm|creative|slogan|tagline|imagine)\b",
    re.IGNORECASE,
)

# Intent -> (label, model string sent to Bifrost)
INTENT_ROUTES = {
    "code": ("code", "bedrock/anthropic.claude-sonnet-4-6-v1:0"),
    "summarize": ("summarize", "bedrock/anthropic.claude-sonnet-4-6-v1:0"),
    "creative": ("creative", "anthropic/claude-sonnet-5"),
    "general_reasoning": ("general_reasoning", "anthropic/claude-sonnet-5"),
}


def classify_intent(query: str) -> str:
    if CODE_PATTERNS.search(query):
        return "code"
    if SUMMARIZE_PATTERNS.search(query):
        return "summarize"
    if CREATIVE_PATTERNS.search(query):
        return "creative"
    return "general_reasoning"


# --- Gateway call ------------------------------------------------------------

async def call_bifrost(model: str, query: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "max_tokens": 512,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {VIRTUAL_KEY}",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{BIFROST_URL}/v1/chat/completions", json=payload, headers=headers
        )
    if resp.status_code == 429:
        raise HTTPException(status_code=429, detail="Rate limit or budget exceeded")
    resp.raise_for_status()
    return resp.json()


@app.post("/chat")
async def chat(req: ChatRequest):
    intent = classify_intent(req.query)
    label, model = INTENT_ROUTES[intent]

    start = time.monotonic()
    result = await call_bifrost(model, req.query)
    elapsed_ms = round((time.monotonic() - start) * 1000)

    answer = (
        result.get("choices", [{}])[0].get("message", {}).get("content", "")
    )
    cache_hit = result.get("bifrost", {}).get("cache_hit", False)

    return {
        "intent": label,
        "routed_to": model,
        "cache_hit": cache_hit,
        "latency_ms": elapsed_ms,
        "answer": answer,
    }


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
