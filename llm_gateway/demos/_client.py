"""Shared helper: an OpenAI SDK client pointed at the LiteLLM gateway.

The whole point of a gateway is that your app code stays vanilla OpenAI-SDK.
You only change the base_url and use the gateway's master key instead of a
real provider key. Everything else (routing, fallbacks, caching, cost
tracking) happens server-side in the gateway.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:4000")
MASTER_KEY = os.environ.get("LITELLM_MASTER_KEY", "sk-demo-master-1234")


def get_client() -> OpenAI:
    """Return an OpenAI client that talks to the LiteLLM gateway."""
    return OpenAI(base_url=GATEWAY_URL, api_key=MASTER_KEY)
