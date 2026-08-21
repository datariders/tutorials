# Bifrost configuration notes

`config.json` is the single file Bifrost reads at startup. A few notes since JSON
doesn't support comments:

- **providers** — two backends: `bedrock-claude` (auth via the ECS task's IAM role, no
  static credentials) and `anthropic-direct` (auth via an API key pulled from the
  `ANTHROPIC_API_KEY` environment variable, which should be sourced from Secrets Manager
  in the task definition, not hardcoded).
- **governance.virtual_keys** — two example teams with different budgets and rate limits,
  used in Lab 3 to demonstrate both a rate-limit rejection and a budget cutoff. Adjust the
  numbers if you want the demo to trip faster or slower during a live session.
- **governance.provider_config** — a limit applied at the provider level regardless of
  which virtual key is calling, useful for staying under your own Bedrock service quota.
- **cache** — semantic caching, matches on meaning rather than exact text.
  `similarity_threshold` closer to 1.0 = stricter matching, fewer (but safer) cache hits.
- **telemetry** — set `OTLP_ENDPOINT` to a collector (e.g. an ADOT collector sidecar
  forwarding to CloudWatch, or Amazon Managed Service for Prometheus) if you want Lab 5's
  dashboard. Safe to leave unset for Labs 0–4.

Before deploying, replace the Bedrock `region` and model ID with what's actually enabled
in your account, and confirm the Anthropic model names match what's currently available
on your Anthropic API key.
