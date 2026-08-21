# Lab exercises

Work through these in order. Each lab ends with a checkpoint you can verify before moving
on. File paths referenced below are relative to the starter kit root.

---

## Lab 0 — Local dry run (do this before touching AWS)

1. `cd infra && docker compose up --build`
   This starts Bifrost (port 8080) and the agent (port 8000) locally, wired together.
2. Set your keys first (see `.env.example` — copy to `.env`):
   - `ANTHROPIC_API_KEY` — your Anthropic API key
   - `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` (or run `aws configure` and mount
     `~/.aws`) — only needed if you want the Bedrock path working locally too
3. Test the agent directly:
   ```
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"query": "Write a Python function that reverses a linked list"}'
   ```
4. **Checkpoint**: response includes `"routed_to"` showing which provider/model handled
   it, and `"intent"` showing the classified category.

---

## Lab 1 — Deploy Bifrost to ECS Fargate

### Fast path: one-command deploy

`infra/deploy.sh` builds both images, pushes them to ECR, and deploys
`infra/cloudformation.yaml` — a self-contained stack that provisions the VPC, ALB, ECS
cluster, IAM roles, Secrets Manager secret, and both services (Bifrost + agent), then
runs a smoke test.

```
export ANTHROPIC_API_KEY=sk-ant-your-key-here
cd infra
./deploy.sh us-east-1 llm-gateway-workshop
```

It prints the Bifrost and agent URLs when done. Skip straight to **Lab 2** afterward, or
read the manual steps below first if you want to understand — or customize — each piece.

**Note on the template's networking**: to avoid NAT gateway cost during the workshop, the
stack runs both services in public subnets with an internet-facing ALB open on ports 8000
and 8080 to `0.0.0.0/0`. That's fine for a few hours of workshop traffic but is not how
you'd run this in production — restrict the ALB security group to known CIDR ranges (or
put it behind a VPN/private ALB) before using this pattern for anything real, the same
way the AWS blog post's reference architecture keeps the gateway on a private network
reached only through your corporate network.

To tear everything down afterward: `aws cloudformation delete-stack --stack-name llm-gateway-workshop-stack`.

### Manual path (to understand each step, or if you're customizing the infrastructure)

1. Push the Bifrost image (or use `maximhq/bifrost:latest` directly — no build needed)
   to ECR if your org requires images to come from your own registry:
   ```
   aws ecr create-repository --repository-name bifrost-gateway
   docker pull maximhq/bifrost:latest
   docker tag maximhq/bifrost:latest <account-id>.dkr.ecr.<region>.amazonaws.com/bifrost-gateway:latest
   aws ecr get-login-password | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
   docker push <account-id>.dkr.ecr.<region>.amazonaws.com/bifrost-gateway:latest
   ```
2. Create the Secrets Manager secret for the Anthropic API key:
   ```
   aws secretsmanager create-secret --name bifrost/anthropic-api-key \
     --secret-string '{"ANTHROPIC_API_KEY":"sk-ant-..."}'
   ```
3. Create the IAM task role (`infra/iam-task-role-trust-policy.json` +
   `infra/iam-task-role-policy.json` in the starter kit) that grants `bedrock:InvokeModel`
   and `bedrock:InvokeModelWithResponseStream`, plus `secretsmanager:GetSecretValue` for
   the secret above.
4. Register the task definition and create the service:
   ```
   aws ecs register-task-definition --cli-input-json file://infra/ecs-task-definition.json
   aws ecs create-service --cluster workshop-cluster --service-name bifrost-gateway \
     --task-definition bifrost-gateway --desired-count 1 --launch-type FARGATE \
     --network-configuration file://infra/network-config.json \
     --load-balancers file://infra/load-balancer-config.json
   ```
5. Confirm the provider config in `bifrost/config.json` matches your Region and Bedrock
   model IDs, and that it's mounted or baked into the task (see comments in the file).
6. **Checkpoint**: `curl http://<alb-dns-name>:8080/v1/chat/completions` (with a dummy
   virtual key header) returns a response from Bedrock, not a connection error.

---

## Lab 2 — Point the agent at the deployed gateway

**If you used `deploy.sh` in Lab 1**, this is already done — the CloudFormation template
wires the agent's `BIFROST_URL` to the ALB automatically. Skip to step 3.

1. Update the agent's `BIFROST_URL` environment variable to the ALB DNS name from Lab 1.
2. Deploy the agent the same way (its own small ECS service, or run it locally against
   the remote gateway for the workshop — either works; `infra/ecs-task-definition-agent.json`
   is provided for the ECS path).
3. Send a few varied queries and confirm each is routed to the intended provider:
   - A code question → Bedrock
   - A general/creative question → Anthropic direct API
4. **Checkpoint**: for each query, the `routed_to` field in the response matches the
   intent you expected, and CloudWatch (or Bifrost's own logs) shows the corresponding
   upstream call.

---

## Lab 3 — Rate limiting and tokenomics

1. Open `bifrost/config.json` → `governance` section. Note the two virtual keys already
   defined: `workshop-team-a` and `workshop-team-b`, each with its own request/token
   limits and a monthly budget.
2. Send requests using `team-a`'s virtual key in a tight loop (a simple shell loop of 10+
   requests in a few seconds is enough) and observe the gateway start returning
   `429 Too Many Requests` once the limit is hit.
3. Check Bifrost's usage/cost view (built-in web UI at `http://<alb-dns-name>:8080`, or
   the `/metrics` Prometheus endpoint) to see tokens and estimated cost attributed to
   each virtual key.
4. Lower `team-b`'s monthly budget to a small number, send a few requests, and observe
   the gateway block further calls once the budget is exhausted.
5. **Checkpoint**: you can show, live, one virtual key hitting its rate limit and a
   different one hitting its budget cap — while a third, unaffected key keeps working.

---

## Lab 4 — Semantic caching

1. In `bifrost/config.json`, confirm the `cache` block is enabled with a similarity
   threshold (start around `0.92`).
2. Send the same question twice in a row — note the response time and check the response
   metadata for a cache-hit indicator.
3. Send a *reworded* version of the same question (same meaning, different words) and
   confirm it still hits the cache.
4. Send a clearly different question and confirm it does **not** hit the cache.
5. **Checkpoint**: attendees can point to the latency difference between a cache hit and
   a cache miss, and explain in their own words when semantic caching helps vs. when it
   risks returning a stale or wrong-context answer.

---

## Lab 5 (stretch) — Load test and observe it all together

1. Use a simple script (`infra/loadtest.py` in the starter kit) to fire a mix of
   query types at the agent concurrently.
2. Watch, at the same time: which provider each request lands on, which requests get
   cached, and which get rate-limited — tying together Labs 2–4.
3. Optional: wire Bifrost's OTLP output to CloudWatch or Amazon Managed Prometheus and
   build a one-panel dashboard showing requests by provider and by virtual key.
