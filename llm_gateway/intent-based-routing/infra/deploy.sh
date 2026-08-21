#!/usr/bin/env bash
#
# End-to-end deploy: builds + pushes the Bifrost and agent images to ECR,
# deploys the CloudFormation stack, waits for it to stabilize, and runs a
# smoke test against the live agent.
#
# Usage:
#   export ANTHROPIC_API_KEY=sk-ant-...
#   ./deploy.sh [aws-region] [project-name]
#
set -euo pipefail

REGION="${1:-us-east-1}"
PROJECT_NAME="${2:-llm-gateway-workshop}"
STACK_NAME="${PROJECT_NAME}-stack"

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "Error: export ANTHROPIC_API_KEY before running this script." >&2
  exit 1
fi

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_HOST="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
BIFROST_REPO="${PROJECT_NAME}-bifrost"
AGENT_REPO="${PROJECT_NAME}-agent"
BIFROST_URI="${ECR_HOST}/${BIFROST_REPO}:latest"
AGENT_URI="${ECR_HOST}/${AGENT_REPO}:latest"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Creating ECR repositories (skipping any that already exist)"
aws ecr create-repository --repository-name "$BIFROST_REPO" --region "$REGION" >/dev/null 2>&1 || true
aws ecr create-repository --repository-name "$AGENT_REPO" --region "$REGION" >/dev/null 2>&1 || true

echo "==> Logging in to ECR"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR_HOST"

echo "==> Building and pushing the Bifrost image (config baked in)"
docker build -f "$REPO_ROOT/infra/bifrost.Dockerfile" -t "$BIFROST_URI" "$REPO_ROOT"
docker push "$BIFROST_URI"

echo "==> Building and pushing the agent image"
docker build -t "$AGENT_URI" "$REPO_ROOT/agent"
docker push "$AGENT_URI"

echo "==> Deploying the CloudFormation stack (this provisions the VPC, ALB, ECS cluster, and both services)"
aws cloudformation deploy \
  --region "$REGION" \
  --stack-name "$STACK_NAME" \
  --template-file "$REPO_ROOT/infra/cloudformation.yaml" \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
      ProjectName="$PROJECT_NAME" \
      AnthropicApiKey="$ANTHROPIC_API_KEY" \
      BifrostImageUri="$BIFROST_URI" \
      AgentImageUri="$AGENT_URI"

echo "==> Reading stack outputs"
BIFROST_URL="$(aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='BifrostURL'].OutputValue" --output text)"
AGENT_URL="$(aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='AgentURL'].OutputValue" --output text)"

echo ""
echo "Bifrost gateway: $BIFROST_URL"
echo "Intent agent:    $AGENT_URL"
echo ""
echo "==> Waiting for the agent service to stabilize"
aws ecs wait services-stable --region "$REGION" \
  --cluster "${PROJECT_NAME}-cluster" --services "${PROJECT_NAME}-stack-AgentService*" 2>/dev/null || true

echo "==> Smoke test: /healthz"
curl -sf "${AGENT_URL}/healthz" && echo " OK" || echo " (not ready yet — services can take a couple of minutes to register with the ALB; retry shortly)"

echo "==> Smoke test: /chat"
curl -s -X POST "${AGENT_URL}/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "Write a Python function that reverses a linked list"}' | python3 -m json.tool || true

echo ""
echo "Done. To tear everything down: aws cloudformation delete-stack --region $REGION --stack-name $STACK_NAME"
