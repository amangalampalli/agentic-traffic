#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE_DIR="${1:-/tmp/agentic-traffic-openenv-push}"
REPO_ID="${2:-Aditya2162/agentic-traffic}"

"${ROOT_DIR}/scripts/prepare_openenv_push.sh" "${STAGE_DIR}"

cd "${STAGE_DIR}"
openenv push . --repo-id "${REPO_ID}"

cd "${ROOT_DIR}"
echo "Returned to ${ROOT_DIR}"
