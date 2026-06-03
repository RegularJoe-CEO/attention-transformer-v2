#!/usr/bin/env bash
# Deprecated: use runpod_fresh_deploy_v2.sh (v1 repo path caused pod confusion).
set -euo pipefail
echo "runpod_fresh_deploy.sh is deprecated — use scripts/runpod_fresh_deploy_v2.sh"
exec "$(dirname "$0")/runpod_fresh_deploy_v2.sh" "$@"