#!/usr/bin/env bash
# Sync attention-transformer-v2 to a RunPod SSH target and run GPU tests.
#
# Usage (from your Mac):
#   export RUNPOD_SSH='root@YOUR_POD_IP -p YOUR_SSH_PORT'
#   bash scripts/sync_and_test_runpod.sh
#
# Or one line:
#   RUNPOD_SSH='root@1.2.3.4 -p 22' bash scripts/sync_and_test_runpod.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="${REMOTE_DIR:-/workspace/attention-transformer-v2}"

if [[ -z "${RUNPOD_SSH:-}" ]]; then
  echo "ERROR: Set RUNPOD_SSH to your pod SSH target."
  echo "  Example: export RUNPOD_SSH='root@123.45.67.89 -p 12345'"
  echo "  Find it in RunPod: Pod → Connect → SSH over exposed TCP"
  echo ""
  echo "Or run manually ON the pod web terminal:"
  echo "  bash scripts/runpod_v2_test.sh"
  echo "  (after uploading/cloning the repo to $REMOTE_DIR)"
  exit 1
fi

# shellcheck disable=SC2086
RSYNC_SSH="ssh $RUNPOD_SSH"

echo "Syncing $ROOT -> $RUNPOD_SSH:$REMOTE_DIR"
# shellcheck disable=SC2086
ssh $RUNPOD_SSH "mkdir -p $REMOTE_DIR"
# shellcheck disable=SC2086
rsync -az --delete -e "ssh $RUNPOD_SSH" \
  --exclude target --exclude .git \
  "$ROOT/" "$RUNPOD_SSH:$REMOTE_DIR/"

echo "Running runpod_v2_test.sh on pod..."
# shellcheck disable=SC2086
ssh $RUNPOD_SSH "chmod +x $REMOTE_DIR/scripts/runpod_v2_test.sh && INSTALL_DIR=$REMOTE_DIR bash $REMOTE_DIR/scripts/runpod_v2_test.sh"

echo "Done. Fetch log from pod: $REMOTE_DIR/runpod_v2_test_*.log"