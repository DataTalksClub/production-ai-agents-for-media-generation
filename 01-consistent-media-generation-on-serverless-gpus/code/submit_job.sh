#!/usr/bin/env bash
set -euo pipefail
: "${WORKER_IMAGE:?Set your published worker image digest}"
: "${BUCKET:?Set an existing bucket name}"
: "${RUN_ID:?Set a unique run ID using letters, digits, underscores or hyphens}"
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid RUN_ID' >&2; exit 1; }
[[ "$BUCKET" =~ ^[a-z0-9][a-z0-9.-]+$ ]] || { echo 'Invalid BUCKET' >&2; exit 1; }
[[ "$WORKER_IMAGE" == *@sha256:* ]] || { echo 'Use a worker image digest' >&2; exit 1; }
# The documented minimum is 1h; this is a failure bound, NOT a spend cap.
# --dry-run validates the request without launching compute.
extra=()
if [[ "${1:-}" == "--dry-run" ]]; then extra+=(--dry-run); fi
nebius ai job create \
  --name "m1-${RUN_ID}" --image "$WORKER_IMAGE" \
  --platform gpu-l40s-a --preset 1gpu-8vcpu-32gb --on-demand \
  --timeout "${JOB_TIMEOUT:-1h}" --restart-policy never \
  --volume "s3://${BUCKET}:/outputs:rw" \
  --env "WORKER_IMAGE=${WORKER_IMAGE}" --env GPU_PRESET=1gpu-8vcpu-32gb \
  --container-command python \
  --args "/app/generate.py --config /app/generation.json --n 3 --out /outputs/runs/01-consistent-media-generation-on-serverless-gpus/${RUN_ID}" \
  "${extra[@]}"
# Copy the Job ID from the output. Observe in console or with `nebius ai job get ID`.
# Completion is not enough: verify PNGs and metrics.json exist in the bucket.
