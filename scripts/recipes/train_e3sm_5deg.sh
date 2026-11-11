#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
exec "${MD_PYTHON:-python}" -u scripts/legacy/train_model.py \
  examples/e3sm-5deg-local.yaml \
  --output runs/mdv1-e3smv2-r1i1p1f1-5deg-m3m2 \
  --config examples/model-5deg.yaml \
  --device mps \
  --threads 4 \
  --load-in-memory "$@"
