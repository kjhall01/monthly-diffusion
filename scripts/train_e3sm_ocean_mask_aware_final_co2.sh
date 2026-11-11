#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${MD_PYTHON:-python}"

"$PYTHON" -u examples/train_model.py \
  examples/e3sm-local.yaml \
  --config examples/model-ocean-mask-aware-final-co2.yaml \
  --output runs/mdv1-e3smv2-r1i1p1f1-ocean-mask-aware-final-co2 \
  --device mps \
  --threads 4 \
  --load-in-memory \
  "$@"
