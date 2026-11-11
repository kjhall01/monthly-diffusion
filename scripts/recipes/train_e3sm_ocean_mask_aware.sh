#!/usr/bin/env bash
# Fresh 1-degree mask-aware ocean-ECC training; append --resume after interruption.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec "${MD_PYTHON:-python}" -u scripts/legacy/train_model.py examples/e3sm-local.yaml \
  --config examples/model-ocean-mask-aware.yaml \
  --output runs/mdv1-e3smv2-r1i1p1f1-ocean-mask-aware \
  --device mps --threads 4 --load-in-memory "$@"
