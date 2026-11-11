#!/usr/bin/env bash
# Fresh staged training; append --resume after an interruption.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec "${MD_PYTHON:-python}" -u scripts/legacy/train_model.py examples/e3sm-local.yaml \
  --output runs/mdv1-e3smv2-r1i1p1f1 --device mps --threads 4 --load-in-memory "$@"
