#!/usr/bin/env bash
# Matched mask-aware 1-degree model, freshly trained on ERA5/ORAS5.
# Append --resume after an interruption. This does not launch long rollouts.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${MD_PYTHON:-python}" -u examples/train_model.py examples/era5-oras5-local.yaml \
  --config examples/model-era5-oras5-sst-support.yaml \
  --output runs/mdv1-era5-oras5-sst-support \
  --device mps --threads 4 --load-in-memory "$@"
