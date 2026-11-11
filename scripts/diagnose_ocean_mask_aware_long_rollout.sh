#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$ROOT/.matplotlib"
mkdir -p "$MPLCONFIGDIR"
"${MD_PYTHON:-python}" -u scripts/diagnose_long_rollout.py \
  examples/diagnostics-ocean-mask-aware.yaml "$@"
