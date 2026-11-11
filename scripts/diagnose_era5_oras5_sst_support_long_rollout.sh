#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$ROOT/.matplotlib"
mkdir -p "$MPLCONFIGDIR"
PYTHON="${MD_PYTHON:-python}"
WHICH="both"
if [[ "${1:-}" == "--trajectory-only" ]]; then WHICH=trajectory; shift; fi
if [[ "${1:-}" == "--control-only" ]]; then WHICH=control; shift; fi
if [[ "$WHICH" != "control" ]]; then
  "$PYTHON" -u scripts/diagnose_long_rollout.py examples/diagnostics-era5-oras5-sst-support.yaml --device mps "$@"
fi
if [[ "$WHICH" != "trajectory" ]]; then
  "$PYTHON" -u scripts/diagnose_long_rollout.py examples/diagnostics-era5-oras5-sst-support.yaml \
    --forecast runs/mdv1-era5-oras5-sst-support/long_rollout/constant_initial_co2_1850_2100_5member_gaussian_spinup1845 \
    --output runs/mdv1-era5-oras5-sst-support/long_rollout/constant_initial_co2_1850_2100_5member_gaussian_spinup1845/diagnostics \
    --scenario-label 'Constant-initial-CO2 control' --skip-decoder "$@"
fi
