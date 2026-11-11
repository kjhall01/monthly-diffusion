#!/usr/bin/env bash
set -euo pipefail

# Full SSP3-7.0 long rollout and the standard frozen-package diagnostics for
# the completed ocean-mask-aware 1-degree model.  The forecast writer is
# resumable: rerunning this script continues from the last complete monthly
# shard after an interruption.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${MD_PYTHON:-python}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
CHECKPOINT="runs/mdv1-e3smv2-r1i1p1f1-ocean-mask-aware/model.pt"
DATA_YAML="examples/e3sm-local.yaml"
FORECAST_DIR="runs/mdv1-e3smv2-r1i1p1f1-ocean-mask-aware/long_rollout/ssp370_1850_2100_5member"
DIAGNOSTICS_DIR="analysis/long_rollout_ocean_mask_aware_epoch176/full_suite"

# Initialization uses January and February 1850 as the required two-month
# history; 3010 generated months then cover March 1850 through December 2100.
completed_shards=0
if [[ -d "$FORECAST_DIR" ]]; then
  completed_shards="$(find "$FORECAST_DIR" -maxdepth 1 -type d -name 'month-*.zarr' | wc -l | tr -d ' ')"
fi
if [[ "$completed_shards" == "3010" ]]; then
  echo "Reusing complete forecast archive ($completed_shards monthly shards)."
else
  "$PYTHON" -u examples/forecast.py \
    "$CHECKPOINT" "$DATA_YAML" \
    --output "$FORECAST_DIR" \
    --months 3010 \
    --members 5 \
    --initialization 1850-02 \
    --resume
fi

# The complete bounded-memory suite has a diagnostics-only entry point too.
bash scripts/diagnose_ocean_mask_aware_long_rollout.sh

echo "Completed forecast and diagnostics:"
echo "  forecast:    $ROOT/$FORECAST_DIR"
echo "  diagnostics: $ROOT/$DIAGNOSTICS_DIR"
