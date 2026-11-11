#!/usr/bin/env bash
# Continue the real-Jan/Feb-1958-initialized five members, then update the plot.
# Repeating this script resumes, never regenerates completed months.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
export MPLBACKEND=Agg MPLCONFIGDIR="$ROOT/.matplotlib"
PYTHON="${MD_PYTHON:-python}"
SOURCE="runs/mdv1-era5-oras5-sst-support/long_rollout/observed_init195802_195803_202512_5member"
OUTPUT="runs/mdv1-era5-oras5-sst-support/long_rollout/observed_init195802_195803_210012_5member"
if [[ ! -e "$OUTPUT" ]]; then
  # APFS copy-on-write clone: retain the old archive and diagnostics without
  # allocating another physical copy of its 16 GiB of immutable field chunks.
  cp -cR "$SOURCE" "$OUTPUT"
fi
"$PYTHON" -u examples/forecast.py \
  runs/mdv1-era5-oras5-sst-support/model.pt examples/era5-oras5-local.yaml \
  --output "$OUTPUT" --initialization 1958-02 \
  --months 1714 --members 5 --seed 20260716 \
  --device mps --threads 4 --require-trained --resume --extend \
  2>&1 | tee -a "$OUTPUT/extension.log"
cd "$ROOT/.."
"$PYTHON" -u frozen/scripts/plot_global_trajectories_comparison.py \
  frozen/examples/global-trajectories-comparison-through2100.yaml
