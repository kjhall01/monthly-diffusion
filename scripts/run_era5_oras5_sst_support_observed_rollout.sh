#!/usr/bin/env bash
# From umd/: bash frozen/scripts/run_era5_oras5_sst_support_observed_rollout.sh
# Repeat the same command to resume. --diagnostics-only reruns/resumes plots.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$ROOT/.matplotlib"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
mkdir -p "$MPLCONFIGDIR"
PYTHON="${MD_PYTHON:-python}"
DIAGNOSTICS_ONLY=false
case "${1:-}" in
  --diagnostics-only) DIAGNOSTICS_ONLY=true; shift ;;
  "") ;;
  *) echo "Usage: bash $0 [--diagnostics-only]" >&2; exit 2 ;;
esac
if [[ $# -ne 0 ]]; then echo "Unexpected additional arguments" >&2; exit 2; fi
OUTPUT="runs/mdv1-era5-oras5-sst-support/long_rollout/observed_init195802_195803_202512_5member"

if [[ "$DIAGNOSTICS_ONLY" == false ]]; then
  # 68*12 - two observed history months = 814 generated months. Check storage
  # without deleting anything; existing shards are validated by forecast.py.
  "$PYTHON" - "$OUTPUT" <<'PY'
from pathlib import Path
import shutil
import sys
directory = Path(sys.argv[1])
completed = len(list(directory.glob('month-*.zarr')))
needed = max(814-completed, 0)*22*1024**2 + 5*1024**3
free = shutil.disk_usage('.').free
print(f'Disk check: {free/1024**3:.1f} GiB free; {needed/1024**3:.1f} GiB conservative budget', flush=True)
if free < needed:
    raise SystemExit('Insufficient space; explicitly free space before restarting. No files were deleted.')
PY
  # Current = Feb 1958, previous = Jan 1958. The common Forecast iterator
  # encodes the two real states once and supplies [current, previous] to BOTH
  # components. Members share the initial history, not their future noises.
  # Same prescribed annual-interpolated CO2, sampler, seed, AR(1) timescales,
  # preprocessing and SST support as the existing projection workflow.
  "$PYTHON" -u examples/forecast.py \
    runs/mdv1-era5-oras5-sst-support/model.pt examples/era5-oras5-local.yaml \
    --output "$OUTPUT" --initialization 1958-02 \
    --months 814 --members 5 --seed 20260716 \
    --device mps --threads 4 --require-trained --resume
fi

# Indices, distributions, spectra, lag ACF, phase locking, teleconnections,
# training climatologies, global trajectories and within-coverage change maps.
# No extra counterfactual decoder-gain sweep: it depends on the unchanged
# checkpoint rather than the chosen initial state and is a separate analysis.
"$PYTHON" -u scripts/diagnose_long_rollout.py \
  examples/diagnostics-era5-oras5-sst-support-observed.yaml --skip-decoder
echo "Finished observation-initialized ERA5/ORAS5 rollout and diagnostics."
