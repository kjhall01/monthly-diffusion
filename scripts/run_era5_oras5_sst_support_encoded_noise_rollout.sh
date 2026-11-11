#!/usr/bin/env bash
# From umd/: bash frozen/scripts/run_era5_oras5_sst_support_encoded_noise_rollout.sh
# Same command resumes both spin-up and physical rollout; never deletes data.
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
OUTPUT="runs/mdv1-era5-oras5-sst-support/long_rollout/ssp370_1850_2100_5member_encoded_noise_spinup1800"

if [[ "$DIAGNOSTICS_ONLY" == false ]]; then
  "$PYTHON" - "$OUTPUT" <<'PY'
from pathlib import Path
import shutil
import sys
directory = Path(sys.argv[1])
completed = len(list(directory.glob('month-*.zarr')))
# 22 MiB/month for all 25 fields + 5 GiB diagnostics + 0.1 GiB spin-up.
needed = max(3012-completed, 0)*22*1024**2 + 5.1*1024**3
free = shutil.disk_usage('.').free
print(f'Disk check: {free/1024**3:.1f} GiB free; {needed/1024**3:.1f} GiB conservative budget', flush=True)
if free < needed:
    raise SystemExit('Insufficient space; explicitly free space before restarting. No files were deleted.')
PY
  # Encode independent field-noise histories [Dec 1799, Nov 1799]. Prefix
  # includes BOTH histories, held at the actual January-1850 prescribed value.
  # Save monthly latents for Jan 1800–Dec 1849, without any physical decoding.
  # Continue unchanged history/noise/RNG into Jan 1850–Dec 2100; no reseeding.
  "$PYTHON" -u examples/forecast.py \
    runs/mdv1-era5-oras5-sst-support/model.pt examples/era5-oras5-local.yaml \
    --output "$OUTPUT" --encoded-noise-start 1800-01 \
    --hold-first-co2-back-to 1799-11 --spinup-months 600 --save-spinup-latents \
    --months 3012 --members 5 --seed 20260716 \
    --device mps --threads 4 --require-trained --resume
fi

# Identical reference/anomaly/plot conventions to the historical baseline.
# Decoder-gain probe included, along with indices, spectra/ACF/distributions,
# teleconnections, training climatology, global trajectories and change maps.
"$PYTHON" -u scripts/diagnose_long_rollout.py \
  examples/diagnostics-era5-oras5-sst-support-encoded-noise.yaml --device mps
echo "Finished encoded-noise initialization + 50-year spin-up + SSP3-7.0 rollout + diagnostics."
