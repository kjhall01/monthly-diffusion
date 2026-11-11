#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$ROOT/.matplotlib"
export OMP_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
mkdir -p "$MPLCONFIGDIR"
PYTHON="${MD_PYTHON:-python}"
WHICH=both
case "${1:-}" in
  --trajectory-only) WHICH=trajectory; shift ;;
  --control-only) WHICH=control; shift ;;
  "") ;;
  *) echo "Usage: bash $0 [--trajectory-only|--control-only]" >&2; exit 2 ;;
esac
if [[ $# -ne 0 ]]; then echo "Unexpected additional arguments" >&2; exit 2; fi

# Conservatively budget 22 MiB per five-member monthly shard plus 5 GiB for
# diagnostics/headroom. Existing completed shards count toward this budget.
"$PYTHON" - "$WHICH" <<'PY'
from pathlib import Path
import shutil
import sys
scenarios = ['ssp370','constant_initial_co2']
if sys.argv[1] == 'trajectory': scenarios = scenarios[:1]
if sys.argv[1] == 'control': scenarios = scenarios[1:]
root = Path('runs/mdv1-era5-oras5/long_rollout')
remaining = 0
for scenario in scenarios:
    directory = root/f'{scenario}_1850_2100_5member_gaussian_spinup1845'
    remaining += max(3012-len(list(directory.glob('month-*.zarr'))),0)
needed = remaining*22*1024**2 + 5*1024**3
free = shutil.disk_usage('.').free
print(f'Disk check: {free/1024**3:.1f} GiB free; {needed/1024**3:.1f} GiB conservative budget',flush=True)
if free < needed:
    raise SystemExit('Insufficient space for the selected workflow. Use --trajectory-only for just SSP3-7.0, '
                     'or free space explicitly before running both. No files were deleted.')
PY

# Both calls validate checkpoint/forcing/initial-state identity even when the
# archive is complete. Never skip this validation merely by counting shards.
# IID normalized history -> Jan 1845–Dec 1849 latent-only spin-up; retained
# decoded fields cover Jan 1850–Dec 2100 (3012 months).
for scenario in ssp370 constant_initial_co2; do
  if [[ "$WHICH" == trajectory && "$scenario" != ssp370 ]]; then continue; fi
  if [[ "$WHICH" == control && "$scenario" == ssp370 ]]; then continue; fi
  options=(--resume)  # Nonempty also works with macOS Bash 3.2 + nounset.
  if [[ "$scenario" == "constant_initial_co2" ]]; then
    options+=(--constant-initial-co2)
  fi
  "$PYTHON" -u examples/forecast.py \
    runs/mdv1-era5-oras5/model.pt examples/era5-oras5-local.yaml \
    --output "runs/mdv1-era5-oras5/long_rollout/${scenario}_1850_2100_5member_gaussian_spinup1845" \
    --gaussian-start 1845-01 --spinup-months 60 --hold-first-co2-back-to 1845-01 \
    --months 3012 --members 5 --seed 20260716 \
    --device mps --threads 4 --require-trained "${options[@]}"
  # Complete each scenario's diagnostics before beginning the next archive.
  if [[ "$scenario" == "ssp370" ]]; then
    bash scripts/diagnose_era5_oras5_long_rollout.sh --trajectory-only
  else
    bash scripts/diagnose_era5_oras5_long_rollout.sh --control-only
  fi
done
echo "Finished ERA5/ORAS5 selected rollout(s) + diagnostics."
