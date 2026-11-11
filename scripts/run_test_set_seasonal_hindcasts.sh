#!/usr/bin/env bash
# Run/resume both current-model DJF hindcasts, scores and figures on MPS.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/frozen${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mdv1-seasonal-mpl"
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
python -u frozen/scripts/test_set_seasonal_hindcasts.py --stage all --device mps "$@"
