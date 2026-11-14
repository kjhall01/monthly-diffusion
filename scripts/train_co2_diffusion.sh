#!/usr/bin/env bash
# Train only diffusion, reusing the selected CO2-free AEs in the prepared run.
# The Python script enforces its four-hour budget and exports the best model.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON="${PYTHON:-python}"
LOG_DIR="$SCRIPT_DIR/logs/test_gridpoint_co2_cfg"
mkdir -p "$LOG_DIR"

"$PYTHON" -u train_latent_diffusion.py 2>&1 | tee -a "$LOG_DIR/diffusion.log"
