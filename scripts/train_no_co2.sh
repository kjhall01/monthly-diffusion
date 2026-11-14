#!/usr/bin/env bash
# Train one fresh run in order. Edit the ordinary Python scripts for settings.
# First align all three RUN_DIRs; set diffusion CONDITION_ON_CO2=False for the
# original CO2-free ablation. Its current defaults select a separate CFG run.
# Each script has TIME_LIMIT_HOURS = 4: stop training at a batch boundary, then
# finish exporting its best model and (for AEs) latent cache before advancing.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON="${PYTHON:-python}"
LOG_DIR="$SCRIPT_DIR/logs/test_gridpoint_no_co2"
mkdir -p "$LOG_DIR"

echo "Atmosphere AE: up to 4 hours of setup/training, then export"
"$PYTHON" -u train_atmosphere_autoencoder.py 2>&1 | tee -a "$LOG_DIR/atmosphere.log"

echo "Ocean AE: up to 4 hours of setup/training, then export"
"$PYTHON" -u train_ocean_autoencoder.py 2>&1 | tee -a "$LOG_DIR/ocean.log"

echo "Diffusion: up to 4 hours of setup/training, then export"
"$PYTHON" -u train_latent_diffusion.py 2>&1 | tee -a "$LOG_DIR/diffusion.log"

echo "Finished. Models are in the RUN_DIR selected by each training script."
echo "Console logs: $LOG_DIR"
