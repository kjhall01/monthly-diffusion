#!/usr/bin/env bash
# Train atmosphere, then ocean, with the settings in the two Python scripts.
# Each stage has four hours for setup/training, followed by best-model export.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON="${PYTHON:-python}"
LOG_DIR="$SCRIPT_DIR/logs/test_gridpoint_no_transforms"
mkdir -p "$LOG_DIR"

echo "Atmosphere AE: up to 4 hours of setup/training, then export"
"$PYTHON" -u train_atmosphere_autoencoder.py 2>&1 | tee -a "$LOG_DIR/atmosphere.log"

echo "Ocean AE: up to 4 hours of setup/training, then export"
"$PYTHON" -u train_ocean_autoencoder.py 2>&1 | tee -a "$LOG_DIR/ocean.log"

echo "Finished both autoencoders. Run generate_ae_reconstruction.py next."
echo "Console logs: $LOG_DIR"
