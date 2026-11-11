"""Controlled cycling-duration sweep; immutable checkpoint, resumable latents.

Use standard-normal *input fields*, encode them, cycle only during an initial
finite interval, then use the unmodified latent transition. Real-data controls
use multiple calendar-matched validation histories, not one chosen ENSO phase.
Only latent trajectories/restarts are archived; finalists can be decoded later.
Run from frozen/: python scripts/experiment_isotropic_cycling_sweep.py
"""

from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from monthly_diffusion import MonthlyDiffusionV1


from monthly_diffusion.experiments.experiment_isotropic_cycling_sweep import (
    OUTPUT,
    CHECKPOINT,
    run,
)

stop_after_years = None

if not torch.backends.mps.is_available():
    raise RuntimeError("MPS required; no silent CPU inference")
torch.set_num_threads(2)
model = MonthlyDiffusionV1.load(CHECKPOINT, device="mps")
base = json.loads(
    (
        ROOT / "analysis/cycled_initialization_ensemble_30years/configuration.json"
    ).read_text()
)
if model.weight_digest() != base["weight_digest"]:
    raise ValueError("Checkpoint changed since original experiments")
run(model, "screen_january", stop_after_years=stop_after_years)
sources = [Path(__file__), ROOT / "monthly_diffusion/forecast.py"]
(OUTPUT / "code_provenance.json").write_text(
    json.dumps(
        {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, indent=2
    )
)
