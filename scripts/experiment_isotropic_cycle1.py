"""Isotropic versus local-moment Gaussian fields, with ONE AE feedback cycle.

Reuse the five-member ERA5 checkpoint, initial field encodings, monthly noise
stream, and 30-year fixed-CO2 protocol from the saved initialization experiments.
Only the missing isotropic/one-cycle arm is generated. All comparison archives
are read-only. No moments are fitted, no model trained, and no full fields saved.

Run: python scripts/experiment_isotropic_cycle1.py
"""

from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import torch

from monthly_diffusion import MonthlyDiffusionV1


from monthly_diffusion.experiments.experiment_isotropic_cycle1 import (
    OUTPUT,
    ORIGINAL,
    CHECKPOINT,
    run_forecast,
    assemble,
    analyze,
    render,
)

stop_after_years = 30

torch.set_num_threads(2)
config = json.loads((ORIGINAL / "configuration.json").read_text())
if not torch.backends.mps.is_available():
    raise RuntimeError("MPS unavailable; refusing silent CPU inference")
OUTPUT.mkdir(parents=True, exist_ok=True)
metadata = dict(
    checkpoint=str(CHECKPOINT),
    weight_digest=config["weight_digest"],
    ae_digest=config["ae_digest"],
    members=5,
    years=30,
    seed=config["seed"],
    first=config["first"],
    co2_ppm=config["co2_ppm"],
    method="N(0,1) normalized input fields -> encode -> predict -> decode/re-encode once -> latent-only",
    initial_state_sha256=hashlib.sha256(
        (ORIGINAL / "initial_states.pt").read_bytes()
    ).hexdigest(),
    comparison="Saved local-moment one-cycle arm; identical monthly noise/AR1/RNG checked every year",
    limitations="One ERA5/ORAS5 checkpoint; five members; January start; 30 years; not a universal equivalence test",
)
record = OUTPUT / "configuration.json"
if record.exists() and json.loads(record.read_text()) != metadata:
    raise ValueError("Configuration changed; choose another output")
record.write_text(json.dumps(metadata, indent=2))
model = MonthlyDiffusionV1.load(CHECKPOINT, device="mps")
if (
    model.weight_digest() != config["weight_digest"]
    or model.weight_digest(autoencoder_only=True) != config["ae_digest"]
):
    raise ValueError("Model differs from saved baseline")
print(
    "Loaded exact baseline checkpoint; starting paired isotropic/one-cycle arm",
    flush=True,
)
run_forecast(model, config, stop_after_years)
del model
torch.mps.empty_cache()
if stop_after_years == 30:
    ds = assemble()
    analyze(ds)
    render(ds)
sources = [
    Path(__file__),
    ROOT / "monthly_diffusion/forecast.py",
    ROOT / "scripts/diagnose_cycling_modes.py",
]
(OUTPUT / "code_provenance.json").write_text(
    json.dumps(
        {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, indent=2
    )
)
