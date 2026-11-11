"""Paired 1- versus 12-month AE-cycling initialization, compared with 0 and 30.

Use the exact saved local-month/gridpoint-matched initial histories, unchanged
checkpoint, seeds, five members, constant 1850 CO2 and 30-year calendar. Only
the number of initial predict/decode/re-encode months changes. Every arm then
uses the original latent-only dynamics; no history or AR1-noise reset at cutoff.

The common cycled prefix is reused from the existing 30-month-cycled archive.
We replay sampling to advance the RNG/noise, verify its predictions agree, then
use the original predicted/fed-back latents and physical indices exactly. This
avoids tiny GPU rounding differences in an otherwise identical prefix becoming
a hidden perturbation. Completed new years are resumable. No decoded field
archives, training or checkpoint modifications are made.
"""

from pathlib import Path
import argparse
import hashlib
import json
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monthly_diffusion import MonthlyDiffusionV1


from monthly_diffusion.experiments.experiment_cycle_duration import (
    COMPONENTS,
    DURATIONS,
    run_experiment,
    analyze,
)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument(
    "--baseline",
    type=Path,
    default=ROOT / "analysis/local_moment_initialization_30years",
)
p.add_argument(
    "--output", type=Path, default=ROOT / "analysis/cycling_duration_30years"
)
p.add_argument("--device", default="mps")
p.add_argument("--stop-after-years", type=int, default=30)
p.add_argument("--analysis-only", action="store_true")
args = p.parse_args()
torch.set_num_threads(4)
baseline_config = json.loads((args.baseline / "configuration.json").read_text())
base = baseline_config["baseline_configuration"]
initial_path = args.baseline / "initial_state.pt"
run = Path(base["checkpoint"]).parent
cache = torch.load(run / "latents.pt", map_location="cpu", weights_only=False)
if cache["ae_digest"] != base["ae_digest"]:
    raise ValueError("Autoencoder identity mismatch")
config = dict(
    baseline=str(args.baseline.resolve()),
    baseline_configuration=baseline_config,
    durations=DURATIONS,
    initial_state_sha256=hashlib.sha256(initial_path.read_bytes()).hexdigest(),
    common_prefix="Reuse exact old cycled prefix; replay sampler to advance RNG and verify prediction agreement",
    primary="Years 11–30; fixed training monthly anomalies; no ensemble mean subtraction or detrending",
    sensitivity="Own monthly climatology and linear trend jointly removed per member/window",
    limitations="Five members; only January start; durations 1 and 12 cannot locate the exact minimum or distinguish duration from calendar effects",
)
args.output.mkdir(parents=True, exist_ok=True)
path = args.output / "configuration.json"
if path.exists() and json.loads(path.read_text()) != config:
    raise ValueError("Configuration changed; choose new output")
path.write_text(json.dumps(config, indent=2))
if not args.analysis_only:
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no CPU fallback")
    model = MonthlyDiffusionV1.load(run / "model.pt", device=args.device)
    if model.weight_digest() != base["weight_digest"]:
        raise ValueError("Checkpoint identity mismatch")
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    run_experiment(
        model, initial, base, args.baseline, args.output, args.stop_after_years
    )
    del model
if args.analysis_only or args.stop_after_years == 30:
    probes = {
        c: np.load(
            ROOT / "analysis/encoded_vs_direct_noise_enso_rollouts" / f"{c}_readout.npz"
        )["coef"]
        for c in COMPONENTS
    }
    reference = (
        ROOT
        / "analysis/cycled_noise_initialization_era5_30months/coupling/encoded_reference_indices.nc"
    )
    analyze(args.output, args.baseline, cache, probes, reference)
(args.output / "code_provenance.json").write_text(
    json.dumps(
        {
            str(Path(__file__).name): hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest()
        },
        indent=2,
    )
)
