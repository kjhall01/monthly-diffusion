"""Paired Gaussian/encoded initialization audit of the frozen latent dynamics.

Run 150 years, ten members, from January 1959. Encoded histories are November
and December 1958 from EACH model's own source/cache, ordered current/previous.
Both groups receive identical diffusion innovations and initially unset noise
histories. No physical decoding, ensemble-mean subtraction, or retraining.

Readouts control training monthly intercepts and linear trend when fitting.
They do NOT extrapolate that trend over the synthetic 150-year run. Report raw
projections, fixed training-month anomalies, and independent window-month and
linear-trend removal. Raw channel distributions and training PCA projections
are additional checks: matching one readout is not proof of attractor recovery.

Both trained transitions are autonomous with respect to CO2. The iterator's
required CO2 argument is held at training mean, unused without decoding. The
future calendar labels therefore indicate elapsed time, not a forced projection.
Yearly atomic latent/state shards permit exact resume without physical Zarrs.
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
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from monthly_diffusion import MonthlyDiffusionV1


from monthly_diffusion.experiments.diagnose_latent_attractor import (
    COMPONENTS,
    RUNS,
    paired_rollout,
    prepare_probes,
    initialization_statistics,
    overview,
    analyze,
)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", choices=RUNS, required=True)
p.add_argument("--years", type=int, default=150)
p.add_argument("--members", type=int, default=10)
p.add_argument("--seed", type=int, default=20261001)
p.add_argument("--device", default="cpu")
p.add_argument(
    "--stage",
    choices=("all", "prepare", "rollout", "analyze", "overview"),
    default="all",
)
p.add_argument(
    "--output", type=Path, default=ROOT / "analysis/latent_initialization_attractor"
)
args = p.parse_args()
torch.set_num_threads(4)
output = args.output / args.model
output.mkdir(parents=True, exist_ok=True)
run = RUNS[args.model]
model = MonthlyDiffusionV1.load(run / "model.pt", device=args.device)
cache = torch.load(run / "latents.pt", map_location="cpu", weights_only=False)
if cache["ae_digest"] != model.weight_digest(autoencoder_only=True):
    raise ValueError("Wrong cached autoencoder")
source = yaml.safe_load((run / "data_sources.yaml").read_text())
metadata = dict(
    checkpoint_sha256=model.weight_digest(),
    stages=model.stages,
    config=model.config.to_dict(),
    sources=source,
    years=args.years,
    members=args.members,
    seed=args.seed,
    device=args.device,
    first_month="1959-01",
    history=["1958-12", "1958-11"],
    encoded_initialization="own source, one two-month history replicated across stochastic members",
    forcing="constant train-mean CO2, unused by these transitions; no physical decoding",
    paired_noise="identical Torch seeds/innovations; Gaussian histories use separate NumPy RNG; both noise histories initially unset",
    anomaly_methods=[
        "fixed_training_monthly",
        "window_monthly",
        "window_monthly_detrend",
    ],
    uncertainty="paired-member bootstrap, 2000 replicates; descriptive, no multiple-testing adjustment",
    limitations="One physical initial history per model; 30-year windows cannot establish very-long-period PDO equilibrium. Readout extrapolation outside training latent manifold is not validated.",
    implementation_sha256={
        str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in (
            ROOT / "monthly_diffusion/forecast.py",
            ROOT / "monthly_diffusion/components/coupled_latent_diffusion.py",
            ROOT / "monthly_diffusion/components/sampling.py",
        )
    },
)
path = output / "configuration.json"
if path.exists() and json.loads(path.read_text()) != metadata:
    raise ValueError("Changed experiment identity; use a fresh output directory")
path.write_text(json.dumps(metadata, indent=2))
reference_path = output / "encoded_reference_latents.nc"
if not reference_path.exists():
    xr.Dataset(
        {
            f"{split}_{c}": (
                (f"{split}_time", "node", "channel"),
                cache[split][c].numpy(),
            )
            for split in ("training", "validation")
            for c in COMPONENTS
        },
        coords={
            f"{split}_time": np.asarray(cache[split + "_time"])
            for split in ("training", "validation")
        },
        attrs=dict(
            autoencoder_sha256=cache["ae_digest"],
            coordinate_system="fixed transition normalized",
        ),
    ).to_netcdf(reference_path)
if args.stage in ("all", "prepare", "analyze"):
    probes, targets = prepare_probes(model, cache, source, output)
if args.stage in ("all", "rollout"):
    paired_rollout(
        model, cache, output, years=args.years, members=args.members, seed=args.seed
    )
if args.stage in ("all", "analyze"):
    analyze(cache, probes, targets, output, years=args.years)
if args.stage in ("all", "analyze", "overview"):
    overview(output)
initialization_statistics(cache, output)
