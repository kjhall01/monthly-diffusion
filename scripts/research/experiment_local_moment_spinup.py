"""Training-month/gridpoint moment-matched Gaussian initialization experiment.

Extends experiment_cycled_spinup WITHOUT changing its saved runs or the model.
Fit mu and population SD in the checkpoint's already transformed/centered/scaled
field coordinates, separately at every grid cell and calendar month, on EXACT
checkpoint training dates. Draw mu + SD * epsilon using the old initializer's
same iid epsilon. No spatial, inter-variable or inter-month correlation is added.
Constant cells stay constant; unsupported cells stay missing. The modern
training climatology is NOT claimed to be a reconstructed preindustrial state.

Two arms: matched-noise ordinary rollout; identical initialization with physical
decode/re-encode for months 1--30 only. Both then use normal latent dynamics.
All model noise, forcing, calendar, members and seeds match the previous study.
Annual files retain all predicted/fed-back latents, physical indices and restart
states. Only two initial months pass through the local moment transformation;
neither subsequent inputs nor the checkpoint preprocessing are changed.
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
import yaml
import dask

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from monthly_diffusion import MonthlyDiffusionV1
from monthly_diffusion.training.common import atomic_save


from monthly_diffusion.experiments.experiment_local_moment_spinup import (
    COMPONENTS,
    METHODS,
    fit_moments,
    matched_history,
    run_experiment,
    analyze,
)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument(
    "--baseline",
    type=Path,
    default=ROOT / "analysis/cycled_initialization_ensemble_30years",
)
p.add_argument(
    "--output", type=Path, default=ROOT / "analysis/local_moment_initialization_30years"
)
p.add_argument("--device", default="mps")
p.add_argument("--fit-only", action="store_true")
p.add_argument("--analysis-only", action="store_true")
p.add_argument("--stop-after-years", type=int, default=30)
args = p.parse_args()
torch.set_num_threads(4)
dask.config.set(scheduler="threads", num_workers=4)
base = json.loads((args.baseline / "configuration.json").read_text())
run = Path(base["checkpoint"]).parent
sources = yaml.safe_load((run / "data_sources.yaml").read_text())
cache = torch.load(run / "latents.pt", map_location="cpu", weights_only=False)
record = dict(
    baseline=str(args.baseline.resolve()),
    baseline_configuration=base,
    methods=list(METHODS),
    sources=sources,
    training_dates=[str(t) for t in cache["training_time"]],
    initialization="mu[variable,month,lat,lon] + sd[variable,month,lat,lon]*iid standard normal, checkpoint normalized field units",
    moments="population ddof=0; finite-only; no variance floor; minimum two finite years; checkpoint SST support",
    limitations="Gaussian marginals only; no spatial/inter-variable/temporal covariance; training climate not preindustrial truth",
)
args.output.mkdir(parents=True, exist_ok=True)
path = args.output / "configuration.json"
if path.exists() and json.loads(path.read_text()) != record:
    raise ValueError("Configuration changed; choose a new output")
path.write_text(json.dumps(record, indent=2))
if not args.analysis_only:
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no CPU fallback")
    model = MonthlyDiffusionV1.load(run / "model.pt", device=args.device)
    if (
        model.weight_digest() != base["weight_digest"]
        or cache["ae_digest"] != base["ae_digest"]
    ):
        raise ValueError("Checkpoint differs from previous experiment")
    fit_moments(model, sources, args.output, np.asarray(cache["training_time"]))
    if args.fit_only:
        raise SystemExit(0)
    initial_path = args.output / "initial_state.pt"
    if initial_path.exists():
        initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    else:
        initial = matched_history(model, base, args.output)
        atomic_save(initial, initial_path)
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
code_paths = [
    Path(__file__),
    *(
        ROOT / "scripts" / name
        for name in (
            "experiment_cycled_spinup.py",
            "diagnose_cycled_coupling.py",
            "diagnose_latent_attractor.py",
        )
    ),
]
record = {
    str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in code_paths
}
record["monthly_moment_files"] = {
    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted((args.output / "monthly_moments").glob("*.nc"))
}
(args.output / "code_provenance.json").write_text(json.dumps(record, indent=2))
