"""Does observed initialization repair weak ENSO in a frozen trained model?

Fit ocean/atmosphere ridge readouts of physical Nino3.4 using training data
only. Monthly intercepts and a linear trend are removed from predictors and
target; blocked-year CV refits these nuisance terms inside every fold.
Validate the readouts on the already-cached held-out validation years.

Compare paired observed-history and IID-Gaussian-history latent forecasts.
Both use the package Forecast iterator, identical seeds, and initially unset
AR(1) noise histories. Only the initial two latent states differ. No decoding,
retraining, changed sampler, or ensemble-mean subtraction is involved.
"""

# Make the repository workflow helpers available when running this file directly.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from monthly_diffusion.checkpoints import load_checkpoint

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from workflows.legacy_model import MonthlyDiffusionV1
from workflows.data.sources import open_canonical_state, open_co2
from workflows.analysis.indices import enso_index


from workflows.experiments.diagnose_enso_initialization import design, fit_readout, metrics, readout, compare_initialization, render

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--run", type=Path, default=ROOT / "runs/mdv1-era5-oras5")
p.add_argument("--sources", type=Path, default=ROOT / "examples/era5-oras5-local.yaml")
p.add_argument(
    "--output", type=Path, default=ROOT / "analysis/era5_enso_observed_initialization"
)
p.add_argument("--device", default="cpu")
p.add_argument("--members", type=int, default=10)
p.add_argument("--seed", type=int, default=20260930)
args = p.parse_args()
torch.set_num_threads(4)
args.output.mkdir(parents=True, exist_ok=True)
model = MonthlyDiffusionV1.load(args.run / "model.pt", device=args.device)
cache = load_checkpoint(args.run / "latents.pt", map_location="cpu", weights_only=False)
if cache["ae_digest"] != model.weight_digest(autoencoder_only=True):
    raise ValueError("Cached latents do not match the autoencoder")
sources = yaml.safe_load(args.sources.read_text())
a, o, _ = open_canonical_state(
    sources["state_zarr"], expected_attrs=sources.get("expected_attrs")
)
co2 = open_co2(sources["co2_netcdf"], expected_sha256=sources.get("co2_sha256"))
# Read only the small Nino3.4 region. Never load the full physical archive.
source_index = enso_index(
    o[["sst"]].sel(time=slice("1958", "2006"), lat=slice(-5, 5), lon=slice(190, 240)),
    remove_monthly=False,
).compute()
dates = np.asarray(cache["training_time"])
val_dates = np.asarray(cache["validation_time"])
metadata = dict(
    checkpoint=str(args.run.resolve() / "model.pt"),
    checkpoint_sha256=model.weight_digest(),
    stages=model.stages,
    seed=args.seed,
    members=args.members,
    device=args.device,
    training=[1966, 2006],
    validation=[1958, 1965],
    sources=sources,
    long_comparison="paired observed versus IID Gaussian history, Jan 1966-Dec 2006",
    initialized_forecasts="October-ending 1958-1963, each 24 months; validation-only targets",
    readout="ridge; 5 contiguous-year CV folds choose alpha; train-only monthly intercepts + linear trend",
    caveat="latent linear diagnostic, not decoded SST; validation was used for model checkpoint selection",
)
path = args.output / "configuration.json"
if path.exists() and json.loads(path.read_text()) != metadata:
    raise ValueError("Diagnostic configuration changed; use a new output directory")
path.write_text(json.dumps(metadata, indent=2))
probes, rows, series = {}, [], {}
for c in ("atmosphere", "ocean"):
    x = cache["training"][c].numpy().reshape(len(dates), -1)
    y = source_index.sel(time=dates).values
    probes[c] = fit_readout(x, y, dates)
    probe = probes[c]
    for split, dt in (("training", dates), ("validation", val_dates)):
        truth = (
            source_index.sel(time=dt).values - design(dt) @ probe["physical_baseline"]
        )
        pred = readout(cache[split][c].numpy(), dt, probe)[:, 0]
        rows.append(
            dict(component=c, split=split, alpha=probe["alpha"], **metrics(pred, truth))
        )
        series[f"{split}_{c}"] = xr.DataArray(
            pred, dims=f"{split}_time", coords={f"{split}_time": dt}
        )
        series[f"{split}_physical"] = xr.DataArray(
            truth, dims=f"{split}_time", coords={f"{split}_time": dt}
        )
    rows.append(
        dict(
            component=c,
            split="blocked_cv_alpha_selection",
            alpha=probe["alpha"],
            **metrics(probe["cv_prediction"], probe["cv_truth"]),
        )
    )
    np.savez(args.output / f"{c}_readout.npz", **probe)
pd.DataFrame(rows).to_csv(args.output / "readout_skill.csv", index=False)
xr.Dataset(series).to_netcdf(args.output / "encoded_readouts.nc")
print(pd.DataFrame(rows).to_string(index=False), flush=True)
long = compare_initialization(
    model,
    cache,
    probes,
    first="1966-01",
    months=492,
    members=args.members,
    seed=args.seed,
    co2=co2,
    output=args.output,
)
short = []
for year in range(1958, 1964):
    ds = compare_initialization(
        model,
        cache,
        probes,
        first=f"{year}-11",
        months=24,
        members=args.members,
        seed=args.seed + year,
        co2=co2,
        output=args.output,
    )
    target = (
        source_index.sel(time=ds.time).values
        - design(ds.time) @ probes["ocean"]["physical_baseline"]
    )
    ds = ds.assign(observed_enso=("time", target))
    short.append(ds)
    print(f"Finished validation initialization October {year}", flush=True)
render(args.output, long, short, series, probes, source_index)
