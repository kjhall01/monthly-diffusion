"""Read ENSO/NAO from both frozen latent components.

This is an autoencoder/readout diagnostic, not a transition rollout.  It fits
blocked-ridge linear probes on the frozen r1 training latents, then reports
how well atmosphere and ocean latents separately reconstruct physical ENSO
(Nino3.4 SST) and NAO (Azores-minus-Iceland MSLP) indices.
"""

from __future__ import annotations

# Make the repository workflow helpers available when running this file directly.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from monthly_diffusion.checkpoints import load_checkpoint

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from workflows.data.sources import open_canonical_state


from workflows.experiments.latent_mode_readouts import physical_indices, ridge_fit, metrics

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, default=ROOT / "runs/mdv1-e3smv2-r1i1p1f1")
parser.add_argument("--data", type=Path, default=ROOT / "examples/e3sm-local.yaml")
parser.add_argument(
    "--output", type=Path, default=ROOT / "analysis/latent_mode_readouts_epoch159"
)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
source = yaml.safe_load(args.data.read_text())
atmos, ocean, _ = open_canonical_state(
    source["state_zarr"], expected_attrs=source.get("expected_attrs")
)
cache = load_checkpoint(args.run / "latents.pt", map_location="cpu", weights_only=False)
rows, series = [], {}
for split, dates in (
    ("training", pd.DatetimeIndex(cache["training_time"].values)),
    ("validation", pd.DatetimeIndex(cache["validation_time"].values)),
):
    a = cache[split]["atmosphere"].numpy().reshape(len(dates), -1)
    o = cache[split]["ocean"].numpy().reshape(len(dates), -1)
    aa = atmos.sel(time=dates)
    oo = ocean.sel(time=dates)
    target = physical_indices(aa, oo)
    for mode in ("ENSO", "NAO", "E", "C"):
        truth = target[mode].values
        for component, features in (("atmosphere", a), ("ocean", o)):
            coef, intercept = ridge_fit(features, truth)
            prediction = features @ coef + intercept
            series[f"{split}_{component}_{mode}"] = prediction
            rows.append(
                dict(
                    split=split,
                    mode=mode,
                    component=component,
                    **metrics(prediction, truth),
                    n=len(truth),
                )
            )
    if split == "training":
        for mode in ("ENSO", "NAO", "E", "C"):
            series[f"{split}_physical_{mode}"] = target[mode].values
summary = pd.DataFrame(rows)
summary.to_csv(args.output / "summary.csv", index=False)
records = []
for key, values in series.items():
    split, component, mode = key.split("_", 2)
    if split == "training":
        records.append(
            xr.DataArray(
                values,
                dims=("time",),
                coords={"time": cache["training_time"]},
                name=key,
            )
        )
xr.merge(records).to_netcdf(args.output / "readouts.nc")
with plt.rc_context(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
):
    fig, axes = plt.subplots(4, 2, figsize=(11, 12), constrained_layout=True)
    for row, mode in enumerate(("ENSO", "NAO", "E", "C")):
        for col, split in enumerate(("training", "validation")):
            dates = pd.DatetimeIndex(cache[f"{split}_time"].values)
            ax = axes[row, col]
            truth = physical_indices(atmos.sel(time=dates), ocean.sel(time=dates))[
                mode
            ].values
            ax.plot(dates, truth, color="black", lw=1.2, label="physical")
            for component, color in (("atmosphere", "#0072B2"), ("ocean", "#D55E00")):
                ax.plot(
                    dates,
                    series[f"{split}_{component}_{mode}"],
                    color=color,
                    lw=1,
                    label=component,
                )
            ax.axhline(0, color=".7", lw=0.6)
            ax.set_title(f"{mode} · {split}")
            ax.grid(alpha=0.15)
            if row == 1:
                ax.set_xlabel("Time")
            if col == 0:
                ax.set_ylabel("Index anomaly")
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8)
    fig.suptitle("Frozen MDv1 latent readouts")
    fig.savefig(args.output / "latent_mode_readouts.png", dpi=220)
    fig.savefig(args.output / "latent_mode_readouts.pdf")
    plt.close(fig)
print(summary.to_string(index=False))
print(f"Wrote {args.output}")
