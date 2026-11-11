"""Read ENSO/NAO from both frozen latent components.

This is an autoencoder/readout diagnostic, not a transition rollout.  It fits
blocked-ridge linear probes on the frozen r1 training latents, then reports
how well atmosphere and ocean latents separately reconstruct physical ENSO
(Nino3.4 SST) and NAO (Azores-minus-Iceland MSLP) indices.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from monthly_diffusion.data.sources import open_canonical_state
from umd.analysis.enso_diversity import nino_boxes, box_ec


def weighted_mean(x):
    w = np.cos(np.deg2rad(x.lat))
    return x.weighted(w).mean(("lat", "lon"))


def region(x, lat0, lat1, lon0, lon1):
    return x.sel(lat=slice(lat0, lat1), lon=slice(lon0, lon1))


def physical_indices(atmos, ocean):
    enso = weighted_mean(region(ocean.sst, -5, 5, 190, 240))
    nao = weighted_mean(region(atmos.mslp, 35, 45, 330, 360)) - weighted_mean(
        region(atmos.mslp, 60, 70, 330, 350))
    boxes = nino_boxes(ocean.sst)
    boxes = boxes - boxes.groupby("time.month").mean("time").sel(month=boxes.time.dt.month).drop_vars("month")
    ec = box_ec(boxes.values)
    result = xr.Dataset({"ENSO": enso, "NAO": nao,
                         "E": ("time", ec[:, 0]), "C": ("time", ec[:, 1])})
    month_mean = result.groupby("time.month").mean("time")
    return result - month_mean.sel(month=result.time.dt.month).drop_vars("month")


def ridge_fit(x, y, alpha=1e-2):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xm, ym = x.mean(0), y.mean()
    xc = x - xm
    gram = xc.T @ xc
    coef = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), xc.T @ (y - ym))
    return coef, ym - xm @ coef


def metrics(pred, truth):
    pred, truth = np.asarray(pred), np.asarray(truth)
    return dict(correlation=float(np.corrcoef(pred, truth)[0, 1]),
                std_ratio=float(pred.std() / truth.std()),
                rmse=float(np.sqrt(np.mean((pred - truth) ** 2))),
                bias=float(np.mean(pred - truth)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / "runs/mdv1-e3smv2-r1i1p1f1")
    parser.add_argument("--data", type=Path, default=ROOT / "examples/e3sm-local.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/latent_mode_readouts_epoch159")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = yaml.safe_load(args.data.read_text())
    atmos, ocean, _ = open_canonical_state(source["state_zarr"], expected_attrs=source.get("expected_attrs"))
    cache = torch.load(args.run / "latents.pt", map_location="cpu", weights_only=False)
    rows, series = [], {}
    for split, dates in (("training", pd.DatetimeIndex(cache["training_time"].values)),
                         ("validation", pd.DatetimeIndex(cache["validation_time"].values))):
        a = cache[split]["atmosphere"].numpy().reshape(len(dates), -1)
        o = cache[split]["ocean"].numpy().reshape(len(dates), -1)
        aa = atmos.sel(time=dates); oo = ocean.sel(time=dates)
        target = physical_indices(aa, oo)
        for mode in ("ENSO", "NAO", "E", "C"):
            truth = target[mode].values
            for component, features in (("atmosphere", a), ("ocean", o)):
                coef, intercept = ridge_fit(features, truth)
                prediction = features @ coef + intercept
                series[f"{split}_{component}_{mode}"] = prediction
                rows.append(dict(split=split, mode=mode, component=component, **metrics(prediction, truth), n=len(truth)))
        if split == "training":
            for mode in ("ENSO", "NAO", "E", "C"):
                series[f"{split}_physical_{mode}"] = target[mode].values
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output / "summary.csv", index=False)
    records = []
    for key, values in series.items():
        split, component, mode = key.split("_", 2)
        if split == "training":
            records.append(xr.DataArray(values, dims=("time",), coords={"time": cache["training_time"]},
                                        name=key))
    xr.merge(records).to_netcdf(args.output / "readouts.nc")
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(4, 2, figsize=(11, 12), constrained_layout=True)
        for row, mode in enumerate(("ENSO", "NAO", "E", "C")):
            for col, split in enumerate(("training", "validation")):
                dates = pd.DatetimeIndex(cache[f"{split}_time"].values)
                ax = axes[row, col]
                truth = physical_indices(atmos.sel(time=dates), ocean.sel(time=dates))[mode].values
                ax.plot(dates, truth, color="black", lw=1.2, label="physical")
                for component, color in (("atmosphere", "#0072B2"), ("ocean", "#D55E00")):
                    ax.plot(dates, series[f"{split}_{component}_{mode}"], color=color, lw=1, label=component)
                ax.axhline(0, color=".7", lw=.6); ax.set_title(f"{mode} · {split}"); ax.grid(alpha=.15)
                if row == 1: ax.set_xlabel("Time")
                if col == 0: ax.set_ylabel("Index anomaly")
        axes[0, 0].legend(frameon=False, ncol=3, fontsize=8)
        fig.suptitle("Frozen MDv1 latent readouts")
        fig.savefig(args.output / "latent_mode_readouts.png", dpi=220)
        fig.savefig(args.output / "latent_mode_readouts.pdf")
        plt.close(fig)
    print(summary.to_string(index=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
