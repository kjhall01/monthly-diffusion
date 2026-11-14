"""All paired ENSO readout trajectories, retaining mean shifts and drift.

Read existing archives only: no inference, refitting, or rollout. Subtract the
fixed 1966–2006 calendar-month means of each component's real encoded ENSO
projection. Do NOT fit a rollout climatology or trend, remove the ensemble
mean, or rescale the amplitude. The first five years remain visible.
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
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


p = argparse.ArgumentParser(description=__doc__)
p.add_argument(
    "--directory",
    type=Path,
    default=ROOT / "analysis/encoded_vs_direct_noise_enso_rollouts",
)
args = p.parse_args()
root = args.directory
record = json.loads((root / "configuration.json").read_text())
cache = load_checkpoint(
    Path(record["checkpoint"]).parent / "latents.pt",
    map_location="cpu",
    weights_only=False,
)
ds = xr.load_dataset(root / "rollout_latents_readouts.nc")
if ds.attrs["checkpoint_sha256"] != record["checkpoint_sha256"]:
    raise ValueError("Readout archive and configuration disagree")
training_month = pd.DatetimeIndex(np.asarray(cache["training_time"])).month
forecast_month = pd.DatetimeIndex(ds.time.values).month
fields = {}
for component in ("atmosphere", "ocean"):
    coef = np.load(root / f"{component}_readout.npz")["coef"]
    reference = (
        cache["training"][component].numpy().reshape(len(training_month), -1) @ coef
    )
    monthly = np.array([reference[training_month == m].mean() for m in range(1, 13)])
    baseline = xr.DataArray(
        monthly[forecast_month - 1], dims="time", coords=dict(time=ds.time)
    )
    fields[component] = ds[f"enso_{component}"] - baseline
    fields[component].attrs.update(
        units="degrees C equivalent",
        definition="latent linear ENSO readout, not decoded SST",
    )
    fields[f"{component}_training_monthly_mean"] = xr.DataArray(
        monthly, dims="month", coords=dict(month=np.arange(1, 13))
    )
corrected = xr.Dataset(
    fields,
    attrs=dict(
        checkpoint_sha256=record["checkpoint_sha256"],
        ae_digest=cache["ae_digest"],
        anomaly_definition="Fixed real-encoded training-period calendar-month means removed",
        detrending="none",
        ensemble_mean_subtraction="none",
        smoothing="none",
        training="1966-2006",
        first_five_years="included; gray shading only",
    ),
)
corrected.to_netcdf(
    root / "fixed_climatology_readouts.nc",
    encoding={v: dict(zlib=True, complevel=2) for v in corrected.data_vars},
)

members = ds.member.values
fig, axes = plt.subplots(
    len(members),
    2,
    figsize=(15, 2.1 * len(members) + 1.5),
    sharex=True,
    sharey=True,
    squeeze=False,
    constrained_layout=True,
)
elapsed = (np.arange(ds.sizes["time"]) + 1) / 12.0
limits = np.concatenate([corrected[c].values.ravel() for c in ("atmosphere", "ocean")])
low, high = float(np.floor(limits.min())), float(np.ceil(limits.max()))
for row, member in enumerate(members):
    for column, component in enumerate(("atmosphere", "ocean")):
        ax = axes[row, column]
        ax.axvspan(0, 5, color="gray", alpha=0.10, zorder=0)
        for kind, color, label in (
            ("direct_latent", "#0072B2", "Direct latent noise"),
            ("encoded_noise", "#D55E00", "Encoded field noise"),
        ):
            values = corrected[component].sel(initialization=kind, member=member)
            ax.plot(elapsed, values, color=color, lw=1.1, label=label)
        ax.axhline(0, color="black", lw=0.7)
        ax.set(xlim=(0, elapsed[-1]), ylim=(low, high))
        ax.grid(alpha=0.15)
        if column == 0:
            ax.set_ylabel(
                f"Member {row + 1}\nENSO readout (°C-equivalent)", fontsize=10
            )
        if row == 0:
            ax.set_title(f"{component.title()} latent readout", fontsize=12)
        if row == len(members) - 1:
            ax.set_xlabel("Years after initialization")
handles, labels = axes[0, 0].get_legend_handles_labels()
handles.append(Patch(facecolor="gray", alpha=0.15))
labels.append("First five years (retained)")
fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
fig.suptitle(
    "All paired latent ENSO trajectories · ERA5/ORAS5 SST-support model\n"
    "Fixed training-month climatology removed · no detrending or ensemble-mean subtraction",
    fontsize=13,
)
for suffix in ("png", "pdf"):
    fig.savefig(
        root / f"enso_all_member_trajectories.{suffix}", dpi=190, bbox_inches="tight"
    )
plt.close(fig)
print(root / "enso_all_member_trajectories.png")
