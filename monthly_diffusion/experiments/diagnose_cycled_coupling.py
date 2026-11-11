"""Paired ENSO consistency after a saved decode/re-encode initialization pilot.

No forecasts are regenerated. Reuse immutable training-fitted latent probes.
Decode saved predicted latents to compute physical Nino3.4 and a gridded SOI
proxy (Tahiti-box minus Darwin-box MSLP). Decode encoded TRAINING latents at
the same fixed CO2 to supply decoder-matched monthly climatologies and a fair
autoencoded physical relationship. Also retain the actual source relationship.
SOI proxy is monthly pressure-difference anomaly / training anomaly SD, not
station SOI. Reverse its sign for comparisons: El Nino positive on both axes.

No rollout-specific climatology, linear detrending, ensemble-mean subtraction,
or fitted rollout recalibration. Correlation centers only for its definition.
Report offsets, spread and departures from the TRAINING regression as well as
correlation. Thirty months/one member cannot establish coupling or causality.
"""

from pathlib import Path
import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[2]
from monthly_diffusion import MonthlyDiffusionV1
from monthly_diffusion.analysis.indices import regional_mean
from monthly_diffusion.data.sources import open_canonical_state

COMPONENTS = ("atmosphere", "ocean")
COLORS = {
    "cycled_predicted": "#D55E00",
    "cycled_fed_back": "#CC79A7",
    "latent_only": "#0072B2",
}
LABELS = {
    "cycled_predicted": "Cycled: prediction",
    "cycled_fed_back": "Cycled: re-encoded",
    "latent_only": "Latent-only control",
}


def fixed_anomalies(values, dates, reference, reference_dates):
    rm = pd.DatetimeIndex(reference_dates).month.values
    m = pd.DatetimeIndex(dates).month.values
    climate = np.stack([reference[rm == i].mean(0) for i in range(1, 13)])
    return values - climate[m - 1], reference - climate[rm - 1]


def pair_metrics(pair, reference):
    """x=ocean/temperature; y=atmosphere/negative SOI. Reference units fixed."""
    x, y = np.asarray(pair).T
    rx, ry = np.asarray(reference).T
    slope, intercept = np.polyfit(rx, ry, 1)
    ref_residual = ry - intercept - slope * rx
    residual = y - intercept - slope * x
    return dict(
        n=len(x),
        correlation=float(np.corrcoef(x, y)[0, 1]),
        increment_correlation=float(np.corrcoef(np.diff(x), np.diff(y))[0, 1]),
        ocean_mean=float(x.mean()),
        atmosphere_mean=float(y.mean()),
        ocean_sd_ratio=float(x.std() / rx.std()),
        atmosphere_sd_ratio=float(y.std() / ry.std()),
        slope=float(np.polyfit(x, y, 1)[0]),
        reference_slope=float(slope),
        relationship_bias=float(residual.mean()),
        relationship_rmse_ratio=float(
            np.sqrt(np.mean(residual**2) / np.mean(ref_residual**2))
        ),
        relationship_centered_scatter_ratio=float(residual.std() / ref_residual.std()),
    )


def lag_curve(pair, maxlag=6):
    """Positive lag = ocean/temperature leads atmosphere/-SOI, no wrapping."""
    x, y = np.asarray(pair).T
    curve = []
    for lag in range(-maxlag, maxlag + 1):
        a, b = (
            (x[:-lag], y[lag:])
            if lag > 0
            else ((x[-lag:], y[:lag]) if lag < 0 else (x, y))
        )
        curve.append(np.corrcoef(a, b)[0, 1])
    return np.asarray(curve)


def physical_indices(atmos, ocean):
    nino = regional_mean(ocean.sst, lat_bounds=(-5.0, 5.0), lon_bounds=(190.0, 240.0))
    tahiti = regional_mean(
        atmos.mslp, lat_bounds=(-20.0, -10.0), lon_bounds=(205.0, 216.0)
    )
    darwin = regional_mean(
        atmos.mslp, lat_bounds=(-18.0, -8.0), lon_bounds=(126.0, 136.0)
    )
    return xr.Dataset(dict(nino34=nino, pressure_difference=tahiti - darwin)).compute()


@torch.inference_mode()
def decode_indices(model, raw, dates, ppm):
    """Streaming batches; retain only two scalars per month, no field archives."""
    outputs = []
    batch = model.config.autoencoder_training.batch_size
    for start in range(0, len(dates), batch):
        z = [
            xr.DataArray(
                raw[c][start : start + batch],
                dims=("time", "node", "channel"),
                coords=dict(time=dates[start : start + batch]),
                name=c,
            )
            for c in COMPONENTS
        ]
        a, o = model.decode(*z, co2=ppm)
        outputs.append(physical_indices(a, o))
        if (start + batch) % 120 == 0 or start + batch >= len(dates):
            print(
                f"Decoded index scalars: {min(start + batch, len(dates))}/{len(dates)}",
                flush=True,
            )
    return xr.concat(outputs, dim="time")


def plot_comparison(reference, candidates, ref_dates, dates, kind, output, windows):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    sx, sy = reference.std(0)
    ref = reference / np.array([sx, sy])
    axes[0, 0].scatter(
        *ref.T, s=9, color=".65", alpha=0.35, label="Encoded training reference"
    )
    slope, intercept = np.polyfit(ref[:, 0], ref[:, 1], 1)
    limit = max(
        3.0, *(float(np.abs(v / np.array([sx, sy])).max()) for v in candidates.values())
    )
    xx = np.array([-limit, limit])
    residual_sd = (ref[:, 1] - intercept - slope * ref[:, 0]).std()
    axes[0, 0].plot(xx, intercept + slope * xx, color="black", lw=1)
    axes[0, 0].fill_between(
        xx,
        intercept + slope * xx - residual_sd,
        intercept + slope * xx + residual_sd,
        color="black",
        alpha=0.10,
    )
    lags = np.arange(-6, 7)
    axes[1, 0].plot(
        lags, lag_curve(reference), color="black", label="Full training reference"
    )
    if len(windows):
        curves = np.stack([lag_curve(w) for w in windows])
        axes[1, 0].fill_between(
            lags,
            *np.quantile(curves, [0.05, 0.95], axis=0),
            color=".5",
            alpha=0.2,
            label="Matched 30-month training windows: 5–95%",
        )
    for name, pair in candidates.items():
        values = pair / np.array([sx, sy])
        color, label = COLORS[name], LABELS[name]
        axes[0, 0].plot(
            values[:, 0],
            values[:, 1],
            "o-",
            ms=3,
            lw=0.6,
            alpha=0.8,
            color=color,
            label=label,
        )
        axes[0, 1].plot(
            np.arange(1, len(pair) + 1),
            values[:, 0],
            color=color,
            label=f"{label}: ocean",
        )
        axes[0, 1].plot(
            np.arange(1, len(pair) + 1), values[:, 1], color=color, ls="--", alpha=0.8
        )
        axes[1, 0].plot(lags, lag_curve(pair), color=color, label=label)
        residual = values[:, 1] - intercept - slope * values[:, 0]
        axes[1, 1].plot(np.arange(1, len(pair) + 1), residual, color=color, label=label)
    axes[1, 1].axhspan(-residual_sd, residual_sd, color="black", alpha=0.10)
    for ax in axes.flat:
        ax.grid(alpha=0.16)
        ax.axhline(0, color=".4", lw=0.6)
    xname, yname = (
        ("Ocean ENSO readout", "Atmosphere ENSO readout")
        if kind == "latent"
        else ("Niño3.4", "−SOI proxy")
    )
    axes[0, 0].set(
        xlabel=f"{xname} / reference SD",
        ylabel=f"{yname} / reference SD",
        title="Joint relationship · fixed monthly anomalies",
    )
    axes[0, 0].legend(fontsize=8, frameon=False)
    axes[0, 1].set(
        xlabel="Generated month",
        ylabel="Reference SD",
        title="Solid: ocean / dashed: atmosphere",
    )
    axes[1, 0].set(
        xlabel="Lag (months); positive = ocean leads",
        ylabel="Correlation",
        ylim=(-1, 1),
        title="Lead–lag synchronization",
    )
    axes[1, 0].legend(fontsize=7, frameon=False)
    axes[1, 1].set(
        xlabel="Generated month",
        ylabel="Atmosphere reference SD",
        title="Departure from reference regression · shading ±1 SD",
    )
    fig.suptitle(
        f"{'Latent ENSO' if kind == 'latent' else 'Physical Niño3.4–SOI'} coupling · 30-month initialization pilot",
        fontsize=13,
    )
    for ext in ("png", "pdf"):
        fig.savefig(output / f"{kind}_coupling.{ext}", dpi=200)
    plt.close(fig)
