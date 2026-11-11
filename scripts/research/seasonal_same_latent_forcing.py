"""Seasonal maps from an existing same-latent decoder response archive."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
VARIABLES = ("t2m", "sst", "mtpr")
LABELS = {
    "t2m": "2-meter temperature",
    "sst": "sea-surface temperature",
    "mtpr": "precipitation",
}
UNITS = {"t2m": "K", "sst": "°C", "mtpr": "kg m⁻² s⁻¹"}


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    type=Path,
    required=True,
    help="same_latent_response.nc from same_latent_forcing.py",
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--variables", nargs="+", default=list(VARIABLES))
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
source = xr.open_dataset(args.input)
dates = pd.DatetimeIndex(source.time.values)
season_fields = {}
for variable in args.variables:
    field = source[f"response_{variable}"]
    seasonal = []
    for season, months in SEASONS.items():
        selected = field.isel(time=np.flatnonzero(np.isin(dates.month, months)))
        seasonal.append(selected.mean("time", skipna=True))
    season_fields[variable] = xr.concat(
        seasonal, dim=xr.IndexVariable("season", list(SEASONS))
    )
archive = xr.Dataset({v: x for v, x in season_fields.items()})
archive.attrs.update(
    dict(source.attrs),
    seasonal_definition="DJF/MAM/JJA/SON means of same-latent monthly response",
)
archive.to_netcdf(args.output / "seasonal_same_latent_response.nc")

rows = []
for variable, field in season_fields.items():
    for season in SEASONS:
        x = field.sel(season=season)
        rows.append(
            {
                "variable": variable,
                "season": season,
                "global_mean": float(x.mean(skipna=True)),
                "spatial_rms": float(np.sqrt((x**2).mean(skipna=True))),
            }
        )
pd.DataFrame(rows).to_csv(args.output / "summary.csv", index=False)

# A compact comparison figure: seasons in rows, variables in columns.
limits = {
    v: max(float(np.nanpercentile(np.abs(season_fields[v].values), 98)), 1.0e-12)
    for v in args.variables
}
fig, axes = plt.subplots(
    4,
    len(args.variables),
    figsize=(4.3 * len(args.variables), 9.0),
    squeeze=False,
    constrained_layout=True,
)
for j, variable in enumerate(args.variables):
    for i, season in enumerate(SEASONS):
        field = season_fields[variable].sel(season=season)
        im = axes[i, j].pcolormesh(
            field.lon,
            field.lat,
            field,
            cmap="RdBu_r",
            vmin=-limits[variable],
            vmax=limits[variable],
            shading="auto",
        )
        axes[i, j].set_title(
            f"{season} · {LABELS.get(variable, variable)}" if i == 0 else season
        )
        axes[i, j].set_xlabel("longitude")
        axes[i, j].set_ylabel("latitude")
        fig.colorbar(
            im, ax=axes[i, j], pad=0.02, fraction=0.046, label=UNITS.get(variable, "")
        )
fig.savefig(
    args.output / "seasonal_same_latent_response.png", dpi=250, bbox_inches="tight"
)
fig.savefig(args.output / "seasonal_same_latent_response.pdf", bbox_inches="tight")
plt.close(fig)

for variable in args.variables:
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.8), constrained_layout=True)
    for ax, season in zip(axes.flat, SEASONS):
        field = season_fields[variable].sel(season=season)
        im = ax.pcolormesh(
            field.lon,
            field.lat,
            field,
            cmap="RdBu_r",
            vmin=-limits[variable],
            vmax=limits[variable],
            shading="auto",
        )
        ax.set_title(season)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
    fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        shrink=0.85,
        label=f"late − early same-latent response ({UNITS.get(variable, '')})",
    )
    fig.suptitle(f"Seasonal same-latent response · {LABELS.get(variable, variable)}")
    fig.savefig(
        args.output / f"{variable}_seasonal_response.png", dpi=250, bbox_inches="tight"
    )
    fig.savefig(args.output / f"{variable}_seasonal_response.pdf", bbox_inches="tight")
    plt.close(fig)
print(f"Wrote {args.output}")
