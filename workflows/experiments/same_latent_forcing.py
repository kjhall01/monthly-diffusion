"""Same-latent decoder CO2-response audit for a frozen MonthlyDiffusionV1 AE.

The physical states are encoded at their historical CO2, then decoded twice
with the identical latent, month, and static fields while only the absolute
CO2 coordinate is replaced by month-matched early and late-period means.
This is a decoder-only forced-response diagnostic; no transition rollout is
performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
import yaml

from workflows.legacy_model import MonthlyDiffusionV1
from workflows.data.sources import open_canonical_state, open_co2
from monthly_diffusion.preprocessing import LatentNormalizer
from workflows.analysis.spatial import area_weights, weighted_mean


KEY_VARIABLES = ("t2m", "sst", "mtpr")
UNITS = {"t2m": "K", "sst": "°C", "mtpr": "mm/day"}
LABELS = {
    "t2m": "2-meter Temperature",
    "sst": "Sea Surface Temperature",
    "mtpr": "Precipitation",
}


def period_monthly_means(
    co2: xr.DataArray, start: int, end: int, months: xr.DataArray
) -> xr.DataArray:
    ref = co2.sel(time=slice(f"{start}-01", f"{end}-12"))
    means = ref.groupby("time.month").mean("time")
    month_numbers = xr.DataArray(np.asarray(months.dt.month.values), dims=("time",))
    values = means.sel(month=month_numbers).values.astype("float32")
    return xr.DataArray(
        values, dims=("time",), coords={"time": months.time}, name="co2"
    )


def select_samples(time: xr.DataArray) -> xr.DataArray:
    years = np.asarray(time.dt.year)
    keep = np.isin(years, [1966, 1986, 2006])
    return time.isel(time=np.flatnonzero(keep))


def decode_response(model, atmos, ocean, co2, early, late):
    samples = select_samples(atmos.time)
    early_co2 = period_monthly_means(co2, *early, samples)
    late_co2 = period_monthly_means(co2, *late, samples)
    logging.info("Encoding and paired-decoding %d monthly states", samples.size)
    # The AE acts independently on each month.  Encode the three contiguous
    # sample years separately, respecting the public API's contiguous-time
    # check without needlessly encoding all 492 training months.
    atmosphere_latents, ocean_latents = [], []
    for year in np.unique(samples.dt.year.values):
        a, o = atmos.sel(time=str(year)), ocean.sel(time=str(year))
        za_year, zo_year = model.encode(a, o, co2=co2)
        atmosphere_latents.append(za_year)
        ocean_latents.append(zo_year)
    za = xr.concat(atmosphere_latents, dim="time").sel(time=samples)
    zo = xr.concat(ocean_latents, dim="time").sel(time=samples)
    ea, eo = model.decode(za, zo, co2=early_co2)
    la, lo = model.decode(za, zo, co2=late_co2)
    early_ds = xr.merge([ea, eo])
    late_ds = xr.merge([la, lo])
    response = late_ds - early_ds
    response.attrs.update(
        definition="D(z_t,month_t,statics,C_late(month_t)) - D(z_t,month_t,statics,C_early(month_t)); z_t=E(x_t,month_t,statics,C_actual(t))",
        early_period=f"{early[0]}-{early[1]}",
        late_period=f"{late[0]}-{late[1]}",
        latent_source=f"{atmos.attrs.get('member_id', 'source')} training states in 1966, 1986, and 2006",
        pairing="same encoded latent, month, and statics; only absolute CO2 changed",
    )
    return early_ds, late_ds, response, early_co2, late_co2


def plot_maps(response, output, early, late):
    for name in KEY_VARIABLES:
        if name not in response:
            continue
        value = response[name].mean("time", skipna=True)
        if name == "mtpr":
            value = (
                value * 86400.0
            )  # Display only; archive and CSV retain native units.
        vmax = float(np.nanpercentile(np.abs(value.values), 99)) or 1.0
        fig, ax = plt.subplots(figsize=(11, 4.8))
        mesh = ax.pcolormesh(
            value.lon,
            value.lat,
            value,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            shading="auto",
        )
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_title(f"{LABELS[name]} · same-latent decoder response")
        cb = fig.colorbar(mesh, ax=ax, pad=0.02, extend="both")
        cb.set_label(UNITS.get(name, ""))
        fig.subplots_adjust(bottom=0.23)
        fig.text(
            0.5,
            0.025,
            f"{late[0]}–{late[1]} CO₂ − {early[0]}–{early[1]} CO₂; month-matched means",
            ha="center",
            fontsize=10,
        )
        fig.savefig(output / f"same_latent_{name}.png", dpi=220, bbox_inches="tight")
        fig.savefig(output / f"same_latent_{name}.pdf", bbox_inches="tight")
        plt.close(fig)
