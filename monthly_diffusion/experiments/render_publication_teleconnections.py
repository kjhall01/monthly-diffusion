#!/usr/bin/env python
"""Render retained monthly Niño3.4 teleconnections in field units per °C.

Run from the repository root with the publication-nino34-t2m-precip YAML.
The saved long-rollout diagnostics subtract each ensemble's exact monthly
ensemble mean and multiply both fields and SST by sqrt(N/(N-1)). Niño3.4 is
the cosine-latitude-weighted SST anomaly over 5S--5N, 170W--120W. There is no
second monthly-climatology removal, detrending, smoothing, or seasonal subset.

Saved OLS maps regress internal fields on independently standardized indices,
pooling members and months with an intercept. For I in °C and z=(I-mean)/sd,
beta_I = beta_z / sd(I), EXACTLY, including under spatially varying missing
field data. This script uses each ensemble's own saved index standard deviation
(ddof=0) on the same sample interval; it does not use the reference's SD for MD.
Finite-ensemble rescaling cancels in this per-degree slope because the same
factor was applied to predictor and response. T2M differences need no Kelvin
offset; precipitation converts from kg m-2 s-1 to mm day-1 with factor 86400.

These are contemporaneous monthly associations, not lagged responses or causal
effects. Pooling historical and SSP3-7.0 months can hide changes in teleconnection
strength with forcing. Per-degree maps assess coupling, not ENSO amplitude.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import xarray as xr
import yaml

from monthly_diffusion.analysis.long_rollout import save_dataset
from monthly_diffusion.analysis.plotting import plot_response_matrix

STYLE = {
    "t2m": ("2-meter Temperature", "°C / °C Niño3.4", "K", 1.0),
    "mtpr": ("Precipitation", "mm day⁻¹ / °C Niño3.4", "kg m-2 s-1", 86400.0),
}


def load_teleconnections(root, variables):
    """Convert verified saved per-SD slopes without reading full rollout fields."""
    root = Path(root)
    maps, indices, counts = {}, {}, {}
    metadata = dict(ensembles={}, input_sha256={})
    intervals = []
    for source in ("reference", "model"):
        index_path = root / source / "indices.nc"
        with xr.open_dataset(index_path) as ds:
            index = ds.ENSO.load()
            anomaly = ds.attrs["anomaly_definition"]
        if index.attrs.get("units") not in ("degC", "°C", "degree_Celsius"):
            raise ValueError("Saved Niño3.4 must be in degrees Celsius")
        if set(index.dims) != {"member", "time"} or not np.isfinite(index).all():
            raise ValueError("Expected finite member-by-time Niño3.4 indices")
        if "exact same-month ensemble mean" not in anomaly:
            raise ValueError(
                "This figure expects the ensemble-residual anomaly definition"
            )
        index_std = float(index.std(("member", "time"), ddof=0))
        if not np.isfinite(index_std) or index_std <= 0:
            raise ValueError("Niño3.4 must have positive finite variance")
        interval = [str(index.time.values[0]), str(index.time.values[-1])]
        intervals.append(interval)
        metadata["ensembles"][source] = dict(
            members=list(map(str, index.member.values)),
            count=index.sizes["member"],
            months=index.sizes["time"],
            pooled_samples=index.size,
            index_std_degC=index_std,
            anomaly_definition=anomaly,
        )
        metadata["input_sha256"][str(index_path.resolve())] = hashlib.sha256(
            index_path.read_bytes()
        ).hexdigest()
        fields, support = {}, {}
        for variable in variables:
            path = root / source / f"{variable}.nc"
            with xr.open_dataset(path) as ds:
                # Older saved maps predate explicit regression_start/end attrs;
                # their full archive coordinates and configuration bind the fit.
                if "per +1 SD" not in ds.attrs["regression"]:
                    raise ValueError("Saved regression is not a per-SD coefficient")
                if ds.regression.attrs.get("units") != STYLE[variable][2]:
                    raise ValueError(f"Unexpected {variable} native units")
                start = np.datetime64(
                    ds.attrs.get("regression_start", ds.time.values[0])
                )
                end = np.datetime64(ds.attrs.get("regression_end", ds.time.values[-1]))
                if start != index.time.values[0] or end != index.time.values[-1]:
                    raise ValueError("Index and regression time intervals differ")
                if not np.array_equal(ds.member.values, index.member.values):
                    raise ValueError("Index and regression members differ")
                fields[variable] = (
                    ds.regression.sel(mode="ENSO", drop=True).load()
                    * STYLE[variable][3]
                    / index_std
                )
                fields[variable].attrs = dict(
                    units=STYLE[variable][1],
                    long_name=STYLE[variable][0] + " regression on Niño3.4",
                    predictor="Niño3.4 SST anomaly (°C)",
                )
                support[variable] = ds.regression_count.sel(
                    mode="ENSO", drop=True
                ).load()
            metadata["input_sha256"][str(path.resolve())] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        maps[source], indices[source], counts[source] = (
            xr.Dataset(fields),
            index.to_dataset(),
            xr.Dataset(support),
        )
    if intervals[0] != intervals[1]:
        raise ValueError("Use matched reference and model regression periods")
    metadata.update(
        period=intervals[0],
        lag_months=0,
        season="all months",
        regression="OLS with intercept, pooled member × month, slope per +1°C Niño3.4",
        conversion="native per-SD slope × physical unit factor / that ensemble's own Niño3.4 SD (ddof=0)",
        nino34_region=dict(lat=[-5, 5], lon_east=[190, 240], weights="cos(latitude)"),
    )
    reference, model = xr.align(maps["reference"], maps["model"], join="exact")
    return reference, model, indices, counts, metadata
