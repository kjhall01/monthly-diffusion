#!/usr/bin/env python
"""Render retained frozen-package response maps without changing diagnostics."""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import xarray as xr
import yaml

from monthly_diffusion.analysis.long_rollout import save_dataset
from monthly_diffusion.analysis.plotting import plot_response_matrix

STYLE = {
    "t2m": ("2-meter Temperature", "°C", "K", 1.0),
    "sst": ("Sea Surface Temperature", "°C", "degC", 1.0),
    "mtpr": ("Precipitation", "mm day⁻¹", "kg m-2 s-1", 86400.0),
}


def load_responses(root, variables):
    """Reuse exact saved fields/periods/members; temperature DIFFERENCES need no offset."""
    root = Path(root)
    responses, inputs, periods, counts = {}, {}, [], {}
    for source in ("reference", "model"):
        maps = {}
        for variable in variables:
            path = root / source / f"{variable}.nc"
            with xr.open_dataset(path) as ds:
                field = ds.forced_response.load()
                if field.attrs.get("units") != STYLE[variable][2]:
                    raise ValueError(
                        f"Unexpected {variable} source units: {field.attrs}"
                    )
                period = json.loads(ds.attrs["periods"])
                periods.append({name: period[name] for name in ("early", "late")})
                # Verify the retained response against its saved climatologies.
                xr.testing.assert_allclose(
                    field,
                    ds.climatology.sel(period="late", drop=True)
                    - ds.climatology.sel(period="early", drop=True),
                )
                maps[variable] = field * STYLE[variable][3]
                maps[variable].attrs["units"] = STYLE[variable][1]
                counts[source] = dict(
                    members=list(map(str, ds.member.values)), count=ds.sizes["member"]
                )
            inputs[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        responses[source] = xr.Dataset(maps)
    if any(period != periods[0] for period in periods):
        raise ValueError("Response maps use different periods")
    reference, model = xr.align(
        responses["reference"], responses["model"], join="exact"
    )
    return (
        reference,
        model,
        dict(periods=periods[0], ensembles=counts, input_sha256=inputs),
    )
