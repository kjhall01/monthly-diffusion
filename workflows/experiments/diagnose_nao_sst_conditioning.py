"""Test whether decoder conditioning creates an NAO-like SST pattern."""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
from workflows.legacy_model import MonthlyDiffusionV1
from workflows.data.sources import open_canonical_state, open_co2


def wm(x):
    return x.weighted(np.cos(np.deg2rad(x.lat))).mean(("lat", "lon"))


def monthly_anom(x):
    clim = x.groupby("time.month").mean("time")
    return x.groupby("time.month") - clim


def nao_index(atmos):
    value = wm(atmos.mslp.sel(lat=slice(35, 45), lon=slice(330, 360))) - wm(
        atmos.mslp.sel(lat=slice(60, 70), lon=slice(330, 350))
    )
    return monthly_anom(value).rename("NAO")


def regression(field, index):
    x = monthly_anom(field)
    y = index - index.mean("time")
    cov = ((x - x.mean("time")) * y).mean("time")
    return cov / (y**2).mean("time")


def compare(pattern, reference, mask):
    w = np.cos(np.deg2rad(pattern.lat)) * mask
    p, r = pattern.where(mask), reference.where(mask)
    dot = (w * p * r).sum(("lat", "lon"), skipna=True)
    pn = np.sqrt((w * p * p).sum(("lat", "lon"), skipna=True))
    rn = np.sqrt((w * r * r).sum(("lat", "lon"), skipna=True))
    return dict(
        pattern_correlation=float(dot / (pn * rn)),
        amplitude_ratio=float(
            (w * p * r).sum(("lat", "lon"), skipna=True)
            / (w * r * r).sum(("lat", "lon"), skipna=True)
        ),
        rms=float(
            np.sqrt(
                (w * (p - r) ** 2).sum(("lat", "lon"), skipna=True)
                / w.sum(("lat", "lon"), skipna=True)
            )
        ),
    )
