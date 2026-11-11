"""Monthly NAO–SST maps from same-latent normal decoder reconstructions."""

from pathlib import Path
import sys, argparse
import numpy as np, pandas as pd, torch, xarray as xr, yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
from monthly_diffusion import MonthlyDiffusionV1
from monthly_diffusion.data.sources import open_canonical_state, open_co2


def wm(x):
    return x.weighted(np.cos(np.deg2rad(x.lat))).mean(("lat", "lon"))


def nao(a):
    v = wm(a.mslp.sel(lat=slice(35, 45), lon=slice(330, 360))) - wm(
        a.mslp.sel(lat=slice(60, 70), lon=slice(330, 350))
    )
    return v.groupby("time.month") - v.groupby("time.month").mean("time")


def reg(field, index):
    x = field - field.mean("time")
    y = index - index.mean("time")
    return ((x * y).mean("time") / (y * y).mean("time")).rename(field.name or "sst")
