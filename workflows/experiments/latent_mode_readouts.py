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

ROOT = Path(__file__).resolve().parents[2]
from workflows.data.sources import open_canonical_state
from workflows.analysis.nino_boxes import nino_boxes, box_ec


def weighted_mean(x):
    w = np.cos(np.deg2rad(x.lat))
    return x.weighted(w).mean(("lat", "lon"))


def region(x, lat0, lat1, lon0, lon1):
    return x.sel(lat=slice(lat0, lat1), lon=slice(lon0, lon1))


def physical_indices(atmos, ocean):
    enso = weighted_mean(region(ocean.sst, -5, 5, 190, 240))
    nao = weighted_mean(region(atmos.mslp, 35, 45, 330, 360)) - weighted_mean(
        region(atmos.mslp, 60, 70, 330, 350)
    )
    boxes = nino_boxes(ocean.sst)
    boxes = boxes - boxes.groupby("time.month").mean("time").sel(
        month=boxes.time.dt.month
    ).drop_vars("month")
    ec = box_ec(boxes.values)
    result = xr.Dataset(
        {"ENSO": enso, "NAO": nao, "E": ("time", ec[:, 0]), "C": ("time", ec[:, 1])}
    )
    month_mean = result.groupby("time.month").mean("time")
    return result - month_mean.sel(month=result.time.dt.month).drop_vars("month")


def ridge_fit(x, y, alpha=1e-2):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xm, ym = x.mean(0), y.mean()
    xc = x - xm
    gram = xc.T @ xc
    coef = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), xc.T @ (y - ym))
    return coef, ym - xm @ coef


def metrics(pred, truth):
    pred, truth = np.asarray(pred), np.asarray(truth)
    return dict(
        correlation=float(np.corrcoef(pred, truth)[0, 1]),
        std_ratio=float(pred.std() / truth.std()),
        rmse=float(np.sqrt(np.mean((pred - truth) ** 2))),
        bias=float(np.mean(pred - truth)),
    )
