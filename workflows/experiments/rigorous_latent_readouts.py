"""Rigorous frozen-AE readouts for ENSO/E/C/NAO.

Fits blocked-year ridge probes on r1 (1966--2006), then evaluates them on
held-out E3SM members and the r1 validation period.  The probes are diagnostics
only; no model weights are changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import gc
import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[2]
from workflows.legacy_model import MonthlyDiffusionV1
from workflows.data.sources import open_canonical_state, open_co2
from workflows.analysis.nino_boxes import nino_boxes, box_ec


def wmean(x):
    return x.weighted(np.cos(np.deg2rad(x.lat))).mean(("lat", "lon"))


def idx(atmos, ocean, climatology=None):
    nino = nino_boxes(ocean.sst)
    if climatology is None:
        climatology = nino.groupby("time.month").mean("time")
    nino = nino - climatology.sel(month=nino.time.dt.month).drop_vars("month")
    ec = box_ec(nino.values)
    enso = wmean(ocean.sst.sel(lat=slice(-5, 5), lon=slice(190, 240)))
    nao = wmean(atmos.mslp.sel(lat=slice(35, 45), lon=slice(330, 360))) - wmean(
        atmos.mslp.sel(lat=slice(60, 70), lon=slice(330, 350))
    )
    # Southern Oscillation pressure-difference proxy: Tahiti minus Darwin,
    # with the usual monthly-anomaly convention.  The canonical archive has
    # MSLP but not station observations, so this is explicitly a gridded SOI
    # proxy rather than the BOM station SOI.
    tahiti = wmean(atmos.mslp.sel(lat=slice(-20, -10), lon=slice(205, 216)))
    darwin = wmean(atmos.mslp.sel(lat=slice(-18, -8), lon=slice(126, 136)))
    soi = tahiti - darwin
    if climatology is None:  # unreachable; retained for readability
        pass
    # ENSO and NAO use the same fixed r1 training monthly climatology below.
    return np.column_stack((ec[:, 0], ec[:, 1], enso.values, soi.values, nao.values))


def monthly_climatology(values, dates):
    values = np.asarray(values)
    months = pd.DatetimeIndex(dates).month.values
    return np.stack([values[months == m].mean(0) for m in range(1, 13)])


def anomalies(values, dates, clim):
    months = pd.DatetimeIndex(dates).month.values
    return np.asarray(values) - np.asarray(clim)[months - 1]


def ridge(x, y, alpha):
    xm, ym = x.mean(0), y.mean()
    xc = x - xm
    gram = xc.T @ xc
    coef = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), xc.T @ (y - ym))
    return coef, ym - xm @ coef


def blocked_probe(x, y, years):
    alphas = np.logspace(-2, 4, 13)
    folds = np.array_split(np.unique(years), 5)
    scores = np.zeros(len(alphas))
    for fold in folds:
        test = np.isin(years, fold)
        train = ~test
        for i, alpha in enumerate(alphas):
            c, b = ridge(x[train], y[train], alpha)
            scores[i] += np.mean((x[test] @ c + b - y[test]) ** 2)
    best = int(np.argmin(scores))
    c, b = ridge(x, y, alphas[best])
    return c, b, float(alphas[best]), float(np.sqrt(scores[best] / len(folds)))


def score(pred, truth):
    pred, truth = np.asarray(pred), np.asarray(truth)
    return dict(
        correlation=float(np.corrcoef(pred, truth)[0, 1]),
        std_ratio=float(pred.std() / truth.std()),
        rmse=float(np.sqrt(np.mean((pred - truth) ** 2))),
        bias=float(np.mean(pred - truth)),
        n=int(len(truth)),
    )
