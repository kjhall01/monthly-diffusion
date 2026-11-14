"""Inspect retained spin-up latents without decoding, refitting or inference.

Primary anomalies subtract the FIXED encoded-training calendar-month mean,
per node/channel. No spin-up monthly climatology, linear trend, ensemble mean
or extrapolated training trend is removed. ENSO probes are the already fitted
training-only ridge readouts, not physical SST. PSD/ACF center their own windows
as required for these statistics, but never linearly detrend them. Window means
are retained separately so a persistent displacement cannot hide in the ACF.
"""

from monthly_diffusion.checkpoints import load_checkpoint
from pathlib import Path
import argparse
import hashlib
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
from workflows.analysis.spectra import lag_autocorrelation, temporal_spectrum
from workflows.analysis.long_rollout_plots import finish

COMPONENTS = ("atmosphere", "ocean")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#6C5B23")


def fixed_monthly_anomalies(values, months, reference, reference_months):
    """values[time,member,...], reference[time,...]; never fit on a rollout."""
    climate = np.stack([reference[reference_months == m].mean(0) for m in range(1, 13)])
    return values - climate[months - 1, None], reference - climate[reference_months - 1]


def window_statistics(values):
    """ENSO readouts [time,member]; preserve means, never join members."""
    x = xr.DataArray(values.T, dims=("member", "time"))
    psd = temporal_spectrum(x, nperseg=min(120, len(values)))
    acf = lag_autocorrelation(x, max_lag=min(60, len(values) - 1), detrend="constant")
    return psd, acf


def load_latent_trajectory(forecast, metadata, through_year=None):
    """Read saved latents ONLY; each monthly restart stores [current, previous].

    Validate the spin-up/decoded boundary and every adjacent history slot so
    neither the duplicate previous month nor an off-by-one step enters analysis.
    """
    blocks = sorted((forecast / "spinup_latents").glob("year-*.pt"))
    if not blocks or [b.name for b in blocks] != [
        f"year-{i:04d}.pt" for i in range(1, len(blocks) + 1)
    ]:
        raise ValueError("Missing/noncontiguous spin-up blocks")
    raw, times = {c: [] for c in COMPONENTS}, []
    for block in blocks:
        saved = load_checkpoint(block, map_location="cpu", weights_only=False)
        times.extend(saved["time"])
        if saved["state"]["step"] != len(times):
            raise ValueError("Spin-up dates and restart step disagree")
        for ci, c in enumerate(COMPONENTS):
            np.testing.assert_array_equal(
                saved["state"]["history"][ci][:, 0], saved["latents"][c][-1]
            )
            raw[c].append(saved["latents"][c])
    expected = np.asarray(
        metadata["spinup_valid_times"][: len(times)], dtype="datetime64[ns]"
    )
    np.testing.assert_array_equal(np.asarray(times), expected)
    if len(times) < 240 or len(times) % 12:
        raise ValueError(
            "This early/late analysis requires at least 20 complete spin-up years"
        )
    if through_year is not None:
        if len(times) != metadata["spinup_months"]:
            raise ValueError(
                "Complete spin-up is required before appending decoded-run latents"
            )
        valid = np.asarray(metadata["valid_times"], dtype="datetime64[ns]")
        end = np.datetime64(f"{through_year}-12", "M")
        positions = np.flatnonzero(valid.astype("datetime64[M]") <= end)
        if len(positions) == 0 or valid[positions[-1]].astype("datetime64[M]") != end:
            raise ValueError(
                "Requested final December is outside the rollout date range"
            )
        previous = [raw[c][-1][-1] for c in COMPONENTS]
        for i in positions:
            path = forecast / f"month-{i + 1:06d}.zarr" / "forecast_state.pt"
            state = load_checkpoint(path, map_location="cpu", weights_only=False)
            np.testing.assert_array_equal(
                np.asarray(state["times"]), np.asarray([valid[i], times[-1]])
            )
            if state["step"] != metadata["spinup_months"] + i + 1:
                raise ValueError(f"Unexpected restart step in {path}")
            for ci, c in enumerate(COMPONENTS):
                np.testing.assert_array_equal(state["history"][ci][:, 1], previous[ci])
                current = state["history"][ci][:, 0].numpy()
                raw[c].append(current[None])
                previous[ci] = current
            times.append(valid[i])
    times = np.asarray(times)
    serial = times.astype("datetime64[M]").astype(int)
    if not np.all(np.diff(serial) == 1):
        raise ValueError("Missing/duplicate month in joined latent trajectory")
    return {c: np.concatenate(v).astype(float) for c, v in raw.items()}, times
