"""Separate initialization offsets from internal variability in saved readouts.

No new inference. The fixed-training-baseline comparison is preserved; these
alternative anomalies must NOT be used as evidence of matching mean states.
The monthly/trend fits use the stated simulation window, not physical years
1966–2006, so they reproduce an anomaly operator, not the original experiment.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal
from monthly_diffusion.experiments.diagnose_isotropic_cycling_sweep import (
    OUTPUT,
    STRONG,
    COMPONENTS,
    plot,
    compare,
)


def run(folder, window=(60, 100)):
    ds = xr.load_dataset(folder / "mode_trajectories.nc")
    selected = slice(window[0] * 12, window[1] * 12)
    months = (
        ds.calendar_month.values
        if "calendar_month" in ds
        else np.repeat(ds.time.dt.month.values[:, None], ds.sizes["member"], axis=1)
    )
    components = (*COMPONENTS, "physical") if "physical" in ds else COMPONENTS
    sensors = (
        tuple(("physical", m) for m in ds["mode"].values)
        if "physical" in ds
        else STRONG
    )
    treatments = {"fixed_training_month": ds.copy(deep=True)}
    for name, detrend in (("own_month", False), ("own_month_and_trend", True)):
        clean = ds.copy(deep=True)
        for c in components:
            for member in range(ds.sizes["member"]):
                mon = months[selected, member]
                design = np.eye(12)[mon - 1]
                if detrend:
                    time = np.linspace(-1, 1, len(mon))
                    design = np.column_stack((design, time))
                x = ds[c].values[selected, :, member, :]
                fit = (
                    design
                    @ np.linalg.lstsq(design, x.reshape(len(mon), -1), rcond=None)[0]
                )
                clean[c].values[selected, :, member, :] = x - fit.reshape(x.shape)
        clean.attrs["anomalies"] = (
            "Each member's own monthly cycle"
            + (" + linear trend" if detrend else "")
            + " removed over displayed window"
        )
        treatments[name] = clean
    clean = ds.copy(deep=True)
    cohort = ds.cohort.values if "cohort" in ds else np.zeros(ds.sizes["member"])
    for group in np.unique(cohort):
        members = np.flatnonzero(cohort == group)
        n = len(members)
        if n < 2:
            raise ValueError("At least two members per calendar cohort required")
        for c in components:
            x = ds[c].isel(member=members)
            clean[c].loc[dict(member=ds.member.values[members])] = np.sqrt(
                n / (n - 1)
            ) * (x - x.mean("member"))
    clean.attrs["anomalies"] = (
        "Exact time-matched ensemble mean removed within each start-season cohort; finite-N correction"
    )
    treatments["ensemble_internal"] = clean
    rows = []
    for treatment, clean in treatments.items():
        out = folder / "seasonality_sensitivity" / treatment
        out.mkdir(parents=True, exist_ok=True)
        plot(clean, out, window=window)
        if treatment == "own_month_and_trend":
            compare(
                clean,
                out,
                windows={f"years{window[0] + 1}_{window[1]}": window},
                metric_names=(
                    "sd_ratio",
                    "acf_rmse",
                    "spectral_cdf_distance",
                    "enso_coupling_difference",
                ),
            )
        for c, m in sensors:
            for method in ds.method.values:
                x = clean[c].sel({"mode": m, "method": method}).values[selected]
                f, p = signal.welch(
                    x.T, fs=12, nperseg=min(240, len(x)), axis=1, detrend="constant"
                )
                seasonal = (abs(f - 1) <= 0.11) | (abs(f - 2) <= 0.11)
                fraction = p[:, seasonal].sum(1) / p[:, 1:].sum(1)
                for member, v in enumerate(fraction):
                    rows.append(
                        dict(
                            treatment=treatment,
                            component=c,
                            mode=m,
                            method=method,
                            member=member,
                            seasonal_band_power_fraction=v,
                            sd=float(x[:, member].std()),
                        )
                    )
    table = pd.DataFrame(rows)
    table.to_csv(folder / "seasonality_sensitivity/seasonal_power.csv", index=False)
    print(
        table[table.method == "observed"]
        .groupby(["component", "mode", "treatment"])["seasonal_band_power_fraction"]
        .mean()
        .to_string(),
        flush=True,
    )
