"""Paired 1- versus 12-month AE-cycling initialization, compared with 0 and 30.

Use the exact saved local-month/gridpoint-matched initial histories, unchanged
checkpoint, seeds, five members, constant 1850 CO2 and 30-year calendar. Only
the number of initial predict/decode/re-encode months changes. Every arm then
uses the original latent-only dynamics; no history or AR1-noise reset at cutoff.

The common cycled prefix is reused from the existing 30-month-cycled archive.
We replay sampling to advance the RNG/noise, verify its predictions agree, then
use the original predicted/fed-back latents and physical indices exactly. This
avoids tiny GPU rounding differences in an otherwise identical prefix becoming
a hidden perturbation. Completed new years are resumable. No decoded field
archives, training or checkpoint modifications are made.
"""

from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
from monthly_diffusion import MonthlyDiffusionV1
from monthly_diffusion.training.common import atomic_save
from monthly_diffusion.analysis.spectra import lag_autocorrelation, temporal_spectrum
from monthly_diffusion.experiments.experiment_cycled_spinup import index_statistics
from monthly_diffusion.experiments.diagnose_cycled_coupling import (
    physical_indices,
    lag_curve,
)
from monthly_diffusion.experiments.diagnose_latent_attractor import anomalies

COMPONENTS = ("atmosphere", "ocean")
DURATIONS = {"cycle1": 1, "cycle12": 12}
DISPLAY = ("matched", "cycle1", "cycle12", "matched_cycle30", "observed")
LABELS = dict(
    matched="0 months",
    cycle1="1 month",
    cycle12="12 months",
    matched_cycle30="30 months",
    observed="Observed initialization",
)
COLORS = dict(
    matched="#56B4E9",
    cycle1="#E69F00",
    cycle12="#D55E00",
    matched_cycle30="#CC79A7",
    observed="#009E73",
)


@torch.inference_mode()
def run_experiment(model, initial, config, baseline, output, stop_year=30):
    forecasts = {}
    for method in DURATIONS:
        f = model.initialize(
            co2=config["co2_ppm"],
            members=config["members"],
            seed=config["seed"],
            gaussian_start=config["first"],
        )
        f.load_state_dict(copy.deepcopy(initial))
        forecasts[method] = f
    shards = sorted(output.glob("year-*.pt"))
    if [p.name for p in shards] != [
        f"year-{i:03d}.pt" for i in range(1, len(shards) + 1)
    ]:
        raise ValueError("Missing annual block")
    if shards:
        previous = torch.load(shards[-1], map_location="cpu", weights_only=False)
        for method, f in forecasts.items():
            f.load_state_dict(previous["states"][method])
            if f.step != len(shards) * 12:
                raise ValueError("Restart step mismatch")
    tick = time.monotonic()
    for year in range(len(shards) + 1, stop_year + 1):
        old = torch.load(
            baseline / f"year-{year:03d}.pt", map_location="cpu", weights_only=False
        )
        cycled = list(old["methods"]).index("matched_cycle30")
        predicted = {c: [] for c in COMPONENTS}
        feedback = {c: [] for c in COMPONENTS}
        physical, corrections, dates = [], [], []
        for month in range(12):
            step = (year - 1) * 12 + month + 1
            records = []
            for method, f in forecasts.items():
                previous = [h[:, 0].clone() for h in f.history]
                za, zo = next(f)
                before = [h[:, 0].detach().cpu().clone() for h in f.history]
                if step <= DURATIONS[method]:
                    # Same treatment as the existing 30-month arm up to here.
                    # Use its exact result while advancing matched sampler RNG.
                    for ci, c in enumerate(COMPONENTS):
                        cached = torch.as_tensor(old["predicted"][c][month, cycled])
                        torch.testing.assert_close(
                            before[ci], cached, rtol=5.0e-5, atol=5.0e-5
                        )
                        before[ci] = cached.clone()
                        f.history[ci][:, 0] = torch.as_tensor(
                            old["fed_back"][c][month, cycled], device=model.device
                        )
                    indices = old["physical"][month, cycled].copy()
                else:
                    a, o = model.decode(za, zo, co2=config["co2_ppm"])
                    idx = physical_indices(a, o)
                    indices = np.column_stack(
                        (idx.nino34.values[0], idx.pressure_difference.values[0])
                    )
                    del a, o, idx
                after = [h[:, 0].detach().cpu().clone() for h in f.history]
                if not np.isfinite(indices).all():
                    raise FloatingPointError("Nonfinite physical indices")
                for h, prev in zip(f.history, previous):
                    torch.testing.assert_close(h[:, 1], prev, rtol=0, atol=0)
                    if not torch.isfinite(h).all():
                        raise FloatingPointError("Nonfinite latent")
                if f.times[0] != old["time"][month]:
                    raise ValueError("Calendar differs from baseline")
                for n, common in zip(f.noise, forecasts["cycle1"].noise):
                    torch.testing.assert_close(n, common, rtol=0, atol=0)
                records.append(
                    dict(
                        before=before,
                        after=after,
                        physical=indices,
                        correction=np.stack(
                            [
                                ((after[i] - before[i]) ** 2)
                                .mean((1, 2))
                                .sqrt()
                                .numpy()
                                for i in range(2)
                            ],
                            -1,
                        ),
                    )
                )
            for ci, c in enumerate(COMPONENTS):
                predicted[c].append(
                    np.stack([r["before"][ci].numpy() for r in records])
                )
                feedback[c].append(np.stack([r["after"][ci].numpy() for r in records]))
            physical.append(np.stack([r["physical"] for r in records]))
            corrections.append(np.stack([r["correction"] for r in records]))
            dates.append(f.times[0])
        for f in forecasts.values():
            target = old["states"]["matched_cycle30"]
            for n, n_old in zip(f.noise, target["noise"]):
                torch.testing.assert_close(n.cpu(), n_old.cpu(), rtol=0, atol=0)
            for k in ("torch", "mps"):
                if k in target["rng"]:
                    torch.testing.assert_close(
                        f.rng[k], target["rng"][k], rtol=0, atol=0
                    )
        atomic_save(
            dict(
                time=np.asarray(dates),
                methods=list(DURATIONS),
                predicted={c: np.stack(v) for c, v in predicted.items()},
                fed_back={c: np.stack(v) for c, v in feedback.items()},
                physical=np.stack(physical),
                correction=np.stack(corrections),
                states={m: f.state_dict() for m, f in forecasts.items()},
            ),
            output / f"year-{year:03d}.pt",
        )
        print(
            f"Year {year}/30 | {(time.monotonic() - tick) / (year - len(shards)):.1f}s/year",
            flush=True,
        )


def analyze(output, baseline, cache, probes, reference_path):
    old = xr.load_dataset(baseline / "index_trajectories.nc")
    shards = [
        torch.load(p, map_location="cpu", weights_only=False)
        for p in sorted(output.glob("year-*.pt"))
    ]
    dates = np.concatenate([s["time"] for s in shards])
    np.testing.assert_array_equal(dates, old.time.values)
    months = pd.DatetimeIndex(dates).month.values
    training_time = np.asarray(cache["training_time"])
    train_month = pd.DatetimeIndex(training_time).month.values
    variables = {}
    for key in ("predicted", "fed_back"):
        readouts = []
        for c in ("ocean", "atmosphere"):
            z = np.concatenate([s[key][c] for s in shards]).astype(float)
            training = cache["training"][c].numpy().astype(float)
            climate = np.stack(
                [training[train_month == m].mean(0) for m in range(1, 13)]
            )
            readouts.append(
                (z - climate[months - 1, None, None]).reshape(*z.shape[:3], -1)
                @ probes[c]
            )
        name = "latent" if key == "predicted" else "fed_back_latent"
        variables[name] = (
            ("time", "method", "member", "component"),
            np.stack(readouts, -1),
        )
    reference = xr.load_dataset(reference_path)
    np.testing.assert_array_equal(reference.time.values, training_time)
    ref = np.column_stack(
        (reference.nino34.values, -reference.pressure_difference.values)
    )
    physical = np.concatenate([s["physical"] for s in shards])
    physical[..., 1] *= -1
    for m in range(1, 13):
        physical[months == m] -= ref[train_month == m].mean(0)
    variables["physical"] = (("time", "method", "member", "component"), physical)
    new = xr.Dataset(
        variables,
        coords=dict(
            time=dates,
            method=list(DURATIONS),
            member=old.member,
            component=old.component,
        ),
    )
    ds = xr.concat(
        [
            old[list(variables)].sel(
                {"method": ["matched", "matched_cycle30", "observed"]}
            ),
            new,
        ],
        dim="method",
        join="exact",
    )
    for kind in ("latent", "physical"):
        ds[kind + "_reference"] = old[kind + "_reference"]
    ds = ds.sel({"method": list(DISPLAY)})
    ds.attrs.update(
        old.attrs,
        comparison="Only initial cycling duration changes: 0,1,12,30 months; all start from same local-matched histories",
    )
    ds["method"].encoding = {}
    ds.to_netcdf(
        output / "index_trajectories.nc",
        encoding={v: dict(zlib=True, complevel=3) for v in ds},
    )
    with xr.open_dataset(output / "index_trajectories.nc") as saved:
        np.testing.assert_array_equal(saved["method"].values, ds["method"].values)
    rows, curves = [], []
    for kind in ("latent", "physical"):
        ref = ds[kind + "_reference"].values
        for period, sl in (
            ("years01_05", slice(0, 60)),
            ("years06_15", slice(60, 180)),
            ("years11_30", slice(120, 360)),
        ):
            for method in DISPLAY:
                for member in ds.member.values:
                    pair = ds[kind].sel({"method": method, "member": member}).values[sl]
                    for treatment in ("fixed_training_month", "own_month_detrended"):
                        x, r = pair, ref
                        if treatment == "own_month_detrended":
                            x = anomalies(x, dates[sl], "window_monthly_detrend")
                            r = anomalies(r, training_time, "window_monthly_detrend")
                        rows.append(
                            dict(
                                kind=kind,
                                period=period,
                                anomalies=treatment,
                                method=method,
                                member=int(member) + 1,
                                **index_statistics(x, r),
                            )
                        )
                    if period == "years11_30":
                        curves.extend(
                            dict(
                                kind=kind,
                                method=method,
                                member=int(member) + 1,
                                lag=lag,
                                correlation=v,
                            )
                            for lag, v in zip(range(-12, 13), lag_curve(pair, 12))
                        )
    table = pd.DataFrame(rows)
    table.to_csv(output / "member_metrics.csv", index=False)
    pd.DataFrame(curves).to_csv(output / "lead_lag_correlations.csv", index=False)
    summary = (
        table.groupby(["kind", "period", "anomalies", "method"])
        .mean(numeric_only=True)
        .drop(columns="member")
        .reset_index()
    )
    summary.to_csv(output / "summary.csv", index=False)
    paired = []
    late = table[table.period == "years11_30"]
    for kind in ("latent", "physical"):
        for anomaly in ("fixed_training_month", "own_month_detrended"):
            group = late[(late.kind == kind) & (late.anomalies == anomaly)]
            base = group[group.method == "matched"].set_index("member")
            for method in ("cycle1", "cycle12", "matched_cycle30"):
                candidate = group[group.method == method].set_index("member")
                for member in base.index:
                    paired.append(
                        dict(
                            kind=kind,
                            anomalies=anomaly,
                            method=method,
                            member=member,
                            **{
                                k: float(candidate.loc[member, k] - base.loc[member, k])
                                for k in (
                                    "ocean_sd_ratio",
                                    "ocean_acf24",
                                    "correlation",
                                    "relationship_rmse_ratio",
                                )
                            },
                        )
                    )
    pd.DataFrame(paired).to_csv(output / "paired_differences.csv", index=False)
    for kind in ("latent", "physical"):
        ref = ds[kind + "_reference"].values
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        for ci, component in enumerate(("ocean", "atmosphere")):
            r = ref[:, ci] / ref[:, ci].std()
            support = np.linspace(-4, 4, 301)
            axes[ci, 0].plot(
                support,
                gaussian_kde(r)(support),
                color="black",
                label="Encoded reference",
            )
            da = xr.DataArray(r, dims="time")
            axes[ci, 1].plot(
                np.arange(61) / 12,
                lag_autocorrelation(da, max_lag=60, detrend="constant"),
                color="black",
            )
            psd = temporal_spectrum(da, nperseg=120).isel(frequency=slice(1, None))
            axes[ci, 2].loglog(1 / psd.frequency / 12, psd, color="black")
            for method in DISPLAY:
                x = (
                    ds[kind]
                    .sel({"method": method, "component": component})
                    .isel(time=slice(120, 360))
                    .transpose("member", "time")
                    / ref[:, ci].std()
                )
                axes[ci, 0].plot(
                    support,
                    gaussian_kde(x.values.ravel())(support),
                    color=COLORS[method],
                    label=LABELS[method],
                )
                acf = lag_autocorrelation(x, max_lag=60, detrend="constant")
                axes[ci, 1].plot(
                    np.arange(61) / 12, acf.mean("member"), color=COLORS[method]
                )
                psd = (
                    temporal_spectrum(x, nperseg=120)
                    .mean("member")
                    .isel(frequency=slice(1, None))
                )
                axes[ci, 2].loglog(1 / psd.frequency / 12, psd, color=COLORS[method])
            axes[ci, 0].set(
                xlabel="Reference SD; offsets retained",
                ylabel=f"{component.title()} density",
            )
            axes[ci, 1].set(xlabel="Lag (years)", ylabel="ACF", ylim=(-0.7, 1))
            axes[ci, 1].axhline(0, color=".5", lw=0.6)
            axes[ci, 2].set(
                xlabel="Period (years)",
                ylabel="PSD (reference SD² × month)",
                xlim=(0.2, 10),
            )
            for ax in axes[ci]:
                ax.grid(alpha=0.15)
        fig.legend(
            *axes[0, 0].get_legend_handles_labels(),
            loc="outside lower center",
            ncol=3,
            frameon=False,
            fontsize=9,
        )
        fig.suptitle(
            f"{kind.title()} ENSO · years 11–30 · initial cycling duration\nAll noise starts use identical local-matched initial histories and innovations"
        )
        for ext in ("png", "pdf"):
            fig.savefig(output / f"{kind}_dynamics.{ext}", dpi=190)
        plt.close(fig)
        fig, axes = plt.subplots(
            len(ds.member), len(DISPLAY), figsize=(18, 11), sharex=True, sharey=True
        )
        for mi, method in enumerate(DISPLAY):
            for j, member in enumerate(ds.member.values):
                x = ds[kind].sel({"method": method, "member": member}).values / ref.std(
                    0
                )
                ax = axes[j, mi]
                ax.plot(np.arange(360) / 12, x[:, 0], color=COLORS[method], lw=0.9)
                ax.plot(
                    np.arange(360) / 12, x[:, 1], color=".2", ls="--", alpha=0.7, lw=0.7
                )
                ax.axhspan(-1, 1, color=".5", alpha=0.08)
                ax.axhline(0, color=".5", lw=0.5)
                duration = dict(
                    matched=0, cycle1=1, cycle12=12, matched_cycle30=30, observed=0
                )[method]
                if duration:
                    ax.axvline(duration / 12, color=".5", ls=":", lw=0.8)
                ax.grid(alpha=0.12)
                if j == 0:
                    ax.set_title(LABELS[method], fontsize=10)
                if mi == 0:
                    ax.set_ylabel(f"Member {j + 1}\nReference SD")
                if j == len(ds.member) - 1:
                    ax.set_xlabel("Simulation years")
        fig.suptitle(
            f"{kind.title()} · solid ocean; dashed atmosphere · fixed training-month anomalies"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        for ext in ("png", "pdf"):
            fig.savefig(output / f"{kind}_all_trajectories.{ext}", dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for row, kind in enumerate(("latent", "physical")):
        baseline_stats = index_statistics(
            ds[kind + "_reference"].values, ds[kind + "_reference"].values
        )
        for col, metric in enumerate(("ocean_sd_ratio", "ocean_acf24", "correlation")):
            ax = axes[row, col]
            ax.axhline(baseline_stats[metric], color="black", lw=1)
            for mi, method in enumerate(DISPLAY):
                values = late[
                    (late.kind == kind)
                    & (late.anomalies == "fixed_training_month")
                    & (late.method == method)
                ][metric]
                ax.scatter(
                    mi + np.linspace(-0.12, 0.12, len(values)),
                    values,
                    s=20,
                    color=COLORS[method],
                )
                ax.plot(
                    [mi - 0.18, mi + 0.18],
                    [values.mean()] * 2,
                    color=COLORS[method],
                    lw=3,
                )
            ax.set_xticks(range(5), ["0", "1", "12", "30", "Observed"], fontsize=9)
            ax.set_xlabel("Cycling months / initialization")
            ax.set_ylabel(
                f"{kind.title()} · "
                + dict(
                    ocean_sd_ratio="ENSO SD / reference",
                    ocean_acf24="ENSO ACF at 24 months",
                    correlation="Atmosphere–ocean correlation",
                )[metric]
            )
            ax.grid(axis="y", alpha=0.15)
    fig.suptitle(
        "Years 11–30 · dots: members; bars: member means; black: encoded reference"
    )
    for ext in ("png", "pdf"):
        fig.savefig(output / f"duration_summary.{ext}", dpi=190)
    plt.close(fig)
    print(
        summary[
            (summary.period == "years11_30")
            & (summary.anomalies == "fixed_training_month")
        ][
            [
                "kind",
                "method",
                "ocean_sd_ratio",
                "ocean_acf24",
                "correlation",
                "relationship_rmse_ratio",
            ]
        ].to_string(index=False),
        flush=True,
    )
