"""Isotropic versus local-moment Gaussian fields, with ONE AE feedback cycle.

Reuse the five-member ERA5 checkpoint, initial field encodings, monthly noise
stream, and 30-year fixed-CO2 protocol from the saved initialization experiments.
Only the missing isotropic/one-cycle arm is generated. All comparison archives
are read-only. No moments are fitted, no model trained, and no full fields saved.

Run: python scripts/research/experiment_isotropic_cycle1.py
"""

from monthly_diffusion.checkpoints import load_checkpoint
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, wasserstein_distance
import torch
import xarray as xr

from workflows.legacy_model import MonthlyDiffusionV1
from monthly_diffusion.checkpoints import atomic_save
from workflows.analysis.spectra import lag_autocorrelation, temporal_spectrum
from workflows.experiments.diagnose_cycling_modes import fixed_month_anomalies, index_fields, scalar_statistics
from workflows.experiments.diagnose_latent_attractor import anomalies

OUTPUT = ROOT / "analysis/isotropic_cycle1_30years"
ORIGINAL = ROOT / "analysis/cycled_initialization_ensemble_30years"
MATCHED = ROOT / "analysis/cycling_duration_30years"
AUDIT = ROOT / "analysis/cycling_duration_modes"
CHECKPOINT = ROOT / "runs/mdv1-era5-oras5-sst-support/model.pt"
COMPONENTS = ("atmosphere", "ocean")
MODES = ("ENSO", "PDO", "NAO", "QBO")
METHODS = ("isotropic_cycle1", "cycle1", "matched", "observed")
LABELS = {
    "isotropic_cycle1": "Isotropic + 1 cycle",
    "cycle1": "Local moments + 1 cycle",
    "matched": "Local moments, no cycle",
    "observed": "Observed initialization",
}
COLORS = {
    "isotropic_cycle1": "#0072B2",
    "cycle1": "#D55E00",
    "matched": "#CC79A7",
    "observed": "#009E73",
}


@torch.inference_mode()
def run_forecast(model, config, stop_after_years):
    initial = load_checkpoint(
        ORIGINAL / "initial_states.pt", map_location="cpu", weights_only=False
    )["latent_only"]
    forecast = model.initialize(
        co2=config["co2_ppm"],
        members=5,
        seed=config["seed"],
        gaussian_start=config["first"],
    )
    # These are the actual saved encodings of masked N(0,1) INPUT FIELDS, not
    # Gaussian latent draws. Reusing them preserves exactly the original seed.
    forecast.load_state_dict(initial)
    forecast.cycle_months = 1
    loading = xr.load_dataarray(AUDIT / "pdo_loading.nc")
    shards = sorted(OUTPUT.glob("year-*.pt"))
    if [p.name for p in shards] != [
        f"year-{y:03d}.pt" for y in range(1, len(shards) + 1)
    ]:
        raise ValueError("Missing annual restart block")
    if shards:
        forecast.load_state_dict(
            load_checkpoint(shards[-1], map_location="cpu", weights_only=False)["state"]
        )
        if forecast.step != 12 * len(shards) or forecast.cycle_months != 1:
            raise ValueError("Restart step/cycling policy disagrees")
    first_block = load_checkpoint(
        ORIGINAL / "year-001.pt", map_location="cpu", weights_only=False
    )
    original_arm = list(first_block["methods"]).index("cycle30")
    started = time.monotonic()
    for year in range(len(shards) + 1, stop_after_years + 1):
        dates, physical, corrections = [], [], []
        predicted = {c: [] for c in COMPONENTS}
        fed_back = {c: [] for c in COMPONENTS}
        for month in range(12):
            previous = [h[:, 0].clone() for h in forecast.history]
            z = next(forecast)
            expected_time = np.datetime64(f"{1799 + year}-{month + 1:02d}-01", "ns")
            if forecast.times[0] != expected_time:
                raise ValueError("Skipped/misordered forecast time")
            for i, c in enumerate(COMPONENTS):
                torch.testing.assert_close(
                    forecast.history[i][:, 1], previous[i], atol=0, rtol=0
                )
                before = z[i].values[0]
                after = forecast.history[i][:, 0].cpu().numpy()
                if not np.isfinite(before).all() or not np.isfinite(after).all():
                    raise FloatingPointError("Nonfinite predicted/fed-back latent")
                if forecast.step == 1:
                    # Independent replay check: the original isotropic 30-cycle
                    # treatment is exactly the requested treatment in month one.
                    np.testing.assert_allclose(
                        before,
                        first_block["predicted"][c][0, original_arm],
                        atol=5e-5,
                        rtol=5e-5,
                    )
                    np.testing.assert_allclose(
                        after,
                        first_block["fed_back"][c][0, original_arm],
                        atol=5e-5,
                        rtol=5e-5,
                    )
                    # Use its exact stored feedback to avoid roundoff becoming
                    # another initialization perturbation; RNG/noise stay live.
                    forecast.history[i][:, 0] = torch.as_tensor(
                        first_block["fed_back"][c][0, original_arm], device=model.device
                    )
                    after = forecast.history[i][:, 0].cpu().numpy()
                else:
                    np.testing.assert_array_equal(before, after)
                predicted[c].append(before.copy())
                fed_back[c].append(after.copy())
            corrections.append(
                np.stack(
                    [
                        np.sqrt(
                            np.mean(
                                (fed_back[c][-1] - predicted[c][-1]) ** 2, axis=(1, 2)
                            )
                        )
                        for c in COMPONENTS
                    ],
                    axis=-1,
                )
            )
            # Decode one forecast month at a time; discard fields after indices.
            a, o = model.decode(*z, co2=config["co2_ppm"])
            index = index_fields(a, o, loading)
            values = np.column_stack([index[m].values[0] for m in MODES])
            if not np.isfinite(values).all():
                raise FloatingPointError("Nonfinite decoded index")
            physical.append(values)
            dates.append(expected_time)
            del a, o, index
        comparison = load_checkpoint(
            MATCHED / f"year-{year:03d}.pt", map_location="cpu", weights_only=False
        )
        common = comparison["states"]["cycle1"]
        np.testing.assert_array_equal(dates, comparison["time"])
        # Both arms must have identical terminal innovations, AR(1) history,
        # and sampling RNG progression even though the physical states diverge.
        for new, old in zip(forecast.noise, common["noise"]):
            torch.testing.assert_close(new.cpu(), old.cpu(), atol=0, rtol=0)
        for key in ("torch", "mps"):
            torch.testing.assert_close(
                forecast.rng[key].cpu(), common["rng"][key].cpu(), atol=0, rtol=0
            )
        atomic_save(
            dict(
                time=np.asarray(dates),
                predicted={c: np.stack(v) for c, v in predicted.items()},
                fed_back={c: np.stack(v) for c, v in fed_back.items()},
                physical=np.stack(physical),
                correction=np.stack(corrections),
                state=forecast.state_dict(),
            ),
            OUTPUT / f"year-{year:03d}.pt",
        )
        seconds = (time.monotonic() - started) / (year - len(shards))
        print(
            f"Year {year}/{stop_after_years} | {seconds:.1f}s/year | ETA {(stop_after_years - year) * seconds / 60:.1f}min | MPS allocated {torch.mps.current_allocated_memory() / 2**30:.2f} GiB",
            flush=True,
        )


def assemble():
    old = xr.load_dataset(AUDIT / "mode_trajectories.nc")
    blocks = [
        load_checkpoint(OUTPUT / f"year-{y:03d}.pt", map_location="cpu", weights_only=False)
        for y in range(1, 31)
    ]
    dates = np.concatenate([b["time"] for b in blocks])
    np.testing.assert_array_equal(dates, old.time.values)
    cache = load_checkpoint(
        CHECKPOINT.parent / "latents.pt", map_location="cpu", weights_only=False
    )
    ref_dates = np.asarray(cache["training_time"])
    np.testing.assert_array_equal(ref_dates, old.training_time.values)
    raw = np.concatenate([b["physical"] for b in blocks])
    ref = xr.load_dataset(AUDIT / "decoded_reference_indices.nc")
    values, _ = fixed_month_anomalies(
        raw, dates, np.column_stack([ref[m] for m in MODES]), ref_dates
    )
    data = {"physical": (("time", "method", "member", "mode"), values[:, None])}
    raw_metrics = []
    for c in COMPONENTS:
        z = np.concatenate([b["predicted"][c] for b in blocks]).astype(float)
        training = cache["training"][c].numpy().astype(float)
        coef = np.stack(
            [np.load(AUDIT / f"probe_{c}_{m}.npz")["coef"] for m in MODES], axis=-1
        )
        projected = z.reshape(len(z), 5, -1) @ coef
        projected_ref = training.reshape(len(training), -1) @ coef
        values, _ = fixed_month_anomalies(projected, dates, projected_ref, ref_dates)
        data[c] = (("time", "method", "member", "mode"), values[:, None])
        # Raw fixed-transition-normalized channels: no anomaly fitting can hide
        # a persistent mean offset or collapse in these extra summaries.
        baseline = []
        for year in range(1, 31):
            block = load_checkpoint(
                MATCHED / f"year-{year:03d}.pt", map_location="cpu", weights_only=False
            )
            baseline.append(
                block["predicted"][c][:, list(block["methods"]).index("cycle1")]
            )
        baseline = np.concatenate(baseline)
        for name, zz in (("isotropic_cycle1", z), ("cycle1", baseline)):
            for label, sl in (
                ("years01_05", slice(0, 60)),
                ("years11_30", slice(120, 360)),
            ):
                for member in range(5):
                    for channel in range(4):
                        x = zz[sl, member, :, channel]
                        raw_metrics.append(
                            dict(
                                method=name,
                                period=label,
                                component=c,
                                member=member + 1,
                                channel=channel,
                                mean=float(x.mean()),
                                sd=float(x.std()),
                                training_mean=float(training[..., channel].mean()),
                                training_sd=float(training[..., channel].std()),
                            )
                        )
    new = xr.Dataset(
        data,
        coords=dict(
            time=dates,
            method=["isotropic_cycle1"],
            member=old.member,
            mode=old["mode"],
            mode_units=old.mode_units,
        ),
    )
    ds = xr.concat(
        [new, old[list(data)].sel({"method": ["cycle1", "matched", "observed"]})],
        dim="method",
        join="exact",
    )
    for name in old.data_vars:
        if name.endswith("reference"):
            ds[name] = old[name]
    ds.attrs.update(
        old.attrs,
        comparison="Isotropic versus locally matched input fields; one feedback cycle in both; paired monthly innovations",
    )
    ds.to_netcdf(
        OUTPUT / "mode_trajectories.nc",
        encoding={v: dict(zlib=True, complevel=3) for v in ds},
    )
    pd.DataFrame(raw_metrics).to_csv(
        OUTPUT / "raw_latent_channel_metrics.csv", index=False
    )
    return ds


def analyze(ds):
    rows, coupling = [], []
    windows = {
        "years01_05": slice(0, 60),
        "years06_15": slice(60, 180),
        "years11_30": slice(120, 360),
    }
    for kind in ("physical", *COMPONENTS):
        for mode in MODES:
            ref = ds[kind + "_reference"].sel(mode=mode).values
            for period, sl in windows.items():
                for method in METHODS:
                    for member in ds.member.values:
                        values = (
                            ds[kind]
                            .sel({"method": method, "member": member, "mode": mode})
                            .values[sl]
                        )
                        for treatment in (
                            "fixed_training_month",
                            "own_month",
                            "own_month_detrended",
                        ):
                            x, r = values, ref
                            if treatment != "fixed_training_month":
                                name = (
                                    "window_monthly"
                                    if treatment == "own_month"
                                    else "window_monthly_detrend"
                                )
                                x = anomalies(x[:, None], ds.time.values[sl], name)[
                                    :, 0
                                ]
                                r = anomalies(
                                    r[:, None], ds.training_time.values, name
                                )[:, 0]
                            rows.append(
                                dict(
                                    kind=kind,
                                    mode=mode,
                                    period=period,
                                    method=method,
                                    member=int(member),
                                    anomalies=treatment,
                                    wasserstein_reference_sd=wasserstein_distance(x, r)
                                    / r.std(),
                                    **scalar_statistics(x, r),
                                )
                            )
    for mode in MODES:
        for method in METHODS:
            for member in ds.member.values:
                a = ds.atmosphere.sel(
                    {"mode": mode, "method": method, "member": member}
                ).values[120:]
                o = ds.ocean.sel(
                    {"mode": mode, "method": method, "member": member}
                ).values[120:]
                coupling.append(
                    dict(
                        mode=mode,
                        method=method,
                        member=int(member),
                        correlation=float(np.corrcoef(a, o)[0, 1]),
                    )
                )
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "member_metrics.csv", index=False)
    table.groupby(["kind", "mode", "period", "anomalies", "method"]).mean(
        numeric_only=True
    ).drop(columns="member").reset_index().to_csv(OUTPUT / "summary.csv", index=False)
    pd.DataFrame(coupling).to_csv(OUTPUT / "atmosphere_ocean_coupling.csv", index=False)
    rng = np.random.default_rng(19)
    bootstrap = rng.integers(0, 5, size=(10000, 5))
    paired = []
    late = table[table.period == "years11_30"]
    for keys, group in late.groupby(["kind", "mode", "anomalies"]):
        iso = group[group.method == "isotropic_cycle1"].set_index("member").sort_index()
        matched = group[group.method == "cycle1"].set_index("member").sort_index()
        for metric in (
            "sd_ratio",
            "acf24",
            "mean_reference_sd",
            "seasonal_mean_sd_ratio",
            "wasserstein_reference_sd",
        ):
            difference = (iso[metric] - matched[metric]).values
            distribution = difference[bootstrap].mean(1)
            lo, hi = np.quantile(distribution, [0.025, 0.975])
            paired.append(
                dict(
                    kind=keys[0],
                    mode=keys[1],
                    anomalies=keys[2],
                    metric=metric,
                    isotropic=float(iso[metric].mean()),
                    local_moments=float(matched[metric].mean()),
                    difference=float(difference.mean()),
                    paired_bootstrap_low=float(lo),
                    paired_bootstrap_high=float(hi),
                    members=5,
                )
            )
    pd.DataFrame(paired).to_csv(OUTPUT / "paired_comparison.csv", index=False)
    return table


def render(ds):
    skill = pd.read_csv(AUDIT / "readout_skill.csv")
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    for kind in ("physical", *COMPONENTS):
        fig, axes = plt.subplots(4, 3, figsize=(13, 11), layout="constrained")
        for row, mode in enumerate(MODES):
            reference = ds[kind + "_reference"].sel(mode=mode).values
            scale = reference.std()
            curves = {"reference": reference[:, None] / scale}
            curves.update(
                {
                    m: ds[kind].sel({"method": m, "mode": mode}).values[120:] / scale
                    for m in ("isotropic_cycle1", "cycle1", "observed")
                }
            )
            bound = max(
                4.0,
                float(
                    np.quantile(
                        np.abs(np.concatenate([v.ravel() for v in curves.values()])),
                        0.995,
                    )
                )
                * 1.1,
            )
            support = np.linspace(-bound, bound, 401)
            for name, values in curves.items():
                color = "black" if name == "reference" else COLORS[name]
                label = (
                    "Encoded training reference"
                    if name == "reference"
                    else LABELS[name]
                )
                axes[row, 0].plot(
                    support,
                    gaussian_kde(values.ravel())(support),
                    color=color,
                    label=label,
                )
                series = xr.DataArray(values.T, dims=("member", "time"))
                acf = lag_autocorrelation(series, max_lag=60, detrend="constant")
                power = temporal_spectrum(series, nperseg=120).isel(
                    frequency=slice(1, None)
                )
                axes[row, 1].plot(np.arange(61) / 12, acf.mean("member"), color=color)
                axes[row, 2].loglog(
                    1 / power.frequency / 12, power.mean("member"), color=color
                )
            title = mode
            if kind in COMPONENTS:
                r = float(
                    skill[
                        (skill.component == kind)
                        & (skill["mode"] == mode)
                        & (skill.split == "validation")
                    ].correlation.iloc[0]
                )
                title += f" · readout validation r={r:.2f}" + (
                    " (weak)" if r < 0.5 else ""
                )
            axes[row, 0].set(
                title=title, xlabel="Encoded-reference SD", ylabel="Density"
            )
            axes[row, 1].set(xlabel="Lag (years)", ylabel="ACF", ylim=(-0.8, 1))
            axes[row, 1].axhline(0, color=".7", lw=0.7)
            axes[row, 2].set(xlabel="Period (years)", ylabel="PSD", xlim=(1 / 6, 10))
            for ax in axes[row]:
                ax.grid(alpha=0.15)
        fig.suptitle(
            f"{kind.title()} · years 11–30 · one-cycle initialization\nFixed training-month anomalies; no trend/ensemble-mean removal",
            fontsize=12,
        )
        fig.legend(
            *axes[0, 0].get_legend_handles_labels(),
            loc="outside lower center",
            ncol=2,
            frameon=False,
        )
        for ext in ("png", "pdf"):
            fig.savefig(OUTPUT / f"{kind}_dynamics.{ext}", dpi=200)
        plt.close(fig)
    fig, axes = plt.subplots(
        4, 3, figsize=(15, 10), sharex=True, sharey="row", layout="constrained"
    )
    for row, mode in enumerate(MODES):
        scale = float(ds.physical_reference.sel(mode=mode).std())
        for col, method in enumerate(("isotropic_cycle1", "cycle1", "observed")):
            ax = axes[row, col]
            ax.plot(
                np.arange(360) / 12,
                ds.physical.sel({"mode": mode, "method": method}) / scale,
                lw=0.7,
                alpha=0.7,
            )
            ax.axvspan(0, 10, color=".8", alpha=0.25)
            ax.axhline(0, color="black", lw=0.6)
            if row == 0:
                ax.set_title(LABELS[method])
            if col == 0:
                ax.set_ylabel(f"{mode}\nReference SD")
            if row == 3:
                ax.set_xlabel("Simulation years")
    fig.suptitle("All five decoded index trajectories · gray = early adjustment")
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT / f"physical_trajectories.{ext}", dpi=200)
    plt.close(fig)
