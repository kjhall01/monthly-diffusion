"""Training-month/gridpoint moment-matched Gaussian initialization experiment.

Extends experiment_cycled_spinup WITHOUT changing its saved runs or the model.
Fit mu and population SD in the checkpoint's already transformed/centered/scaled
field coordinates, separately at every grid cell and calendar month, on EXACT
checkpoint training dates. Draw mu + SD * epsilon using the old initializer's
same iid epsilon. No spatial, inter-variable or inter-month correlation is added.
Constant cells stay constant; unsupported cells stay missing. The modern
training climatology is NOT claimed to be a reconstructed preindustrial state.

Two arms: matched-noise ordinary rollout; identical initialization with physical
decode/re-encode for months 1--30 only. Both then use normal latent dynamics.
All model noise, forcing, calendar, members and seeds match the previous study.
Annual files retain all predicted/fed-back latents, physical indices and restart
states. Only two initial months pass through the local moment transformation;
neither subsequent inputs nor the checkpoint preprocessing are changed.
"""

from monthly_diffusion.checkpoints import load_checkpoint
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
import yaml
import dask

ROOT = Path(__file__).resolve().parents[2]
from workflows.legacy_model import MonthlyDiffusionV1
from workflows.data.sources import open_canonical_state
from workflows.data.dataloader import month_features
from monthly_diffusion.checkpoints import atomic_save
from workflows.analysis.spectra import lag_autocorrelation, temporal_spectrum
from workflows.experiments.experiment_cycled_spinup import index_statistics
from workflows.experiments.diagnose_cycled_coupling import physical_indices, lag_curve
from workflows.experiments.diagnose_latent_attractor import anomalies

COMPONENTS = ("atmosphere", "ocean")
METHODS = ("matched", "matched_cycle30")
DISPLAY = ("latent_only", "matched", "cycle30", "matched_cycle30", "observed")
LABELS = {
    "latent_only": "Unit noise",
    "matched": "Local-matched noise",
    "cycle30": "Unit noise + 30-month cycling",
    "matched_cycle30": "Local-matched + 30-month cycling",
    "observed": "Observed initialization",
    "continuous": "Continuous cycling",
}
COLORS = dict(
    latent_only="#0072B2",
    matched="#56B4E9",
    cycle30="#D55E00",
    matched_cycle30="#CC79A7",
    observed="#009E73",
    continuous=".6",
)


def monthly_moments(field):
    """Finite-only temporal moments; caller supplies training dates only.

    Population SD (ddof=0), in normalized field coordinates, not physical units.
    Require two finite years for a monthly cell; never fabricate missing values.
    In particular SD=0 remains exactly zero, not an arbitrary variance floor.
    """
    x = field.astype("float64").where(np.isfinite(field))
    grouped = x.groupby("time.month")
    count = grouped.count("time")
    mean = grouped.mean("time", skipna=True).where(count >= 2)
    sd = grouped.std("time", skipna=True, ddof=0).where(count >= 2)
    return xr.Dataset(
        dict(
            mean=mean.astype("float32"),
            std=sd.astype("float32"),
            count=count.astype("int16"),
        )
    )


def fit_moments(model, sources, output, training_times):
    """Stream one normalized variable at a time; keep the checkpoint immutable."""
    atmos, ocean, _ = open_canonical_state(
        sources["state_zarr"], expected_attrs=sources.get("expected_attrs")
    )
    np.testing.assert_array_equal(model.preprocessor.training_times, training_times)
    folder = output / "monthly_moments"
    folder.mkdir(exist_ok=True)
    rows = []
    for component, source in zip(COMPONENTS, (atmos, ocean)):
        # Exact cache/checkpoint dates, not a new date range or all available data.
        normalized = model.preprocessor.encode(
            source.sel(time=training_times), component
        )
        np.testing.assert_array_equal(normalized.time.values, training_times)
        for name in model.names[component]:
            path = folder / f"{name}.nc"
            if not path.exists():
                tick = time.monotonic()
                x = normalized[name].transpose("time", "lat", "lon").compute()
                stats = monthly_moments(x)
                stats.attrs.update(
                    variable=name,
                    component=component,
                    training_start=str(training_times[0]),
                    training_end=str(training_times[-1]),
                    training_months=len(training_times),
                    coordinate_system="checkpoint normalized field",
                    sd_ddof=0,
                    missing_policy="fewer than two finite samples: unavailable; exact zero SD retained",
                )
                stats.to_netcdf(
                    path, encoding={v: dict(zlib=True, complevel=3) for v in stats}
                )
                del x
                print(f"Fitted {name}: {time.monotonic() - tick:.1f}s", flush=True)
            with xr.open_dataset(path) as stats:
                mu, sd = stats["mean"].values, stats["std"].values
                valid = np.isfinite(mu) & np.isfinite(sd)
                rows.append(
                    dict(
                        variable=name,
                        component=component,
                        valid_month_cells=int(valid.sum()),
                        constant_fraction=float((sd[valid] == 0).mean()),
                        local_sd_p10=float(np.quantile(sd[valid], 0.1)),
                        local_sd_median=float(np.median(sd[valid])),
                        local_sd_p90=float(np.quantile(sd[valid], 0.9)),
                        local_mean_abs_median=float(np.median(np.abs(mu[valid]))),
                    )
                )
    pd.DataFrame(rows).to_csv(output / "initialization_moment_summary.csv", index=False)
    atmos.close()
    ocean.close()


@torch.inference_mode()
def matched_history(model, baseline, output):
    """Same random draws as Forecast's encoded-noise path; only affine rescaling."""
    f = model.initialize(
        co2=baseline["co2_ppm"],
        members=baseline["members"],
        seed=baseline["seed"],
        gaussian_start=baseline["first"],
    )
    dates = xr.DataArray(f.times, dims="time", coords=dict(time=f.times))
    months = month_features(dates).to(model.device)
    force = torch.full(
        (2, 1),
        (baseline["co2_ppm"] - model.preprocessor.co2_mean)
        / model.preprocessor.co2_std,
        dtype=torch.float32,
        device=model.device,
    )
    month_numbers = pd.DatetimeIndex(f.times).month.values  # CURRENT, PREVIOUS
    audit = []
    for ci, component in enumerate(COMPONENTS):
        stats = [
            xr.load_dataset(output / "monthly_moments" / f"{v}.nc").sel(
                month=month_numbers
            )
            for v in model.names[component]
        ]
        mu = np.stack(
            [s["mean"].transpose("month", "lat", "lon").values for s in stats], axis=1
        )
        sd = np.stack(
            [s["std"].transpose("month", "lat", "lon").values for s in stats], axis=1
        )
        valid = np.isfinite(mu) & np.isfinite(sd)
        train_valid = np.stack(
            [
                np.asarray(model.preprocessor.stats[v]["mask"], bool)
                for v in model.names[component]
            ]
        )
        valid &= train_valid[None]
        if component == "ocean" and model.preprocessor.ocean_support_mask is not None:
            valid &= np.asarray(model.preprocessor.ocean_support_mask, bool)[None, None]
        net = getattr(model.autoencoder, component)
        mask = (
            torch.tensor(valid[:, 0], device=model.device) if net.mask_aware else None
        )
        rng = np.random.default_rng(baseline["seed"] + 1000 + ci)
        draws = []
        for member in range(baseline["members"]):
            eps = rng.standard_normal(mu.shape).astype("float32")
            values = np.where(valid, mu + sd * eps, np.nan).astype("float32")
            z = net.encode(
                torch.tensor(values, device=model.device),
                months,
                force,
                node_mask=mask,
            )
            z = model.latent_normalizers[component].normalize(z)
            if not torch.isfinite(z).all():
                raise FloatingPointError("Moment-matched encoding is not finite")
            draws.append(z)
            for vi, name in enumerate(model.names[component]):
                for ti, month in enumerate(month_numbers):
                    v = valid[ti, vi]
                    x = values[ti, vi][v]
                    audit.append(
                        dict(
                            component=component,
                            variable=name,
                            member=member + 1,
                            month=int(month),
                            finite_cells=int(v.sum()),
                            input_mean=float(x.mean()),
                            input_std=float(x.std()),
                            epsilon_mean=float(eps[ti, vi][v].mean()),
                            epsilon_std=float(eps[ti, vi][v].std()),
                        )
                    )
        f.history[ci] = torch.stack(draws)
    pd.DataFrame(audit).to_csv(output / "initial_input_audit.csv", index=False)
    return f.state_dict()


@torch.inference_mode()
def run_experiment(model, initial, baseline, baseline_path, output, stop_year=30):
    forecasts = {}
    for method in METHODS:
        f = model.initialize(
            co2=baseline["co2_ppm"],
            members=baseline["members"],
            seed=baseline["seed"],
            gaussian_start=baseline["first"],
        )
        f.load_state_dict(copy.deepcopy(initial))
        forecasts[method] = f
    shards = sorted(output.glob("year-*.pt"))
    if [p.name for p in shards] != [
        f"year-{y:03d}.pt" for y in range(1, len(shards) + 1)
    ]:
        raise ValueError("Missing annual block")
    if shards:
        saved = load_checkpoint(shards[-1], map_location="cpu", weights_only=False)
        for method, f in forecasts.items():
            f.load_state_dict(saved["states"][method])
            if f.step != len(shards) * 12:
                raise ValueError("Restart step mismatch")
    tick = time.monotonic()
    for year in range(len(shards) + 1, stop_year + 1):
        old = load_checkpoint(
            baseline_path / f"year-{year:03d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        predicted = {c: [] for c in COMPONENTS}
        feedback = {c: [] for c in COMPONENTS}
        dates, physical, corrections = [], [], []
        for month in range(12):
            step = 12 * (year - 1) + month + 1
            data = []
            for method, f in forecasts.items():
                previous = [h[:, 0].clone() for h in f.history]
                za, zo = next(f)
                before = [h[:, 0].detach().cpu().clone() for h in f.history]
                a, o = model.decode(za, zo, co2=baseline["co2_ppm"])
                idx = physical_indices(a, o)
                if not bool(np.isfinite(idx.to_array()).all()):
                    raise FloatingPointError("Nonfinite physical index")
                if method == "matched_cycle30" and step <= baseline["cycle_months"]:
                    encoded = [[], []]
                    for member in range(baseline["members"]):
                        pair = model.encode(
                            a.isel(member=member, drop=True),
                            o.isel(member=member, drop=True),
                            co2=baseline["co2_ppm"],
                        )
                        for ci, z in enumerate(pair):
                            encoded[ci].append(z.values[0])
                    for ci in range(2):
                        f.history[ci][:, 0] = torch.tensor(
                            np.stack(encoded[ci]), device=model.device
                        )
                after = [h[:, 0].detach().cpu().clone() for h in f.history]
                for h, prev in zip(f.history, previous):
                    torch.testing.assert_close(h[:, 1], prev, rtol=0, atol=0)
                    if not torch.isfinite(h).all():
                        raise FloatingPointError("Nonfinite latent")
                for n, common in zip(f.noise, forecasts["matched"].noise):
                    torch.testing.assert_close(n, common, rtol=0, atol=0)
                if f.times[0] != old["time"][month]:
                    raise ValueError("Calendar mismatch with baseline")
                data.append(
                    dict(
                        before=before,
                        after=after,
                        physical=np.column_stack(
                            (idx.nino34.values[0], idx.pressure_difference.values[0])
                        ),
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
                del a, o, idx
            for ci, c in enumerate(COMPONENTS):
                predicted[c].append(np.stack([d["before"][ci].numpy() for d in data]))
                feedback[c].append(np.stack([d["after"][ci].numpy() for d in data]))
            physical.append(np.stack([d["physical"] for d in data]))
            corrections.append(np.stack([d["correction"] for d in data]))
            dates.append(f.times[0])
        # End-of-year noise and RNG must exactly equal the prior experiment.
        # This also detects hidden RNG draws or a changed sampler/noise process.
        old_state = old["states"]["latent_only"]
        for f in forecasts.values():
            for n, old_n in zip(f.noise, old_state["noise"]):
                torch.testing.assert_close(n.cpu(), old_n.cpu(), rtol=0, atol=0)
            for key in ("torch", "mps"):
                if key in old_state["rng"]:
                    torch.testing.assert_close(
                        f.rng[key], old_state["rng"][key], rtol=0, atol=0
                    )
        atomic_save(
            dict(
                time=np.asarray(dates),
                methods=METHODS,
                predicted={c: np.stack(v) for c, v in predicted.items()},
                fed_back={c: np.stack(v) for c, v in feedback.items()},
                physical=np.stack(physical),
                correction=np.stack(corrections),
                states={m: f.state_dict() for m, f in forecasts.items()},
            ),
            output / f"year-{year:03d}.pt",
        )
        elapsed = time.monotonic() - tick
        print(
            f"Year {year}/30 | {elapsed / (year - len(shards)):.1f}s/year", flush=True
        )


def analyze(output, baseline_path, cache, probes, physical_reference):
    old = xr.load_dataset(baseline_path / "index_trajectories.nc")
    shards = [
        load_checkpoint(p, map_location="cpu", weights_only=False)
        for p in sorted(output.glob("year-*.pt"))
    ]
    dates = np.concatenate([s["time"] for s in shards])
    np.testing.assert_array_equal(dates, old.time.values)
    months = pd.DatetimeIndex(dates).month.values
    training_time = np.asarray(cache["training_time"])
    trmonth = pd.DatetimeIndex(training_time).month.values
    variables = {}
    for key in ("predicted", "fed_back"):
        readouts = []
        for c in ("ocean", "atmosphere"):
            z = np.concatenate([s[key][c] for s in shards]).astype(float)
            train = cache["training"][c].numpy().astype(float)
            climate = np.stack([train[trmonth == m].mean(0) for m in range(1, 13)])
            readouts.append(
                (z - climate[months - 1, None, None]).reshape(*z.shape[:3], -1)
                @ probes[c]
            )
        name = "latent" if key == "predicted" else "fed_back_latent"
        variables[name] = (
            ("time", "method", "member", "component"),
            np.stack(readouts, -1),
        )
    ref = xr.load_dataset(physical_reference)
    np.testing.assert_array_equal(ref.time.values, training_time)
    r = np.column_stack((ref.nino34.values, -ref.pressure_difference.values))
    phys = np.concatenate([s["physical"] for s in shards])
    phys[..., 1] *= -1
    for m in range(1, 13):
        phys[months == m] -= r[trmonth == m].mean(0)
    variables["physical"] = (("time", "method", "member", "component"), phys)
    new = xr.Dataset(
        variables,
        coords=dict(
            time=dates, method=list(METHODS), member=old.member, component=old.component
        ),
    )
    ds = xr.concat([old[list(variables)], new], dim="method", join="exact")
    ds["latent_reference"] = old.latent_reference
    ds["physical_reference"] = old.physical_reference
    ds.attrs.update(
        old.attrs,
        initialization="Local training-month moments; original arms reused unchanged",
    )
    # The source NetCDF's fixed-width Unicode encoding is too short for the
    # newly appended method names. Infer encoding afresh, without truncation.
    ds["method"].encoding = {}
    ds.to_netcdf(
        output / "index_trajectories.nc",
        encoding={v: dict(zlib=True, complevel=3) for v in ds},
    )
    with xr.open_dataset(output / "index_trajectories.nc") as saved:
        np.testing.assert_array_equal(saved["method"].values, ds["method"].values)
    rows, curves = [], []
    for kind in ("latent", "physical"):
        reference = ds[kind + "_reference"].values
        for period, sl in (
            ("years01_05", slice(0, 60)),
            ("years06_15", slice(60, 180)),
            ("years11_30", slice(120, 360)),
        ):
            for method in ds.method.values:
                for member in ds.member.values:
                    pair = ds[kind].sel({"method": method, "member": member}).values[sl]
                    for treatment in ("fixed_training_month", "own_month_detrended"):
                        x, r = pair, reference
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
    late = table[(table.period == "years11_30")]
    for kind in ("latent", "physical"):
        for treatment in ("fixed_training_month", "own_month_detrended"):
            sub = late[(late.kind == kind) & (late.anomalies == treatment)]
            for candidate, control in (
                ("matched", "latent_only"),
                ("matched_cycle30", "cycle30"),
                ("matched_cycle30", "matched"),
            ):
                a = sub[sub.method == candidate].set_index("member")
                b = sub[sub.method == control].set_index("member")
                for member in a.index:
                    paired.append(
                        dict(
                            kind=kind,
                            anomalies=treatment,
                            candidate=candidate,
                            control=control,
                            member=member,
                            **{
                                v: float(a.loc[member, v] - b.loc[member, v])
                                for v in (
                                    "ocean_sd_ratio",
                                    "ocean_acf24",
                                    "correlation",
                                    "relationship_rmse_ratio",
                                )
                            },
                        )
                    )
    pd.DataFrame(paired).to_csv(output / "paired_differences.csv", index=False)
    old_initial = load_checkpoint(
        baseline_path / "initial_states.pt", map_location="cpu", weights_only=False
    )["latent_only"]
    new_initial = load_checkpoint(
        output / "initial_state.pt", map_location="cpu", weights_only=False
    )
    initial_rows = []
    for ci, c in enumerate(COMPONENTS):
        for ti, m in enumerate((12, 11)):
            ref_latents = cache["training"][c].numpy()[trmonth == m]
            rms_sd = np.sqrt(ref_latents.var(0, ddof=1).mean())
            for method, state in (
                ("latent_only", old_initial),
                ("matched", new_initial),
            ):
                z = state["history"][ci][:, ti].numpy()
                initial_rows.append(
                    dict(
                        component=c,
                        month=m,
                        method=method,
                        mean_pattern_error_in_reference_sd=float(
                            np.sqrt(((z.mean(0) - ref_latents.mean(0)) ** 2).mean())
                            / rms_sd
                        ),
                        initial_spread_ratio=float(
                            np.sqrt(z.var(0, ddof=1).mean()) / rms_sd
                        ),
                    )
                )
    pd.DataFrame(initial_rows).to_csv(
        output / "initial_latent_comparison.csv", index=False
    )
    mask_rows = []
    for path in sorted((output / "monthly_moments").glob("*.nc")):
        stats = xr.load_dataset(path)
        ever_valid = stats["count"].sum("month") > 0
        for month in (11, 12):
            mask_rows.append(
                dict(
                    variable=path.stem,
                    month=month,
                    ever_valid_cells=int(ever_valid.sum()),
                    additional_unavailable_cells=int(
                        (ever_valid & (stats["count"].sel(month=month) < 2)).sum()
                    ),
                )
            )
    pd.DataFrame(mask_rows).to_csv(output / "initial_support_audit.csv", index=False)
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
                xlabel="Fixed-reference SD; offsets retained",
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
        fig.suptitle(f"{kind.title()} ENSO · years 11–30 · member-mean ACF and spectra")
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
                ax.axvline(2.5, color=".5", ls=":", lw=0.8)
                ax.grid(alpha=0.12)
                if j == 0:
                    ax.set_title(LABELS[method].replace(" + ", "\n+ "), fontsize=9)
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
    fig, axes = plt.subplots(
        2,
        len(DISPLAY),
        figsize=(16, 7),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    for row, kind in enumerate(("latent", "physical")):
        reference = ds[kind + "_reference"].values
        r = reference / reference.std(0)
        slope, intercept = np.polyfit(r[:, 0], r[:, 1], 1)
        residual_sd = (r[:, 1] - intercept - slope * r[:, 0]).std()
        xx = np.array([-4.0, 4.0])
        for col, method in enumerate(DISPLAY):
            x = ds[kind].sel({"method": method}).isel(
                time=slice(120, 360)
            ).values.reshape(-1, 2) / reference.std(0)
            ax = axes[row, col]
            ax.scatter(r[:, 0], r[:, 1], s=5, c=".55", alpha=0.2, rasterized=True)
            ax.fill_between(
                xx,
                intercept + slope * xx - residual_sd,
                intercept + slope * xx + residual_sd,
                color=".4",
                alpha=0.1,
            )
            ax.plot(xx, intercept + slope * xx, color="black", lw=0.8)
            ax.scatter(
                x[:, 0], x[:, 1], s=5, color=COLORS[method], alpha=0.25, rasterized=True
            )
            ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal")
            ax.grid(alpha=0.12)
            if row == 0:
                ax.set_title(LABELS[method].replace(" + ", "\n+ "), fontsize=9)
            if col == 0:
                ax.set_ylabel(
                    "Atmospheric ENSO readout\nReference SD"
                    if kind == "latent"
                    else "−SOI proxy\nReference SD"
                )
            if row == 1:
                ax.set_xlabel("Ocean / Niño3.4 · reference SD")
    fig.suptitle(
        "ENSO atmosphere–ocean relationships · years 11–30\nGray: encoded reference; black: training regression ±1 residual SD; colored: five rollout members"
    )
    for ext in ("png", "pdf"):
        fig.savefig(output / f"joint_relationships.{ext}", dpi=190)
    plt.close(fig)
    (output / "analysis_configuration.json").write_text(
        json.dumps(
            dict(
                primary="Fixed training monthly anomalies; no trend or ensemble mean subtraction; years 11–30",
                sensitivity="Own-window monthly climatology and linear trend jointly removed independently per member; same operation on reference",
                reference="Training encoded latents; physical reference decoded at same fixed 1850 CO2; not preindustrial observations",
                physical_indices="Area-weighted Niño3.4 SST and negative Tahiti-minus-Darwin pressure proxy",
                aggregation="Individual member time statistics then member mean; only marginal/joint display pools members",
                limitations="Five members, one checkpoint, one fixed CO2; no claim of causal coupling or all-field physical realism",
                support="D20 cells without >=2 finite samples in the initialization calendar month remain missing; see initial_support_audit.csv",
            ),
            indent=2,
        )
    )
    print(
        summary[
            (summary.period == "years11_30")
            & (summary.anomalies == "fixed_training_month")
        ][
            [
                "kind",
                "method",
                "correlation",
                "ocean_sd_ratio",
                "ocean_acf24",
                "relationship_rmse_ratio",
            ]
        ].to_string(index=False),
        flush=True,
    )
