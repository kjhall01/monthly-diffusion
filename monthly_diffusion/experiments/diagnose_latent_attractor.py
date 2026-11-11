"""Paired Gaussian/encoded initialization audit of the frozen latent dynamics.

Run 150 years, ten members, from January 1959. Encoded histories are November
and December 1958 from EACH model's own source/cache, ordered current/previous.
Both groups receive identical diffusion innovations and initially unset noise
histories. No physical decoding, ensemble-mean subtraction, or retraining.

Readouts control training monthly intercepts and linear trend when fitting.
They do NOT extrapolate that trend over the synthetic 150-year run. Report raw
projections, fixed training-month anomalies, and independent window-month and
linear-trend removal. Raw channel distributions and training PCA projections
are additional checks: matching one readout is not proof of attractor recovery.

Both trained transitions are autonomous with respect to CO2. The iterator's
required CO2 argument is held at training mean, unused without decoding. The
future calendar labels therefore indicate elapsed time, not a forced projection.
Yearly atomic latent/state shards permit exact resume without physical Zarrs.
"""

from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats
import torch
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[2]
from monthly_diffusion import MonthlyDiffusionV1
from monthly_diffusion.forecast import Forecast
from monthly_diffusion.training.common import atomic_save
from monthly_diffusion.data.sources import open_canonical_state
from monthly_diffusion.analysis.indices import (
    enso_index,
    nao_index,
    pdo_index,
    pdo_loading,
)
from monthly_diffusion.analysis.spectra import lag_autocorrelation
from monthly_diffusion.experiments.diagnose_enso_initialization import (
    fit_readout,
    design,
    metrics,
)

COMPONENTS = ("atmosphere", "ocean")
MODES = ("ENSO", "NAO", "PDO")
INITIALS = ("encoded", "gaussian")
LABELS = dict(
    encoded="1958 encoded history",
    gaussian="IID Gaussian history",
    reference="Encoded training reference",
)
COLORS = dict(encoded="#0072B2", gaussian="#D55E00", reference="#222222")
RUNS = dict(era=ROOT / "runs/mdv1-era5-oras5", e3sm=ROOT / "runs/mdv1-e3smv2-r1i1p1f1")
TITLES = dict(era="ERA5/ORAS5", e3sm="E3SMv2")


def anomalies(values, dates, method, training_monthly=None):
    """Preserve offsets in fixed-climatology mode; never subtract ensemble mean."""
    values = np.asarray(values, dtype=float)
    months = pd.DatetimeIndex(dates).month.to_numpy() - 1
    if method == "fixed_training_monthly":
        return values - np.asarray(training_monthly)[months, None]
    matrix = np.eye(12)[months]
    if method == "window_monthly_detrend":
        matrix = np.column_stack((matrix, np.linspace(-1, 1, len(values))))
    elif method != "window_monthly":
        raise ValueError(method)
    return values - matrix @ np.linalg.lstsq(matrix, values, rcond=None)[0]


def curve_statistics(values):
    """Memberwise spectra/ACF; never join end of one member to next member."""
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite readout")
    x = xr.DataArray(values.T, dims=("member", "time"))
    acf = lag_autocorrelation(x, max_lag=120, detrend="constant").values
    frequency, psd = signal.welch(
        values, fs=1.0, nperseg=min(240, len(values)), axis=0, detrend="constant"
    )
    return frequency, psd.T, acf


def paired_rollout(model, cache, output, *, years, members, seed):
    """Persist yearly outputs together with their exact subsequent forecast state."""
    dates = np.asarray(cache["validation_time"]).astype("datetime64[M]")
    positions = [
        int(np.flatnonzero(dates == np.datetime64(d, "M"))[0])
        for d in ("1958-12", "1958-11")
    ]
    for kind in INITIALS:
        folder = output / kind
        folder.mkdir(exist_ok=True)
        forecast = Forecast(
            model,
            None,
            None,
            co2=model.preprocessor.co2_mean,
            members=members,
            seed=seed,
            gaussian_start="1959-01",
        )
        if kind == "encoded":
            forecast.history = [
                cache["validation"][c][positions]
                .unsqueeze(0)
                .expand(members, -1, -1, -1)
                .clone()
                .to(model.device)
                for c in COMPONENTS
            ]
        if not (folder / "initial.pt").exists():
            atomic_save(forecast.state_dict(), folder / "initial.pt")
        shards = sorted(folder.glob("year-*.pt"))
        if [p.name for p in shards] != [
            f"year-{i:03d}.pt" for i in range(1, len(shards) + 1)
        ]:
            raise ValueError("Missing yearly shard")
        if shards:
            saved = torch.load(shards[-1], map_location="cpu", weights_only=False)
            forecast.load_state_dict(saved["state"])
            assert forecast.step == len(shards) * 12
        tick = time.monotonic()
        start = forecast.step
        for year in range(len(shards) + 1, years + 1):
            latents = {c: [] for c in COMPONENTS}
            times = []
            for _ in range(12):
                pair = next(forecast)
                expected = np.datetime64("1959-01", "M") + np.timedelta64(
                    forecast.step - 1, "M"
                )
                assert pair[0].time.values[0].astype("datetime64[M]") == expected
                times.append(pair[0].time.values[0])
                for c, z in zip(COMPONENTS, pair):
                    if not np.isfinite(z).all():
                        raise FloatingPointError("Unstable latent rollout")
                    latents[c].append(z.values[0])
            atomic_save(
                dict(
                    time=np.array(times),
                    latents={c: np.stack(x) for c, x in latents.items()},
                    state=forecast.state_dict(),
                ),
                folder / f"year-{year:03d}.pt",
            )
            if year % 5 == 0 or year == years:
                print(
                    f"{output.name} {kind}: year {year}/{years}; {(time.monotonic() - tick) / max(1, forecast.step - start):.3f} s/month",
                    flush=True,
                )


def prepare_probes(model, cache, source, output):
    target_path = output / "physical_indices.nc"
    dt = np.asarray(cache["training_time"])
    dv = np.asarray(cache["validation_time"])
    if not target_path.exists():
        a, o, _ = open_canonical_state(
            source["state_zarr"], expected_attrs=source.get("expected_attrs")
        )
        a = a.sel(time=slice("1958", "2006"))
        o = o.sel(time=slice("1958", "2006"))
        enso = enso_index(
            o.sel(lat=slice(-5, 5), lon=slice(190, 240)), remove_monthly=False
        ).compute()
        nao = (
            nao_index(
                a[["mslp"]].sel(lat=slice(35, 70), lon=slice(330, 360)),
                remove_monthly=False,
            )
            / 100.0
        ).compute()
        region = o[["sst"]].sel(lat=slice(20, 70), lon=slice(110, 260)).load()
        train = region.sel(time=dt)
        values = train.sst.values.reshape(len(dt), -1).astype(float)
        valid = np.isfinite(values).all(0)
        d = design(dt)
        residual = np.full_like(values, np.nan)
        residual[:, valid] = (
            values[:, valid] - d @ np.linalg.lstsq(d, values[:, valid], rcond=None)[0]
        )
        anomaly = xr.Dataset(
            dict(sst=(train.sst.dims, residual.reshape(train.sst.shape))),
            coords=train.coords,
        )
        loading = pdo_loading(anomaly, remove_monthly=False)
        loading.attrs = {
            k: int(v) if isinstance(v, (bool, np.bool_)) else v
            for k, v in loading.attrs.items()
        }
        loading.to_netcdf(output / "pdo_loading.nc")
        pdo = pdo_index(region, loading=loading, remove_monthly=False).compute()
        xr.Dataset(dict(ENSO=enso, NAO=nao, PDO=pdo)).to_netcdf(target_path)
        a.close()
        o.close()
    targets = xr.load_dataset(target_path)
    probes = {}
    rows = []
    for component in COMPONENTS:
        x = cache["training"][component].numpy().reshape(len(dt), -1)
        for mode in MODES:
            path = output / f"probe_{component}_{mode}.npz"
            if path.exists():
                probe = dict(np.load(path))
            else:
                probe = fit_readout(x, targets[mode].sel(time=dt).values, dt)
                projections = x @ probe["coef"]
                probe["projection_monthly"] = np.array(
                    [
                        projections[pd.DatetimeIndex(dt).month == month].mean()
                        for month in range(1, 13)
                    ]
                )
                np.savez(path, **probe)
            probes[component, mode] = probe
            for split, dates in (("training", dt), ("validation", dv)):
                xx = cache[split][component].numpy().reshape(len(dates), -1)
                pred = (xx - design(dates) @ probe["latent_baseline"]) @ probe["coef"]
                truth = (
                    targets[mode].sel(time=dates).values
                    - design(dates) @ probe["physical_baseline"]
                )
                rows.append(
                    dict(
                        component=component,
                        mode=mode,
                        split=split,
                        alpha=float(probe["alpha"]),
                        **metrics(pred, truth),
                    )
                )
            rows.append(
                dict(
                    component=component,
                    mode=mode,
                    split="blocked_cv_alpha_selection",
                    alpha=float(probe["alpha"]),
                    **metrics(probe["cv_prediction"], probe["cv_truth"]),
                )
            )
    pd.DataFrame(rows).to_csv(output / "readout_skill.csv", index=False)
    print(
        pd.DataFrame(rows).query("split=='validation'").to_string(index=False),
        flush=True,
    )
    return probes, targets


def initialization_statistics(cache, output):
    """Channel standardization need not remove the spatial mean latent field."""
    rows = []
    for component in COMPONENTS:
        z = cache["training"][component].numpy().astype(float)
        adjacent_corr = float(np.corrcoef(z[1:].ravel(), z[:-1].ravel())[0, 1])
        for ch in range(z.shape[-1]):
            x = z[..., ch]
            total = x.var()
            rows.append(
                dict(
                    component=component,
                    channel=ch + 1,
                    pooled_variance=total,
                    fixed_spatial_variance_fraction=float(x.mean(0).var() / total),
                    temporal_variance_fraction=float(x.var(0).mean() / total),
                    raw_adjacent_month_correlation_all_channels=adjacent_corr,
                )
            )
    pd.DataFrame(rows).to_csv(output / "initialization_statistics.csv", index=False)


def load_latents(output, kind):
    blocks = [
        torch.load(p, map_location="cpu", weights_only=False)
        for p in sorted((output / kind).glob("year-*.pt"))
    ]
    return xr.Dataset(
        {
            c: (
                ("time", "member", "node", "channel"),
                np.concatenate([b["latents"][c] for b in blocks]),
            )
            for c in COMPONENTS
        },
        coords=dict(
            time=np.concatenate([b["time"] for b in blocks]),
            member=range(blocks[0]["latents"]["ocean"].shape[1]),
        ),
    )


def finish(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def overview(output):
    """Compact summaries and paired-path checks, independent of anomaly fitting."""
    table = pd.read_csv(output / "window_metrics.csv")
    comparison = pd.read_csv(output / "initialization_differences.csv")
    primary = table[table.anomalization == "window_monthly_detrend"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), constrained_layout=True)
    for i, component in enumerate(COMPONENTS):
        for j, mode in enumerate(MODES):
            ax = axes[i, j]
            for kind in INITIALS:
                rows = primary[
                    (primary.component == component)
                    & (primary["mode"] == mode)
                    & (primary.initialization == kind)
                ]
                x = (rows.first_lead_year + rows.last_lead_year) / 2
                ax.plot(x, rows.std_ratio, "o-", color=COLORS[kind], label=LABELS[kind])
            ax.axhline(
                1, color="black", ls="--", lw=0.8, label="Encoded training variability"
            )
            ax.set(
                title=f"{component} · {mode}",
                xlabel="Years since initialization",
                ylabel="Std / encoded training std",
            )
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.15)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        TITLES[output.name]
        + " · 30-year variability windows\nMonthly cycle and linear trend removed per member/window; no ensemble-mean subtraction",
        fontsize=12,
    )
    finish(fig, output / "mode_amplitude_overview")
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), constrained_layout=True)
    for i, component in enumerate(COMPONENTS):
        for j, mode in enumerate(MODES):
            ax = axes[i, j]
            for method, label in [
                ("fixed_training_monthly", "Fixed training monthly climatology"),
                ("window_monthly", "Window monthly climatology"),
                ("window_monthly_detrend", "Window monthly + linear trend"),
            ]:
                rows = comparison[
                    (comparison.component == component)
                    & (comparison["mode"] == mode)
                    & (comparison.anomalization == method)
                ]
                ax.plot(
                    (rows.first_lead_year + rows.last_lead_year) / 2,
                    rows.gaussian_over_encoded_std,
                    "o-",
                    label=label,
                )
            ax.axhline(1, color="black", lw=0.8)
            ax.set(
                title=f"{component} · {mode}",
                xlabel="Years since initialization",
                ylabel="Std: Gaussian / encoded init",
            )
            ax.grid(alpha=0.15)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        TITLES[output.name] + " · sensitivity to anomaly definition", fontsize=12
    )
    finish(fig, output / "anomaly_definition_sensitivity")
    ensembles = {k: load_latents(output, k) for k in INITIALS}
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    distances = {}
    checks = []
    for ax, c in zip(axes, COMPONENTS):
        difference = ensembles["gaussian"][c] - ensembles["encoded"][c]
        rms = np.sqrt((difference**2).mean(("node", "channel")))
        distances[c] = rms
        yearly = rms.coarsen(time=12).mean()
        for m in yearly.member.values:
            ax.semilogy(
                np.arange(1, len(yearly.time) + 1),
                yearly.sel(member=m).clip(min=1.0e-9),
                lw=0.8,
            )
        ax.set(
            title=c,
            xlabel="Years since initialization",
            ylabel="Paired latent RMS difference",
        )
        ax.grid(alpha=0.15)
    fig.suptitle(
        TITLES[output.name]
        + " · same-noise path separation\nPath synchronization is stronger than distributional agreement; lack of synchronization alone does not prove different attractors",
        fontsize=11,
    )
    finish(fig, output / "paired_latent_separation")
    xr.Dataset(distances).to_netcdf(output / "paired_latent_separation.nc")
    for path in sorted((output / "encoded").glob("year-*.pt")):
        a = torch.load(path, map_location="cpu", weights_only=False)["state"]
        b = torch.load(
            output / "gaussian" / path.name, map_location="cpu", weights_only=False
        )["state"]
        checks.append(
            dict(
                year=int(path.stem.split("-")[1]),
                times_identical=a["times"] == b["times"],
                atmosphere_noise_max_difference=float(
                    (a["noise"][0] - b["noise"][0]).abs().max()
                ),
                ocean_noise_max_difference=float(
                    (a["noise"][1] - b["noise"][1]).abs().max()
                ),
            )
        )
    pd.DataFrame(checks).to_csv(output / "paired_noise_checks.csv", index=False)


def analyze(cache, probes, targets, output, *, years):
    """Compare five independent 30-year windows; member bootstrap is descriptive."""
    ensembles = {kind: load_latents(output, kind) for kind in INITIALS}
    dates = ensembles["encoded"].time.values
    train_dates = np.asarray(cache["training_time"])
    windows = [(lo, min(lo + 30, years)) for lo in range(0, years, 30)]
    if any(hi - lo < 20 for lo, hi in windows):
        raise ValueError("Use full windows of at least twenty years")
    rows = []
    member_rows = []
    difference_rows = []
    curves = []
    readouts = {}
    channel_rows = []
    pc_rows = []
    rng = np.random.default_rng(724)
    for component in COMPONENTS:
        for mode in MODES:
            probe = probes[component, mode]
            train = (
                cache["training"][component].numpy().reshape(len(train_dates), -1)
                @ probe["coef"]
            )
            projected = {
                k: ds[component].values.reshape(len(dates), len(ds.member), -1)
                @ probe["coef"]
                for k, ds in ensembles.items()
            }
            for k, values in projected.items():
                readouts[f"{k}_{component}_{mode}"] = (
                    ("time", "member"),
                    values
                    - probe["projection_monthly"][
                        pd.DatetimeIndex(dates).month - 1, None
                    ],
                )
            primary = {}
            for method in (
                "fixed_training_monthly",
                "window_monthly",
                "window_monthly_detrend",
            ):
                reference = anomalies(
                    train[:, None], train_dates, method, probe["projection_monthly"]
                )
                reference_std = float(np.sqrt(reference.var(0).mean()))
                rf, rp, ra = curve_statistics(reference)
                for lo, hi in windows:
                    sl = slice(lo * 12, hi * 12)
                    member_stats = {}
                    for kind, values in projected.items():
                        x = anomalies(
                            values[sl], dates[sl], method, probe["projection_monthly"]
                        )
                        f, psd, acf = curve_statistics(x)
                        sd = x.std(0)
                        pooled_std = x.std()
                        band = (f >= 1 / 60) & (f <= 1 / 24)
                        row = dict(
                            component=component,
                            mode=mode,
                            anomalization=method,
                            initialization=kind,
                            first_lead_year=lo + 1,
                            last_lead_year=hi,
                            mean=float(x.mean()),
                            pooled_std=float(pooled_std),
                            std=float(np.sqrt(np.mean(sd**2))),
                            reference_std=reference_std,
                            std_ratio=float(np.sqrt(np.mean(sd**2)) / reference_std),
                            member_offset_std=float(x.mean(0).std()),
                            acf1=float(acf[:, 1].mean()),
                            acf12=float(acf[:, 12].mean()),
                            acf24=float(acf[:, 24].mean()),
                            min_acf_12_60=float(acf[:, 12:61].mean(0).min()),
                            minimum_acf_lag=int(12 + acf[:, 12:61].mean(0).argmin()),
                            two_to_five_year_power_fraction=float(
                                psd[:, band].sum() / psd[:, 1:].sum()
                            ),
                            wasserstein_vs_encoded_reference=float(
                                stats.wasserstein_distance(x.ravel(), reference.ravel())
                                / reference_std
                            ),
                        )
                        rows.append(row)
                        member_stats[kind] = (sd, acf, psd, x)
                        for i in range(len(sd)):
                            member_rows.append(
                                dict(
                                    component=component,
                                    mode=mode,
                                    anomalization=method,
                                    initialization=kind,
                                    first_lead_year=lo + 1,
                                    last_lead_year=hi,
                                    member=i,
                                    mean=float(x[:, i].mean()),
                                    std=float(sd[i]),
                                    acf24=float(acf[i, 24]),
                                )
                            )
                        if method == "window_monthly_detrend":
                            primary[kind, lo] = (x, f, psd, acf)
                            curves.append(
                                xr.Dataset(
                                    dict(
                                        psd=(("member", "frequency"), psd),
                                        acf=(("member", "lag"), acf),
                                    ),
                                    coords=dict(
                                        member=range(len(sd)),
                                        frequency=f,
                                        lag=range(121),
                                    ),
                                ).expand_dims(
                                    component=[component],
                                    mode=[mode],
                                    initialization=[kind],
                                    window=[lo + 1],
                                )
                            )
                    es, ea, ep, ex = member_stats["encoded"]
                    gs, ga, gp, gx = member_stats["gaussian"]
                    bootstrap = rng.integers(0, len(es), (2000, len(es)))
                    ratios = np.sqrt(
                        (gs[bootstrap] ** 2).mean(1) / (es[bootstrap] ** 2).mean(1)
                    )
                    acfdelta = (ga[bootstrap, 24] - ea[bootstrap, 24]).mean(1)
                    difference_rows.append(
                        dict(
                            component=component,
                            mode=mode,
                            anomalization=method,
                            first_lead_year=lo + 1,
                            last_lead_year=hi,
                            gaussian_over_encoded_std=float(
                                np.sqrt(np.mean(gs**2) / np.mean(es**2))
                            ),
                            std_ratio_ci_low=float(np.quantile(ratios, 0.025)),
                            std_ratio_ci_high=float(np.quantile(ratios, 0.975)),
                            gaussian_minus_encoded_acf24=float(
                                (ga[:, 24] - ea[:, 24]).mean()
                            ),
                            acf24_difference_ci_low=float(np.quantile(acfdelta, 0.025)),
                            acf24_difference_ci_high=float(
                                np.quantile(acfdelta, 0.975)
                            ),
                            acf_curve_rmse=float(
                                np.sqrt(np.mean((ga.mean(0)[1:] - ea.mean(0)[1:]) ** 2))
                            ),
                            log_psd_rmse=float(
                                np.sqrt(
                                    np.mean(
                                        (
                                            np.log(gp.mean(0)[1:])
                                            - np.log(ep.mean(0)[1:])
                                        )
                                        ** 2
                                    )
                                )
                            ),
                            wasserstein_gaussian_vs_encoded=float(
                                stats.wasserstein_distance(gx.ravel(), ex.ravel())
                                / reference_std
                            ),
                        )
                    )
            # Five window rows, with distributions, spectra and ACF in columns.
            fig, axes = plt.subplots(
                len(windows),
                3,
                figsize=(13, 2.65 * len(windows)),
                constrained_layout=True,
                squeeze=False,
            )
            reference = anomalies(train[:, None], train_dates, "window_monthly_detrend")
            rf, rp, ra = curve_statistics(reference)
            all_values = np.concatenate(
                [v[0].ravel() for v in primary.values()] + [reference.ravel()]
            )
            limits = np.quantile(all_values, [0.001, 0.999])
            density_grid = np.linspace(*limits, 250)
            for i, (lo, hi) in enumerate(windows):
                for kind in (*INITIALS, "reference"):
                    x, f, psd, acf = (
                        primary[kind, lo]
                        if kind != "reference"
                        else (reference, rf, rp, ra)
                    )
                    axes[i, 0].plot(
                        density_grid,
                        stats.gaussian_kde(x.ravel())(density_grid),
                        color=COLORS[kind],
                        label=LABELS[kind],
                        lw=1.3,
                    )
                    axes[i, 1].plot(
                        1 / (12 * f[1:]), psd.mean(0)[1:], color=COLORS[kind], lw=1.3
                    )
                    axes[i, 2].plot(
                        np.arange(121) / 12, acf.mean(0), color=COLORS[kind], lw=1.3
                    )
                    if kind != "reference":
                        axes[i, 2].fill_between(
                            np.arange(121) / 12,
                            *np.quantile(acf, [0.1, 0.9], axis=0),
                            color=COLORS[kind],
                            alpha=0.08,
                        )
                axes[i, 0].set_ylabel(f"Years {lo + 1}–{hi}\nDensity")
                axes[i, 1].set(xscale="log", yscale="log", xlim=(0.16, 20))
                axes[i, 2].axhline(0, color=".5", lw=0.6)
                axes[i, 2].set_ylim(-0.7, 1.0)
                for ax in axes[i]:
                    ax.grid(alpha=0.16)
            for ax, title in zip(
                axes[0], ("Distribution", "Power spectrum", "Lag autocorrelation")
            ):
                ax.set_title(title)
            for ax, label in zip(
                axes[-1],
                (
                    "Readout (hPa)" if mode == "NAO" else "Readout (°C-equivalent)",
                    "Period (years)",
                    "Lag (years)",
                ),
            ):
                ax.set_xlabel(label)
            fig.legend(
                *axes[0, 0].get_legend_handles_labels(),
                loc="outside lower center",
                ncol=3,
                frameon=False,
            )
            fig.suptitle(
                f"{TITLES[output.name]} · {component} {mode}\nWithin-window monthly cycle and trend removed separately for each member; no ensemble-mean subtraction",
                fontsize=12,
            )
            finish(fig, output / f"{component}_{mode}_windows")
        # Raw latent channel marginals plus fixed training PCA; NO anomalization.
        training = cache["training"][component].numpy()
        flat = training.reshape(len(training), -1).astype(float)
        mu = flat.mean(0)
        _, singular, vt = np.linalg.svd(flat - mu, full_matrices=False)
        pcs = vt[:16]
        pc_scale = singular[:16] / np.sqrt(len(flat))
        np.savez(
            output / f"{component}_pca.npz",
            mean=mu,
            components=pcs,
            training_std=pc_scale,
            explained_variance_fraction=singular[:16] ** 2 / np.sum(singular**2),
        )
        fig, axes = plt.subplots(
            len(windows),
            5,
            figsize=(16, 2.5 * len(windows)),
            constrained_layout=True,
            squeeze=False,
        )
        pc_reference = (flat - mu) @ pcs.T / pc_scale
        for i, (lo, hi) in enumerate(windows):
            projected = {}
            for kind, ds in ensembles.items():
                x = ds[component].isel(time=slice(lo * 12, hi * 12)).values
                projected[kind] = (
                    (x.reshape(-1, x.shape[-2] * x.shape[-1]) - mu) @ pcs.T / pc_scale
                )
                for ch in range(training.shape[-1]):
                    v = x[..., ch].ravel()
                    r = training[..., ch].ravel()
                    channel_rows.append(
                        dict(
                            component=component,
                            initialization=kind,
                            first_lead_year=lo + 1,
                            last_lead_year=hi,
                            channel=ch,
                            mean=float(v.mean()),
                            std=float(v.std()),
                            reference_mean=float(r.mean()),
                            reference_std=float(r.std()),
                            wasserstein_vs_encoded_reference=float(
                                stats.wasserstein_distance(v, r)
                            ),
                            skew=float(stats.skew(v)),
                            kurtosis_excess=float(stats.kurtosis(v)),
                        )
                    )
                for pc in range(16):
                    v = projected[kind][:, pc]
                    pc_rows.append(
                        dict(
                            component=component,
                            initialization=kind,
                            first_lead_year=lo + 1,
                            last_lead_year=hi,
                            pc=pc + 1,
                            mean=float(v.mean()),
                            std=float(v.std()),
                            wasserstein_vs_encoded_reference=float(
                                stats.wasserstein_distance(v, pc_reference[:, pc])
                            ),
                        )
                    )
            for ch in range(training.shape[-1]):
                for kind in (*INITIALS, "reference"):
                    v = (
                        training[..., ch].ravel()
                        if kind == "reference"
                        else ensembles[kind][component]
                        .isel(time=slice(lo * 12, hi * 12))
                        .values[..., ch]
                        .ravel()
                    )
                    axes[i, ch].hist(
                        v,
                        bins=np.linspace(-5, 5, 100),
                        density=True,
                        histtype="step",
                        color=COLORS[kind],
                        label=LABELS[kind],
                    )
                if i == 0:
                    axes[i, ch].set_title(f"Raw latent channel {ch + 1}")
                if ch == 0:
                    axes[i, ch].set_ylabel(f"Years {lo + 1}–{hi}\nDensity")
            for kind in (*INITIALS, "reference"):
                pc = pc_reference if kind == "reference" else projected[kind]
                axes[i, 4].scatter(
                    pc[:: max(1, len(pc) // 500), 0],
                    pc[:: max(1, len(pc) // 500), 1],
                    s=4,
                    alpha=0.25,
                    color=COLORS[kind],
                    rasterized=True,
                )
            axes[i, 4].set(xlabel="PC1 / training SD", ylabel="PC2 / training SD")
            if i == 0:
                axes[i, 4].set_title("Joint latent state")
        fig.legend(
            *axes[0, 0].get_legend_handles_labels(),
            loc="outside lower center",
            ncol=3,
            frameon=False,
        )
        fig.suptitle(
            f"{TITLES[output.name]} · {component} · RAW latent distributions\nFixed transition-normalized units; no mean/cycle/trend removal; channel histograms display −5 to +5",
            fontsize=12,
        )
        finish(fig, output / f"{component}_raw_latents")
    xr.Dataset(
        readouts, coords=dict(time=dates, member=ensembles["encoded"].member)
    ).to_netcdf(output / "readouts.nc")
    xr.combine_by_coords(curves, combine_attrs="drop_conflicts").to_netcdf(
        output / "spectra_acf.nc"
    )
    pd.DataFrame(rows).to_csv(output / "window_metrics.csv", index=False)
    pd.DataFrame(member_rows).to_csv(output / "member_metrics.csv", index=False)
    pd.DataFrame(difference_rows).to_csv(
        output / "initialization_differences.csv", index=False
    )
    pd.DataFrame(channel_rows).to_csv(output / "raw_channel_metrics.csv", index=False)
    pd.DataFrame(pc_rows).to_csv(output / "pca_metrics.csv", index=False)
    # All-member latent means expose persistent groups hidden by detrending.
    fig, axes = plt.subplots(2, 2, figsize=(13, 6), constrained_layout=True)
    for i, c in enumerate(COMPONENTS):
        for j, k in enumerate(INITIALS):
            annual = ensembles[k][c].mean(("node", "channel")).coarsen(time=12).mean()
            for member in annual.member.values:
                axes[i, j].plot(
                    np.arange(1, years + 1), annual.sel(member=member), lw=0.8
                )
            axes[i, j].set(
                title=f"{c} · {LABELS[k]}",
                xlabel="Years since initialization",
                ylabel="Raw latent mean",
            )
            axes[i, j].grid(alpha=0.15)
    finish(fig, output / "raw_latent_mean_trajectories")
    print("Analysis complete:", output, flush=True)
