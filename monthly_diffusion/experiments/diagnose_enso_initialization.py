"""Does observed initialization repair weak ENSO in a frozen trained model?

Fit ocean/atmosphere ridge readouts of physical Nino3.4 using training data
only. Monthly intercepts and a linear trend are removed from predictors and
target; blocked-year CV refits these nuisance terms inside every fold.
Validate the readouts on the already-cached held-out validation years.

Compare paired observed-history and IID-Gaussian-history latent forecasts.
Both use the package Forecast iterator, identical seeds, and initially unset
AR(1) noise histories. Only the initial two latent states differ. No decoding,
retraining, changed sampler, or ensemble-mean subtraction is involved.
"""

from pathlib import Path
import argparse
import json
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[2]
from monthly_diffusion import MonthlyDiffusionV1
from monthly_diffusion.forecast import Forecast
from monthly_diffusion.data.sources import open_canonical_state, open_co2
from monthly_diffusion.analysis.indices import enso_index
from monthly_diffusion.analysis.spectra import lag_autocorrelation


def design(dates):
    dates = pd.DatetimeIndex(dates)
    years = (
        dates.year.to_numpy() * 12 + dates.month.to_numpy() - (1966 * 12 + 1)
    ) / 12.0
    return np.column_stack((np.eye(12)[dates.month - 1], years))


def fit_readout(x, y, dates):
    """Training-only ridge fit; blocked-year CV (not nested performance CV)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = design(dates)
    years = pd.DatetimeIndex(dates).year.to_numpy()
    alphas = np.logspace(-2, 4, 13)
    errors = np.zeros(len(alphas))
    cv_predictions = np.empty((len(alphas), len(y)))
    cv_truth = np.empty(len(y))
    for fold in np.array_split(np.unique(years), 5):
        test = np.isin(years, fold)
        bx = np.linalg.lstsq(d[~test], x[~test], rcond=None)[0]
        by = np.linalg.lstsq(d[~test], y[~test], rcond=None)[0]
        xx, yy = x - d @ bx, y - d @ by
        cv_truth[test] = yy[test]
        u, s, vt = np.linalg.svd(xx[~test], full_matrices=False)
        for j, alpha in enumerate(alphas):
            coef = vt.T @ ((s / (s * s + alpha)) * (u.T @ yy[~test]))
            pred = xx[test] @ coef
            errors[j] += np.sum((pred - yy[test]) ** 2)
            cv_predictions[j, test] = pred
    j = int(errors.argmin())
    bx = np.linalg.lstsq(d, x, rcond=None)[0]
    by = np.linalg.lstsq(d, y, rcond=None)[0]
    xx, yy = x - d @ bx, y - d @ by
    u, s, vt = np.linalg.svd(xx, full_matrices=False)
    coef = vt.T @ ((s / (s * s + alphas[j])) * (u.T @ yy))
    return dict(
        coef=coef,
        latent_baseline=bx,
        physical_baseline=by,
        alpha=float(alphas[j]),
        cv_prediction=cv_predictions[j],
        cv_truth=cv_truth,
    )


def metrics(prediction, truth):
    return dict(
        correlation=float(np.corrcoef(prediction, truth)[0, 1]),
        rmse=float(np.sqrt(np.mean((prediction - truth) ** 2))),
        std=float(np.std(prediction)),
        reference_std=float(np.std(truth)),
    )


def readout(z, dates, probe):
    # xarray Forecast output is [time, member, node, channel]; no renormalizing.
    flat = np.asarray(z).reshape(len(dates), -1, len(probe["coef"]))
    return (flat - design(dates)[:, None, :] @ probe["latent_baseline"]) @ probe["coef"]


def compare_initialization(
    model, cache, probes, *, first, months, members, seed, co2, output
):
    """Run two histories with paired diffusion innovations; preserve raw latents."""
    destination = output / f"paired_{first}_{months}months.nc"
    if destination.exists():
        return xr.load_dataset(destination)
    first = np.datetime64(first, "M")
    dates = np.concatenate(
        [
            np.asarray(cache[s + "_time"]).astype("datetime64[M]")
            for s in ("validation", "training")
        ]
    )
    positions = [
        int(np.flatnonzero(dates == first - np.timedelta64(k, "M"))[0]) for k in (1, 2)
    ]
    assert dates[positions[0]] == dates[positions[1]] + np.timedelta64(1, "M")
    histories = {}
    for component in ("atmosphere", "ocean"):
        z = torch.cat([cache[s][component] for s in ("validation", "training")])
        histories[component] = (
            z[positions].unsqueeze(0).expand(members, -1, -1, -1).clone()
        )
    products = []
    for initial in ("observed", "gaussian"):
        f = Forecast(
            model,
            None,
            None,
            co2=co2,
            members=members,
            seed=seed,
            gaussian_start=str(first),
        )
        if initial == "observed":
            f.history = [histories[c].to(model.device) for c in ("atmosphere", "ocean")]
            # Cached training and validation histories have the SAME fixed
            # normalizer. Forecast expects [current, previous], not chronology.
            for i, c in enumerate(histories):
                torch.testing.assert_close(
                    f.history[i].cpu(), histories[c], rtol=0, atol=0
                )
        start_history = [h.cpu().numpy() for h in f.history]
        latents = {c: [] for c in histories}
        valid_times = []
        tick = time.monotonic()
        for step in range(months):
            a, o = next(f)
            assert a.time.values[0].astype("datetime64[M]") == first + np.timedelta64(
                step, "M"
            )
            valid_times.append(a.time.values[0])
            for c, z in (("atmosphere", a), ("ocean", o)):
                latents[c].append(z.values[0])
            if (step + 1) % 120 == 0:
                print(
                    f"{first} {initial}: {step + 1}/{months} months, {time.monotonic() - tick:.1f}s",
                    flush=True,
                )
        fields = {}
        for i, c in enumerate(histories):
            z = np.stack(latents[c])
            fields[f"latent_{c}"] = (("time", "member", "node", "channel"), z)
            fields[f"enso_{c}"] = (
                ("time", "member"),
                readout(z, valid_times, probes[c]),
            )
            fields[f"initial_{c}"] = (
                ("member", "history", "node", "channel"),
                start_history[i],
            )
        products.append(
            xr.Dataset(
                fields,
                coords=dict(
                    time=valid_times,
                    member=range(members),
                    history=["current", "previous"],
                ),
            ).expand_dims(initialization=[initial])
        )
    ds = xr.concat(products, "initialization")
    ds.attrs.update(
        seed=seed,
        first_forecast_month=str(first),
        normalized_latents="fixed training normalizers",
        initial_noise_history="None for both components in both experiments",
        history_order="current, previous",
        checkpoint_sha256=model.weight_digest(),
    )
    ds.to_netcdf(
        destination, encoding={v: dict(zlib=True, complevel=2) for v in ds.data_vars}
    )
    return ds


def render(output, long, short, series, probes, source_index):
    """Report amplitude separately from forecast skill and ensemble spread."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    colors = dict(observed="#0072B2", gaussian="#D55E00")
    labels = dict(
        observed="Observed initialization", gaussian="Gaussian initialization"
    )
    dates = long.time.values
    truth = (
        source_index.sel(time=dates).values
        - design(dates) @ probes["ocean"]["physical_baseline"]
    )
    memory = []
    for lead in (1, 3, 6, 12, 24, 60, 120, 240, 492):
        row = dict(lead_month=lead)
        for component in ("atmosphere", "ocean"):
            pair = long[f"enso_{component}"].isel(time=lead - 1)
            row[f"paired_{component}_enso_rms_difference"] = float(
                np.sqrt(
                    np.mean(
                        (
                            pair.sel(initialization="observed")
                            - pair.sel(initialization="gaussian")
                        )
                        ** 2
                    )
                )
            )
        memory.append(row)
    pd.DataFrame(memory).to_csv(output / "initialization_memory.csv", index=False)
    axes[0, 0].plot(
        series["validation_physical"].validation_time,
        series["validation_physical"],
        color="black",
        lw=1.2,
        label="ORAS5",
    )
    axes[0, 0].plot(
        series["validation_ocean"].validation_time,
        series["validation_ocean"],
        color=colors["observed"],
        lw=1,
        label="Encoded ocean readout",
    )
    axes[0, 0].set(
        title="Held-out readout check · 1958–1965", ylabel="Niño3.4 anomaly (°C)"
    )
    axes[0, 0].legend(frameon=False, fontsize=8)
    years = pd.DatetimeIndex(dates).year.to_numpy()
    summary = []
    postspin = years >= 1971
    post_design = design(dates[postspin])
    post_truth = (
        truth[postspin]
        - post_design @ np.linalg.lstsq(post_design, truth[postspin], rcond=None)[0]
    )
    post_variances, post_acfs, member_rows = {}, {}, []
    for initial in colors:
        values = long.enso_ocean.sel(initialization=initial).values
        d = design(dates)
        anomalies = values - d @ np.linalg.lstsq(d, values, rcond=None)[0]
        # Recenter/detrend only for long-run variability, NOT for forecast skill.
        parts = []
        for lo, hi in [
            (1966, 1970),
            (1971, 1980),
            (1981, 1990),
            (1991, 2000),
            (2001, 2006),
        ]:
            keep = (years >= lo) & (years <= hi)
            std = float(anomalies[keep].std())
            parts.append(std)
            summary.append(
                dict(
                    initialization=initial,
                    first_year=lo,
                    last_year=hi,
                    std=std,
                    reference_std=float(truth[keep].std()),
                )
            )
        axes[1, 0].plot(
            [1968, 1975.5, 1985.5, 1995.5, 2003.5],
            parts,
            "o-",
            color=colors[initial],
            label=labels[initial],
        )
        # A fresh Gaussian history has a substantial transient. Exclude FIVE
        # years from BOTH runs before judging long-run oscillatory dynamics.
        settled = (
            values[postspin]
            - post_design
            @ np.linalg.lstsq(post_design, values[postspin], rcond=None)[0]
        )
        post_variances[initial] = settled.var(0)
        all_acfs = lag_autocorrelation(
            xr.DataArray(settled.T, dims=("member", "time")), max_lag=60
        )
        post_acfs[initial] = all_acfs.sel(lag_months=24).values
        for member in range(settled.shape[1]):
            member_rows.append(
                dict(
                    initialization=initial,
                    member=member,
                    first_year=1971,
                    last_year=2006,
                    std=float(np.sqrt(post_variances[initial][member])),
                    acf24=float(post_acfs[initial][member]),
                )
            )
        acf = all_acfs.mean("member")
        axes[1, 1].plot(
            acf.lag_months / 12, acf, color=colors[initial], label=labels[initial]
        )
        summary.append(
            dict(
                initialization=initial,
                first_year=1966,
                last_year=2006,
                std=float(anomalies.std()),
                reference_std=float(truth.std()),
            )
        )
        summary.append(
            dict(
                initialization=initial,
                first_year=1971,
                last_year=2006,
                std=float(settled.std()),
                reference_std=float(post_truth.std()),
                acf24=float(acf.sel(lag_months=24)),
            )
        )
        axes[1, 2].plot(
            dates[:120],
            anomalies[:120, 0],
            color=colors[initial],
            label=labels[initial],
            lw=1,
        )
    acf = lag_autocorrelation(xr.DataArray(post_truth, dims="time"), max_lag=60)
    axes[1, 1].plot(acf.lag_months / 12, acf, color="black", label="ORAS5")
    axes[1, 0].axhline(
        truth.std(), color="black", ls="--", label="ORAS5 full-period SD"
    )
    axes[1, 0].set(
        title="Long-run amplitude after initialization",
        ylabel="Standard deviation (°C)",
        xlabel="Year",
    )
    axes[1, 1].set(
        title="1971–2006 ACF · first 5 years excluded",
        xlabel="Lag (years)",
        ylabel="ACF",
    )
    axes[1, 2].set(title="First 10 years · one member", ylabel="Niño3.4 readout (°C)")
    scores = []
    targets = np.stack([ds.observed_enso.values for ds in short])
    persistence = np.array(
        [
            float(
                source_index.sel(
                    time=ds.time.values[0] - np.timedelta64(1, "D"), method="pad"
                )
            )
            - (
                design(
                    [ds.time.values[0].astype("datetime64[M]") - np.timedelta64(1, "M")]
                )
                @ probes["ocean"]["physical_baseline"]
            ).item()
            for ds in short
        ]
    )
    for initial in colors:
        pred = np.stack(
            [ds.enso_ocean.sel(initialization=initial).values for ds in short]
        )
        for lead in range(24):
            scores.append(
                dict(
                    initialization=initial,
                    lead_month=lead + 1,
                    cases=len(short),
                    **metrics(pred[:, lead].mean(-1), targets[:, lead]),
                    member_rms_amplitude=float(np.sqrt(np.mean(pred[:, lead] ** 2))),
                    ensemble_spread=float(
                        np.sqrt(np.mean(pred[:, lead].var(-1, ddof=1)))
                    ),
                    persistence_rmse=float(
                        np.sqrt(np.mean((persistence - targets[:, lead]) ** 2))
                    ),
                )
            )
        group = pd.DataFrame(scores).query("initialization == @initial")
        axes[0, 1].plot(
            group.lead_month, group.rmse, color=colors[initial], label=labels[initial]
        )
        axes[0, 2].plot(
            group.lead_month,
            group.correlation,
            color=colors[initial],
            label=labels[initial],
        )
    axes[0, 1].plot(
        group.lead_month,
        group.persistence_rmse,
        ls="--",
        color="black",
        label="Persistence",
    )
    axes[0, 1].set(
        title="Validation forecasts · ensemble mean",
        xlabel="Lead (months)",
        ylabel="RMSE (°C)",
    )
    axes[0, 2].set(
        title="Validation forecasts · only 6 starts",
        xlabel="Lead (months)",
        ylabel="Anomaly correlation",
        ylim=(-1, 1),
    )
    for ax in axes.flat:
        ax.axhline(0, color=".7", lw=0.6)
        ax.grid(alpha=0.15)
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[1, 1].legend(frameon=False, fontsize=8)
    fig.suptitle("ERA5/ORAS5 · latent ENSO readout · initialization sensitivity")
    fig.savefig(output / "enso_initialization_comparison.png", dpi=220)
    fig.savefig(output / "enso_initialization_comparison.pdf")
    plt.close(fig)
    pd.DataFrame(summary).to_csv(output / "long_run_amplitude.csv", index=False)
    pd.DataFrame(scores).to_csv(output / "initialized_forecast_scores.csv", index=False)
    pd.DataFrame(member_rows).to_csv(
        output / "postspin_member_statistics.csv", index=False
    )
    # Whole member pairs, never independent months. This is conditional on a
    # SINGLE observed initialization; it does not sample initialization dates.
    n = long.sizes["member"]
    picks = np.random.default_rng(83).integers(0, n, size=(10000, n))
    ratios = np.sqrt(
        post_variances["observed"][picks].mean(1)
        / post_variances["gaussian"][picks].mean(1)
    )
    acf_difference = (post_acfs["observed"][picks] - post_acfs["gaussian"][picks]).mean(
        1
    )
    (output / "paired_member_bootstrap.json").write_text(
        json.dumps(
            dict(
                period=[1971, 2006],
                draws=10000,
                seed=83,
                resampling="whole paired members",
                std_ratio_observed_over_gaussian_quantiles025_50_975=np.quantile(
                    ratios, [0.025, 0.5, 0.975]
                ).tolist(),
                acf24_observed_minus_gaussian_quantiles025_50_975=np.quantile(
                    acf_difference, [0.025, 0.5, 0.975]
                ).tolist(),
                reference_std=float(post_truth.std()),
                reference_acf24=float(acf.sel(lag_months=24)),
                caveat="One observed starting history; not uncertainty across observed initialization dates.",
            ),
            indent=2,
        )
    )
    fig, axes = plt.subplots(
        2, 3, figsize=(14, 7), constrained_layout=True, sharey=True
    )
    for ax, ds in zip(axes.flat, short):
        pred = ds.enso_ocean.sel(initialization="observed")
        for k in pred.member:
            ax.plot(
                ds.time,
                pred.sel(member=k),
                color=colors["observed"],
                alpha=0.15,
                lw=0.6,
            )
        ax.plot(
            ds.time,
            pred.mean("member"),
            color=colors["observed"],
            lw=1.7,
            label="Forecast mean",
        )
        ax.plot(ds.time, ds.observed_enso, color="black", lw=1.5, label="ORAS5")
        ax.axhline(0, color=".6", lw=0.7)
        ax.grid(alpha=0.15)
        ax.set_title(f"Initialized October {pd.Timestamp(ds.time.values[0]).year}")
        ax.tick_params(axis="x", labelrotation=25)
    axes[0, 0].legend(frameon=False)
    axes[0, 0].set_ylabel("Niño3.4 readout (°C)")
    axes[1, 0].set_ylabel("Niño3.4 readout (°C)")
    fig.suptitle(
        "Observed initialization · 24-month latent ENSO forecasts · validation years"
    )
    fig.savefig(output / "validation_enso_plumes.png", dpi=220)
    fig.savefig(output / "validation_enso_plumes.pdf")
    plt.close(fig)
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
