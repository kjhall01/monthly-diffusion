"""Compare finite cycling treatments with real-initialized rollout controls.

Read annual latent shards in bounded memory. No readout is refitted. Confidence
intervals resample whole members, never individual months or grid cells.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import gaussian_kde
import torch
import xarray as xr

from monthly_diffusion.experiments.experiment_isotropic_cycling_sweep import (
    OUTPUT,
    CHECKPOINT,
    AUDIT,
    COMPONENTS,
    MODES,
)
from monthly_diffusion.experiments.diagnose_cycling_modes import fixed_month_anomalies

STRONG = (
    ("ocean", "ENSO"),
    ("ocean", "PDO"),
    ("atmosphere", "ENSO"),
    ("atmosphere", "NAO"),
    ("atmosphere", "QBO"),
)
BOUNDS = dict(
    mean=(-0.15, 0.15),
    sd_ratio=(0.85, 1.15),
    acf_rmse=(0, 0.10),
    spectral_cdf_distance=(0, 0.10),
    wasserstein=(0, 0.15),
    monthly_mean_rmse=(0, 0.15),
    enso_coupling_difference=(-0.10, 0.10),
)


def assemble(folder):
    cache = torch.load(
        CHECKPOINT.parent / "latents.pt", map_location="cpu", weights_only=False
    )
    refs, coefs = {}, {}
    for c in COMPONENTS:
        coefs[c] = np.stack(
            [np.load(AUDIT / f"probe_{c}_{m}.npz")["coef"] for m in MODES], -1
        )
        tr = cache["training"][c].numpy().astype(float)
        refs[c] = tr.reshape(len(tr), -1) @ coefs[c]
    dates, projected, means, spreads = [], {c: [] for c in COMPONENTS}, [], []
    paths = sorted(folder.glob("year-*.pt"))
    if not paths:
        raise FileNotFoundError("No completed years")
    for y, path in enumerate(paths, 1):
        if path.name != f"year-{y:03d}.pt":
            raise ValueError("Missing annual block")
        block = torch.load(path, map_location="cpu", weights_only=False)
        dates.append(block["time"])
        channel_mean, channel_sd = [], []
        for c in COMPONENTS:
            z = block["predicted"][c].astype(float)
            projected[c].append(z.reshape(*z.shape[:3], -1) @ coefs[c])
            channel_mean.append(z.mean(-2))
            channel_sd.append(z.std(-2))
        means.append(np.stack(channel_mean, axis=3))
        spreads.append(np.stack(channel_sd, axis=3))
    dates = np.concatenate(dates)
    ref_dates = np.asarray(cache["training_time"])
    data = {}
    for c in COMPONENTS:
        x, ref = fixed_month_anomalies(
            np.concatenate(projected[c]), dates, refs[c], ref_dates
        )
        data[c] = (("time", "method", "member", "mode"), x)
        data[c + "_reference"] = (("training_time", "mode"), ref)
    data["channel_mean"] = (
        ("time", "method", "member", "component", "channel"),
        np.concatenate(means),
    )
    data["channel_spatial_sd"] = (
        ("time", "method", "member", "component", "channel"),
        np.concatenate(spreads),
    )
    ds = xr.Dataset(
        data,
        coords=dict(
            time=dates,
            method=block["methods"],
            member=np.arange(z.shape[2]),
            mode=list(MODES),
            training_time=ref_dates,
            component=list(COMPONENTS),
            channel=np.arange(z.shape[-1]),
        ),
    )
    ds.attrs.update(
        anomalies="Fixed training-month baseline; no detrending or ensemble mean removal",
        comparison="Real-data-initialized rollout, not observed climate",
        source=str(folder),
    )
    ds.to_netcdf(
        folder / "mode_trajectories.nc",
        encoding={v: dict(zlib=True, complevel=3) for v in ds},
    )
    return ds


def features(values, dates, grid):
    """Per-member summaries, preserving levels and separating temporal spread."""
    x = np.asarray(values, float).T  # member, time
    centered = x - x.mean(1, keepdims=True)
    denominator = np.sum(centered**2, axis=1)
    acf = np.stack(
        [
            np.sum(centered[:, :-lag] * centered[:, lag:], axis=1) / denominator
            for lag in range(1, 61)
        ],
        axis=1,
    )
    frequency, psd = signal.welch(
        x, fs=1, nperseg=min(240, x.shape[1]), axis=1, detrend="constant"
    )
    psd = psd[:, 1:]
    spectral_cdf = np.cumsum(psd / psd.sum(1, keepdims=True), axis=1)
    cdf = np.stack(
        [np.searchsorted(np.sort(v), grid, side="right") / len(v) for v in x]
    )
    if np.asarray(dates).ndim == 1:
        months = pd.DatetimeIndex(dates).month.to_numpy()
        monthly = np.stack([x[:, months == m].mean(1) for m in range(1, 13)], axis=1)
    else:
        # Pooled January/July cohorts have the same simulation age but different
        # calendar months. Never mislabel July's seasonal phase as January.
        months = np.asarray(dates)
        if months.shape != values.shape:
            raise ValueError("Per-member calendar does not match trajectories")
        monthly = np.array(
            [
                [x[i, months[:, i] == m].mean() for m in range(1, 13)]
                for i in range(x.shape[0])
            ]
        )
    return dict(
        mean=x.mean(1),
        sd=x.std(1),
        acf=acf,
        spectral_cdf=spectral_cdf,
        cdf=cdf,
        monthly=monthly,
    )


def differences(a, b, wa, wb, dx):
    """Rows of weights can be point estimates or whole-member bootstrap draws."""
    scale = wb @ b["sd"]
    return dict(
        mean=(wa @ a["mean"] - wb @ b["mean"]) / scale,
        sd_ratio=(wa @ a["sd"]) / scale,
        acf_rmse=np.sqrt(np.mean((wa @ a["acf"] - wb @ b["acf"]) ** 2, axis=1)),
        spectral_cdf_distance=np.max(
            np.abs(wa @ a["spectral_cdf"] - wb @ b["spectral_cdf"]), axis=1
        ),
        wasserstein=np.sum(np.abs(wa @ a["cdf"] - wb @ b["cdf"]), axis=1) * dx / scale,
        monthly_mean_rmse=np.sqrt(
            np.mean((wa @ a["monthly"] - wb @ b["monthly"]) ** 2, axis=1)
        )
        / scale,
    )


def compare(
    ds, folder, windows=None, bootstrap_samples=2000, seed=220, metric_names=None
):
    """Prespecified practical-equivalence bounds; never call nonsignificance a pass."""
    years = len(ds.time) // 12
    if windows is None:
        windows = {
            "years11_30": (10, 30),
            "years21_60": (20, 60),
            "years41_60": (40, 60),
        }
    windows = {k: v for k, v in windows.items() if v[1] <= years}
    n = ds.sizes["member"]
    rng = np.random.default_rng(seed)
    weight = np.full((1, n), 1 / n)
    boot = np.zeros((bootstrap_samples, n))
    independent_boot = np.zeros_like(boot)
    cohorts = ds.cohort.values if "cohort" in ds.coords else np.zeros(n, dtype=int)
    for cohort in np.unique(cohorts):
        indices = np.flatnonzero(cohorts == cohort)
        size = len(indices)
        boot[:, indices] = (
            rng.multinomial(size, np.ones(size) / size, size=bootstrap_samples) / n
        )
        independent_boot[:, indices] = (
            rng.multinomial(size, np.ones(size) / size, size=bootstrap_samples) / n
        )
    rows, member_rows = [], []
    sensors = STRONG if "physical" not in ds else tuple(("physical", m) for m in MODES)
    for period, (start, stop) in windows.items():
        window = ds.isel(time=slice(start * 12, stop * 12))
        dates = (
            window.calendar_month.values
            if "calendar_month" in window
            else window.time.values
        )
        for c, m in sensors:
            all_values = window[c].sel(mode=m).values
            lo, hi = float(all_values.min()), float(all_values.max())
            grid = np.linspace(lo - 0.01 * (hi - lo), hi + 0.01 * (hi - lo), 2049)
            ref = window[c].sel({"mode": m, "method": "observed"}).values
            reference = features(ref, dates, grid)
            for method in ds.method.values:
                x = window[c].sel({"mode": m, "method": method}).values
                ft = features(x, dates, grid)
                for member in range(n):
                    member_rows.append(
                        dict(
                            period=period,
                            component=c,
                            mode=m,
                            method=method,
                            member=member,
                            mean=ft["mean"][member],
                            sd=ft["sd"][member],
                            acf24=ft["acf"][member, 23],
                        )
                    )
                if method == "observed":
                    continue
                point = differences(ft, reference, weight, weight, grid[1] - grid[0])
                samples = differences(
                    ft,
                    reference,
                    boot,
                    independent_boot if method == "observed_independent" else boot,
                    grid[1] - grid[0],
                )
                for metric, values in samples.items():
                    low, high = np.quantile(values, [0.05, 0.95])
                    lower, upper = BOUNDS[metric]
                    estimate = float(point[metric][0])
                    rows.append(
                        dict(
                            period=period,
                            component=c,
                            mode=m,
                            method=method,
                            metric=metric,
                            estimate=estimate,
                            low90=float(low),
                            high90=float(high),
                            lower_bound=lower,
                            upper_bound=upper,
                            point_pass=lower <= estimate <= upper,
                            interval_pass=lower <= low and high <= upper,
                        )
                    )
        # Cross-component ENSO synchrony is a separate check, not another EOF.
        for method in ds.method.values:
            if method == "observed" or "atmosphere" not in window:
                continue
            correlations = []
            for name in (method, "observed"):
                a = window.atmosphere.sel({"mode": "ENSO", "method": name}).values
                o = window.ocean.sel({"mode": "ENSO", "method": name}).values
                correlations.append(
                    np.array([np.corrcoef(a[:, i], o[:, i])[0, 1] for i in range(n)])
                )
            bs = (
                boot @ correlations[0]
                - (independent_boot if method == "observed_independent" else boot)
                @ correlations[1]
            )
            low, high = np.quantile(bs, [0.05, 0.95])
            estimate = float((correlations[0] - correlations[1]).mean())
            rows.append(
                dict(
                    period=period,
                    component="coupling",
                    mode="ENSO",
                    method=method,
                    metric="enso_coupling_difference",
                    estimate=estimate,
                    low90=float(low),
                    high90=float(high),
                    lower_bound=-0.1,
                    upper_bound=0.1,
                    point_pass=abs(estimate) <= 0.1,
                    interval_pass=low >= -0.1 and high <= 0.1,
                )
            )
    table = pd.DataFrame(rows)
    if metric_names is not None:
        # Anomaly-only comparisons cannot test mean-state agreement: centering
        # makes those checks pass by construction. Keep only requested metrics.
        unknown = set(metric_names) - set(BOUNDS)
        if unknown:
            raise ValueError(f"Unknown agreement metrics: {unknown}")
        table = table[table.metric.isin(metric_names)].copy()
    table.to_csv(folder / "agreement_metrics.csv", index=False)
    pd.DataFrame(member_rows).to_csv(folder / "member_metrics.csv", index=False)
    if len(table):
        normalized = []
        for row in table.itertuples():
            target = 1 if row.metric == "sd_ratio" else 0
            normalized.append(abs(row.estimate - target) / (row.upper_bound - target))
        table["fraction_of_tolerance"] = normalized
        ranking = (
            table.groupby(["period", "method"])
            .agg(
                tests=("point_pass", "size"),
                point_passes=("point_pass", "sum"),
                interval_passes=("interval_pass", "sum"),
                worst_tolerance_fraction=("fraction_of_tolerance", "max"),
                mean_tolerance_fraction=("fraction_of_tolerance", "mean"),
            )
            .reset_index()
        )
        ranking.to_csv(folder / "ranking.csv", index=False)
        print(ranking.to_string(index=False), flush=True)
    return table


def plot(ds, folder, methods=None, window=(20, 60)):
    if len(ds.time) < window[1] * 12:
        return
    if methods is None:
        methods = [
            m
            for m in (
                "cycle0",
                "cycle1",
                "cycle12",
                "cycle30",
                "cycle60",
                "cycle120",
                "observed",
                "observed_independent",
            )
            if m in ds.method
        ]
    styles = {
        m: dict(color=plt.get_cmap("viridis")(i / max(1, len(methods) - 3)), lw=1.25)
        for i, m in enumerate(methods)
    }
    styles["cycle12"] = dict(color="#0072b2", lw=1.35)
    styles["cycle60"] = dict(color="#d55e00", lw=1.35)
    styles["observed"] = dict(color="black", lw=2)
    styles["observed_independent"] = dict(color=".5", lw=1.7, ls="--")
    sensors = STRONG if "physical" not in ds else tuple(("physical", m) for m in MODES)
    fig, axes = plt.subplots(
        len(sensors), 3, figsize=(14, 2.5 * len(sensors)), layout="constrained"
    )
    w = ds.isel(time=slice(window[0] * 12, window[1] * 12))
    for row, (c, mode) in enumerate(sensors):
        ref = w[c].sel({"method": "observed", "mode": mode}).values
        scale = ref.std(axis=0).mean()
        bound = max(
            4, float(np.quantile(np.abs(w[c].sel(mode=mode)) / scale, 0.998)) * 1.1
        )
        grid = np.linspace(-bound, bound, 2049)
        for method in methods:
            x = w[c].sel({"method": method, "mode": mode}).values / scale
            f = features(x, w.time.values, grid)
            label = (
                method.replace("cycle", "Cycle ")
                .replace("observed_independent", "Real ICs (independent)")
                .replace("observed", "Real ICs")
            )
            axes[row, 0].plot(
                grid, gaussian_kde(x.ravel())(grid), label=label, **styles[method]
            )
            axes[row, 1].plot(np.arange(1, 61) / 12, f["acf"].mean(0), **styles[method])
            fr, p = signal.welch(
                x.T, fs=1, nperseg=min(240, len(x)), axis=1, detrend="constant"
            )
            axes[row, 2].loglog(1 / fr[1:] / 12, p[:, 1:].mean(0), **styles[method])
        axes[row, 0].set(
            title=f"{c.title()} · {mode}", xlabel="Real-rollout SD", ylabel="Density"
        )
        axes[row, 1].set(xlabel="Lag (years)", ylabel="ACF", ylim=(-0.8, 1))
        axes[row, 1].axhline(0, color=".7", lw=0.6)
        axes[row, 2].set(xlabel="Period (years)", ylabel="PSD", xlim=(1 / 6, 20))
        for ax in axes[row]:
            ax.grid(alpha=0.15)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    description = ds.attrs.get(
        "anomalies",
        "Fixed training-month baseline; no detrending or ensemble mean removal",
    )
    fig.suptitle(
        f"Initialization cycling · simulation years {window[0] + 1}–{window[1]}\n{description}"
    )
    for ext in ("png", "pdf"):
        fig.savefig(
            folder / f"dynamics_years{window[0] + 1}_{window[1]}.{ext}", dpi=180
        )
    plt.close(fig)


def convergence(ds, folder, methods=None):
    """Overlapping 20-year windows show adjustment, not independent evidence."""
    if methods is None:
        methods = [
            m
            for m in (
                "cycle0",
                "cycle1",
                "cycle12",
                "cycle30",
                "cycle60",
                "cycle120",
                "observed",
                "observed_independent",
            )
            if m in ds.method
        ]
    sensors = STRONG
    fig, axes = plt.subplots(len(sensors), 3, figsize=(14, 11), layout="constrained")
    rows = []
    for row, (c, mode) in enumerate(sensors):
        scale = float(ds[c + "_reference"].sel(mode=mode).std())
        for j, method in enumerate(methods):
            xs, means, sds, acfs = [], [], [], []
            for start in range(0, len(ds.time) // 12 - 19, 10):
                duration = int(method[5:]) if method.startswith("cycle") else 0
                if 12 * start < duration:
                    continue
                x = (
                    ds[c]
                    .sel({"method": method, "mode": mode})
                    .values[start * 12 : (start + 20) * 12]
                )
                f = features(
                    x,
                    ds.time.values[start * 12 : (start + 20) * 12],
                    np.linspace(x.min(), x.max(), 65),
                )
                xs.append(start + 10)
                means.append(f["mean"].mean() / scale)
                sds.append(f["sd"].mean() / scale)
                acfs.append(f["acf"][:, 23].mean())
                rows.append(
                    dict(
                        component=c,
                        mode=mode,
                        method=method,
                        start_year=start + 1,
                        end_year=start + 20,
                        mean_reference_sd=means[-1],
                        sd_ratio=sds[-1],
                        acf24=acfs[-1],
                    )
                )
            style = dict(
                color=plt.get_cmap("viridis")(j / max(1, len(methods) - 3)), lw=1.25
            )
            if method == "observed":
                style = dict(color="black", lw=2)
            if method == "observed_independent":
                style = dict(color=".5", ls="--", lw=1.7)
            for col, values in enumerate((means, sds, acfs)):
                axes[row, col].plot(xs, values, marker="o", ms=3, label=method, **style)
                axes[row, col].grid(alpha=0.15)
                axes[row, col].set_xlabel("20-year window midpoint (simulation year)")
            axes[row, 0].set_ylabel(f"{c.title()} {mode}")
        for col, title in enumerate(
            (
                "Mean / encoded-reference SD",
                "Within-member SD / reference SD",
                "24-month ACF",
            )
        ):
            axes[0, col].set_title(title)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Adjustment after cycling ends · overlapping windows, not independent replicates"
    )
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"convergence.{ext}", dpi=180)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(folder / "convergence.csv", index=False)


def bulk_latents(folder, windows):
    """Raw spatial mean/scales, so readouts cannot conceal other latent shifts."""
    rows = []
    for period, (start, stop) in windows.items():
        sums, squares = {}, {}
        count = 0
        for y in range(start + 1, stop + 1):
            block = torch.load(
                folder / f"year-{y:03d}.pt", map_location="cpu", weights_only=False
            )
            count += len(block["time"])
            for c in COMPONENTS:
                z = block["predicted"][c].astype(float)
                if c not in sums:
                    sums[c] = np.zeros_like(z[0])
                    squares[c] = np.zeros_like(z[0])
                sums[c] += z.sum(0)
                squares[c] += (z * z).sum(0)
        baseline = block["methods"].index("observed")
        for c in COMPONENTS:
            mean = sums[c] / count
            variance = np.maximum(squares[c] / count - mean**2, 0)
            reference = mean[baseline].mean(0)
            for i, method in enumerate(block["methods"]):
                for channel in range(mean.shape[-1]):
                    rows.append(
                        dict(
                            period=period,
                            component=c,
                            method=method,
                            channel=channel,
                            mean_map_rmse=float(
                                np.sqrt(
                                    np.mean(
                                        (
                                            mean[i, ..., channel].mean(0)
                                            - reference[:, channel]
                                        )
                                        ** 2
                                    )
                                )
                            ),
                            mean=float(mean[i, ..., channel].mean()),
                            temporal_rms_sd=float(
                                np.sqrt(variance[i, ..., channel].mean())
                            ),
                            reference_temporal_rms_sd=float(
                                np.sqrt(variance[baseline, ..., channel].mean())
                            ),
                        )
                    )
    table = pd.DataFrame(rows)
    table.to_csv(folder / "bulk_latents.csv", index=False)
    return table


def pool_confirmation(
    folders, output, windows=None, plot_window=(60, 100), cohort_names=None
):
    """Combine independent cohorts by simulation age, preserving calendars."""
    pieces = []
    offset = 0
    first = None
    if windows is None:
        windows = {
            "years21_60": (20, 60),
            "years61_100": (60, 100),
            "years81_100": (80, 100),
        }
    if cohort_names is None:
        cohort_names = [folder.name for folder in folders]
    if len(cohort_names) != len(folders) or len(set(cohort_names)) != len(folders):
        raise ValueError("Each independent cohort needs a distinct name")
    for folder, cohort in zip(folders, cohort_names):
        ds = xr.load_dataset(folder / "mode_trajectories.nc")
        if first is None:
            first = ds.time.values
        if len(ds.time) != len(first):
            raise ValueError("Confirmation horizons differ")
        months = ds.time.dt.month.values
        n = ds.sizes["member"]
        ds = ds.assign_coords(
            time=first,
            member=np.arange(offset, offset + n),
            cohort=("member", np.repeat(cohort, n)),
        )
        ds["calendar_month"] = (
            ("time", "member"),
            np.repeat(months[:, None], n, axis=1),
        )
        offset += n
        pieces.append(ds)
    combined = xr.concat(
        pieces, "member", data_vars="minimal", coords="minimal", compat="equals"
    )
    combined.attrs.update(
        source_folders=";".join(str(p) for p in folders),
        time_axis="Simulation age labeled by first cohort dates; calendar_month preserves actual member calendar",
        uncertainty="Whole-member bootstrap stratified by starting-calendar cohort",
    )
    output.mkdir(exist_ok=True)
    combined.to_netcdf(output / "mode_trajectories.nc")
    compare(combined, output, windows=windows)
    plot(combined, output, window=plot_window)
    return combined


def joint_and_seasonal(ds, folder, window):
    """Secondary checks, not criteria used to select cycling durations."""
    w = ds.isel(time=slice(window[0] * 12, window[1] * 12))
    names = [f"{c}_{m}" for c in COMPONENTS for m in MODES]
    rows, seasonal = [], []
    for method in ds.method.values:
        values = np.concatenate(
            [w[c].sel({"method": method}).values for c in COMPONENTS], axis=-1
        )
        for member in range(ds.sizes["member"]):
            corr = np.corrcoef(values[:, member], rowvar=False)
            for i, name in enumerate(names):
                for j, other in enumerate(names):
                    if j > i:
                        rows.append(
                            dict(
                                method=method,
                                member=member,
                                first=name,
                                second=other,
                                correlation=corr[i, j],
                            )
                        )
        calendar = (
            w.calendar_month.values
            if "calendar_month" in w
            else np.repeat(w.time.dt.month.values[:, None], ds.sizes["member"], axis=1)
        )
        for c, m in STRONG:
            x = w[c].sel({"method": method, "mode": m}).values
            for member in range(ds.sizes["member"]):
                for month in range(1, 13):
                    v = x[calendar[:, member] == month, member]
                    seasonal.append(
                        dict(
                            method=method,
                            member=member,
                            component=c,
                            mode=m,
                            month=month,
                            mean=float(v.mean()),
                            sd=float(v.std()),
                        )
                    )
    tag = f"years{window[0] + 1}_{window[1]}"
    pd.DataFrame(rows).to_csv(
        folder / f"cross_mode_correlations_{tag}.csv", index=False
    )
    s = pd.DataFrame(seasonal)
    s.to_csv(folder / f"seasonal_variance_{tag}.csv", index=False)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.2), layout="constrained")
    for ax, (c, m) in zip(axes, STRONG):
        table = s[(s.component == c) & (s["mode"] == m)]
        for method, g in table.groupby("method", sort=False):
            v = g.groupby("month").sd.mean()
            ax.plot(
                v.index,
                v.values,
                label=method,
                color="black"
                if method == "observed"
                else ".5"
                if method == "observed_independent"
                else None,
                ls="--" if method == "observed_independent" else "-",
                lw=2 if method == "observed" else 1.3,
            )
        ax.set(
            title=f"{c.title()} {m}",
            xlabel="Calendar month",
            ylabel="Index SD",
            xticks=[1, 4, 7, 10, 12],
        )
        ax.grid(alpha=0.15)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"seasonal_variance_{tag}.{ext}", dpi=180)
    plt.close(fig)


def paired_separation(folder):
    """Same-innovation path distance: useful context, not distribution equality."""
    rows = []
    for y, path in enumerate(sorted(folder.glob("year-*.pt")), 1):
        b = torch.load(path, map_location="cpu", weights_only=False)
        k = b["methods"].index("observed")
        for c in COMPONENTS:
            z = b["predicted"][c].astype(float)
            for i, method in enumerate(b["methods"]):
                if method == "observed":
                    continue
                rms = np.sqrt(np.mean((z[:, i] - z[:, k]) ** 2, axis=(0, 2, 3)))
                for member, value in enumerate(rms):
                    rows.append(
                        dict(
                            year=y,
                            component=c,
                            method=method,
                            member=member,
                            rmse=value,
                        )
                    )
    table = pd.DataFrame(rows)
    table.to_csv(folder / "paired_latent_separation.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, c in zip(axes, COMPONENTS):
        for method, g in table[table.component == c].groupby("method", sort=False):
            values = g.groupby("year").rmse.mean()
            ax.plot(
                values.index,
                values.values,
                label=method,
                ls="--" if method == "observed_independent" else "-",
                color="black" if method == "observed_independent" else None,
            )
        ax.set(
            title=c.title(),
            xlabel="Simulation year",
            ylabel="Paired full-latent RMS difference",
        )
        ax.grid(alpha=0.15)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Distance from real-initialized histories · dashed control has independent innovations"
    )
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"paired_latent_separation.{ext}", dpi=180)
    plt.close(fig)
    return table


def anomaly_sensitivity(ds, folder, window):
    """Descriptive ACF/spread sensitivity; primary results retain all offsets.

    Own-climatology removal and detrending are deliberately not evidence of
    matching mean states: by construction they erase that part of the test.
    Every member's actual calendar is used, including pooled seasonal cohorts.
    """
    w = ds.isel(time=slice(window[0] * 12, window[1] * 12))
    calendar = (
        w.calendar_month.values
        if "calendar_month" in w
        else np.repeat(w.time.dt.month.values[:, None], ds.sizes["member"], axis=1)
    )
    sensors = STRONG if "physical" not in ds else tuple(("physical", m) for m in MODES)
    rows = []
    for c, m in sensors:
        for method in ds.method.values:
            raw = w[c].sel({"method": method, "mode": m}).values
            own = raw.copy()
            detrended = raw.copy()
            for member in range(own.shape[1]):
                for month in range(1, 13):
                    selected = calendar[:, member] == month
                    own[selected, member] -= own[selected, member].mean()
                design = np.column_stack(
                    (np.eye(12)[calendar[:, member] - 1], np.linspace(-1, 1, len(raw)))
                )
                detrended[:, member] = (
                    raw[:, member]
                    - design @ np.linalg.lstsq(design, raw[:, member], rcond=None)[0]
                )
            for name, values in (
                ("fixed_training_month", raw),
                ("own_month", own),
                ("own_month_linear_detrend", detrended),
            ):
                ft = features(
                    values,
                    calendar,
                    np.linspace(float(values.min()), float(values.max()), 257),
                )
                for member in range(values.shape[1]):
                    rows.append(
                        dict(
                            component=c,
                            mode=m,
                            method=method,
                            member=member,
                            anomalization=name,
                            mean=ft["mean"][member],
                            sd=ft["sd"][member],
                            acf12=ft["acf"][member, 11],
                            acf24=ft["acf"][member, 23],
                            acf36=ft["acf"][member, 35],
                        )
                    )
    table = pd.DataFrame(rows)
    table.to_csv(
        folder / f"anomaly_sensitivity_years{window[0] + 1}_{window[1]}.csv",
        index=False,
    )
    return table
