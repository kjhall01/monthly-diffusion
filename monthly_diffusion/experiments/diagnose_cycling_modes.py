"""ENSO/PDO/NAO/QBO audit of the saved, paired initialization experiments.

No new forecasts: fit train-only, validation-checked latent ridge readouts,
then stream the saved *predicted* latents through the unchanged decoder.
All physical indices use package definitions and one training-only PDO EOF.
Primary anomalies subtract fixed training-month means, with no detrending
and no ensemble-mean subtraction. Compare each stochastic trajectory before
averaging statistics, not statistics of the ensemble-mean trajectory.
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
from scipy.stats import gaussian_kde
import torch
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[2]
from monthly_diffusion import MonthlyDiffusionV1
from monthly_diffusion.analysis.indices import (
    enso_index,
    nao_index,
    pdo_index,
    pdo_loading,
    qbo_index,
)
from monthly_diffusion.analysis.spectra import lag_autocorrelation, temporal_spectrum
from monthly_diffusion.data.sources import open_canonical_state
from monthly_diffusion.experiments.diagnose_enso_initialization import (
    fit_readout,
    design,
    metrics,
)
from monthly_diffusion.experiments.diagnose_latent_attractor import anomalies
from monthly_diffusion.experiments.experiment_cycle_duration import (
    DISPLAY,
    LABELS,
    COLORS,
)

MODES = ("ENSO", "PDO", "NAO", "QBO")
COMPONENTS = ("atmosphere", "ocean")
ARCHIVES = {
    "matched": ROOT / "analysis/local_moment_initialization_30years",
    "cycle1": ROOT / "analysis/cycling_duration_30years",
    "cycle12": ROOT / "analysis/cycling_duration_30years",
    "matched_cycle30": ROOT / "analysis/local_moment_initialization_30years",
    "observed": ROOT / "analysis/cycled_initialization_ensemble_30years",
}
UNITS = {"ENSO": "°C", "PDO": "°C × fixed EOF projection", "NAO": "hPa", "QBO": "m s−1"}


def fixed_month_anomalies(values, dates, reference, reference_dates):
    """Preserve forecast offsets/trends/spread; never fit on forecast values."""
    rm = pd.DatetimeIndex(reference_dates).month.to_numpy()
    fm = pd.DatetimeIndex(dates).month.to_numpy()
    climate = np.stack([reference[rm == m].mean(0) for m in range(1, 13)])
    shape = (len(dates),) + (1,) * (values.ndim - reference.ndim) + reference.shape[1:]
    return values - climate[fm - 1].reshape(shape), reference - climate[rm - 1]


def index_fields(atmos, ocean, loading):
    """Raw physical indices; seasonality removed only in the analysis step."""
    return xr.Dataset(
        dict(
            ENSO=enso_index(ocean, remove_monthly=False),
            PDO=pdo_index(ocean, loading=loading, remove_monthly=False),
            NAO=nao_index(atmos, remove_monthly=False) / 100.0,
            QBO=qbo_index(atmos, remove_monthly=False),
        )
    ).compute()


def prepare_readouts(model, cache, sources, output):
    """Ridge alpha via blocked-year training CV; independent 1958–65 check."""
    target_path = output / "source_indices.nc"
    if not target_path.exists():
        a, o, _ = open_canonical_state(
            sources["state_zarr"], expected_attrs=sources.get("expected_attrs")
        )
        a = a.sel(time=slice("1958", "2006"))
        o = model.preprocessor.mask_ocean(o).sel(time=slice("1958", "2006"))
        region = o[["sst"]].sel(lat=slice(20, 70), lon=slice(110, 260)).compute()
        train = region.sel(time=np.asarray(cache["training_time"]))
        x = train.sst.values.reshape(len(train.time), -1).astype(float)
        valid = np.isfinite(x).all(0)
        d = design(train.time)
        residual = np.full_like(x, np.nan)
        residual[:, valid] = (
            x[:, valid] - d @ np.linalg.lstsq(d, x[:, valid], rcond=None)[0]
        )
        anomaly = train.copy(data={"sst": residual.reshape(train.sst.shape)})
        loading = pdo_loading(anomaly, remove_monthly=False)
        loading.attrs = {
            k: int(v) if isinstance(v, (bool, np.bool_)) else v
            for k, v in loading.attrs.items()
        }
        loading.attrs.update(
            training_period="1966–2006",
            preprocessing="monthly intercepts + linear trend removed before EOF",
        )
        loading.to_netcdf(output / "pdo_loading.nc")
        source = xr.Dataset(
            dict(
                ENSO=enso_index(
                    o.sel(lat=slice(-5, 5), lon=slice(190, 240)), remove_monthly=False
                ),
                PDO=pdo_index(region, loading=loading, remove_monthly=False),
                NAO=nao_index(
                    a.sel(lat=slice(35, 70), lon=slice(330, 360)), remove_monthly=False
                )
                / 100.0,
                QBO=qbo_index(a.sel(lat=slice(-5, 5)), remove_monthly=False),
            )
        ).compute()
        for mode in MODES:
            source[mode].attrs["units"] = UNITS[mode]
        source.to_netcdf(target_path)
    source = xr.load_dataset(target_path)
    probes, rows = {}, []
    for c in COMPONENTS:
        for mode in MODES:
            path = output / f"probe_{c}_{mode}.npz"
            dates = np.asarray(cache["training_time"])
            if not path.exists():
                p = fit_readout(
                    cache["training"][c].numpy().reshape(len(dates), -1),
                    source[mode].sel(time=dates).values,
                    dates,
                )
                np.savez(path, **p)
            with np.load(path) as saved:
                p = dict(saved)
            probes[c, mode] = p
            for split in ("training", "validation"):
                dates = np.asarray(cache[split + "_time"])
                x = cache[split][c].numpy().reshape(len(dates), -1)
                prediction = (x - design(dates) @ p["latent_baseline"]) @ p["coef"]
                truth = (
                    source[mode].sel(time=dates).values
                    - design(dates) @ p["physical_baseline"]
                )
                rows.append(
                    dict(
                        component=c,
                        mode=mode,
                        split=split,
                        alpha=float(p["alpha"]),
                        **metrics(prediction, truth),
                    )
                )
            rows.append(
                dict(
                    component=c,
                    mode=mode,
                    split="blocked_cv_alpha_selection",
                    alpha=float(p["alpha"]),
                    **metrics(p["cv_prediction"], p["cv_truth"]),
                )
            )
    skill = pd.DataFrame(rows)
    skill.to_csv(output / "readout_skill.csv", index=False)
    print(skill[skill.split == "validation"].to_string(index=False), flush=True)
    return probes


def read_annual_latents(year):
    """Select each named arm, checking calendar and shape across archives."""
    blocks = {
        p: torch.load(p / f"year-{year:03d}.pt", map_location="cpu", weights_only=False)
        for p in set(ARCHIVES.values())
    }
    dates = next(iter(blocks.values()))["time"]
    expected = pd.date_range(f"{1799 + year}-01-01", periods=12, freq="MS").values
    np.testing.assert_array_equal(dates, expected)
    z = {c: [] for c in COMPONENTS}
    for method in DISPLAY:
        block = blocks[ARCHIVES[method]]
        np.testing.assert_array_equal(block["time"], dates)
        i = list(block["methods"]).index(method)
        for c in COMPONENTS:
            values = np.asarray(block["predicted"][c])[:, i]
            assert values.shape == (12, 5, 162, 4) and np.isfinite(values).all()
            z[c].append(values)
    return dates, {c: np.stack(v, axis=1) for c, v in z.items()}


@torch.inference_mode()
def decode_saved(model, cache, config, output):
    """Bounded-memory decode, annual restart shards, no stored physical maps."""
    loading = xr.load_dataarray(output / "pdo_loading.nc")
    destination = output / "decoded_reference_indices.nc"
    if not destination.exists():
        dates = np.asarray(cache["training_time"])
        pieces = []
        for start in range(0, len(dates), 4):
            z = [
                xr.DataArray(
                    cache["training"][c][start : start + 4].numpy(),
                    dims=("time", "node", "channel"),
                    coords={"time": dates[start : start + 4]},
                )
                for c in COMPONENTS
            ]
            a, o = model.decode(*z, co2=config["co2_ppm"])
            pieces.append(index_fields(a, o, loading))
            if (start + 4) % 120 == 0:
                print(f"Reference decoded {start + 4}/{len(dates)}", flush=True)
        ref = xr.concat(pieces, "time")
        ref.attrs.update(
            co2_ppm=config["co2_ppm"], weight_digest=config["weight_digest"]
        )
        ref.to_netcdf(destination)
    shards = output / "decoded_years"
    shards.mkdir(exist_ok=True)
    tick = time.monotonic()
    old_enso = xr.load_dataset(
        ROOT / "analysis/cycling_duration_30years/index_trajectories.nc"
    )
    old_reference = xr.load_dataset(
        ROOT
        / "analysis/cycled_noise_initialization_era5_30months/coupling/encoded_reference_indices.nc"
    )
    train_months = pd.DatetimeIndex(old_reference.time.values).month
    nino_climate = np.array(
        [old_reference.nino34.values[train_months == m].mean() for m in range(1, 13)]
    )
    for year in range(1, 31):
        path = shards / f"year-{year:03d}.nc"
        if path.exists():
            continue
        dates, latent = read_annual_latents(year)
        by_method = []
        for i, method in enumerate(DISPLAY):
            months = []
            for t, date in enumerate(dates):
                z = [
                    xr.DataArray(
                        latent[c][t : t + 1, i],
                        dims=("time", "member", "node", "channel"),
                        coords={"time": [date], "member": np.arange(5)},
                    )
                    for c in COMPONENTS
                ]
                a, o = model.decode(*z, co2=config["co2_ppm"])
                months.append(index_fields(a, o, loading))
            by_method.append(xr.concat(months, "time").expand_dims(method=[method]))
        ds = xr.concat(by_method, "method")
        ds["method"].encoding = {}
        # ENSO is an independent numerical replay check against saved physical indices.
        previous = old_enso.physical.sel(
            {"method": list(DISPLAY), "component": "ocean", "time": dates}
        )
        current = ds.ENSO - nino_climate[None, :, None]
        np.testing.assert_allclose(
            current.transpose(*previous.dims), previous, atol=5e-5, rtol=5e-5
        )
        ds.to_netcdf(path)
        print(
            f"Decoded five methods: year {year}/30 | elapsed {time.monotonic() - tick:.1f}s",
            flush=True,
        )


def scalar_statistics(x, reference):
    """Population SD within member; ACF uses full-record variance normalization."""
    x, reference = np.asarray(x), np.asarray(reference)
    if not np.isfinite(x).all() or not np.isfinite(reference).all():
        raise ValueError("Nonfinite index")
    da = xr.DataArray(x, dims="time")
    acf = lag_autocorrelation(da, max_lag=min(60, len(x) - 1), detrend="constant")
    spec = temporal_spectrum(da, nperseg=min(120, len(x)))
    f = spec.frequency.values
    positive = f > 0
    power = spec.values[positive]
    peak = float(1 / f[positive][np.argmax(power)])
    seasonal = np.array([x[m::12].mean() for m in range(12)])
    return dict(
        sd_ratio=float(x.std() / reference.std()),
        mean_reference_sd=float(x.mean() / reference.std()),
        sd=float(x.std()),
        reference_sd=float(reference.std()),
        spectral_peak_months=peak,
        seasonal_mean_sd_ratio=float(seasonal.std() / reference.std()),
        **{f"acf{lag}": float(acf.sel(lag_months=lag)) for lag in (1, 6, 12, 24, 36)},
    )


def assemble(cache, probes, output):
    ref_dates = np.asarray(cache["training_time"])
    raw = xr.concat(
        [
            xr.load_dataset(p)
            for p in sorted((output / "decoded_years").glob("year-*.nc"))
        ],
        "time",
    )
    assert raw.sizes["time"] == 360
    ref = xr.load_dataset(output / "decoded_reference_indices.nc")
    source = xr.load_dataset(output / "source_indices.nc").sel(time=ref_dates)
    dates = raw.time.values
    annual = [read_annual_latents(y)[1] for y in range(1, 31)]
    variables, reference, observed = [], [], []
    for mode in MODES:
        x, r = fixed_month_anomalies(
            raw[mode].transpose("time", "method", "member").values,
            dates,
            ref[mode].values,
            ref_dates,
        )
        _, truth = fixed_month_anomalies(
            source[mode].values, ref_dates, source[mode].values, ref_dates
        )
        variables.append(x)
        reference.append(r)
        observed.append(truth)
    data = dict(
        physical=(("time", "method", "member", "mode"), np.stack(variables, -1)),
        physical_reference=(("training_time", "mode"), np.stack(reference, -1)),
        source_reference=(("training_time", "mode"), np.stack(observed, -1)),
    )
    for c in COMPONENTS:
        z = np.concatenate([v[c] for v in annual]).astype(float)
        tr = cache["training"][c].numpy().astype(float)
        coef = np.stack([probes[c, m]["coef"] for m in MODES], -1)
        projected = z.reshape(*z.shape[:3], -1) @ coef
        projected_ref = tr.reshape(len(tr), -1) @ coef
        x, r = fixed_month_anomalies(projected, dates, projected_ref, ref_dates)
        data[c] = (("time", "method", "member", "mode"), x)
        data[c + "_reference"] = (("training_time", "mode"), r)
    ds = xr.Dataset(
        data,
        coords=dict(
            time=dates,
            training_time=ref_dates,
            method=list(DISPLAY),
            member=np.arange(1, 6),
            mode=list(MODES),
            mode_units=("mode", [UNITS[m] for m in MODES]),
        ),
    )
    ds.attrs.update(
        anomalies="Fixed training-month climatology removed; offsets retained; no trend or ensemble mean removal",
        physical_reference="Decode training encoded latents at the same fixed 1850 CO2; not preindustrial observations",
        latent_reference="Training-latent linear projections; ridge fits remove month/trend only during fitting",
        reference_period="1966–2006",
        primary_window="Simulation years 11–30",
    )
    ds.to_netcdf(
        output / "mode_trajectories.nc",
        encoding={v: dict(zlib=True, complevel=3) for v in ds},
    )
    return ds


def analyze(ds, output):
    rows, window_rows, coupling = [], [], []
    pd.DataFrame(
        [
            dict(
                mode=m,
                decoded_constant_co2_std=float(ds.physical_reference.sel(mode=m).std()),
                actual_training_std=float(ds.source_reference.sel(mode=m).std()),
                decoded_to_actual_std_ratio=float(
                    ds.physical_reference.sel(mode=m).std()
                    / ds.source_reference.sel(mode=m).std()
                ),
                correlation=float(
                    np.corrcoef(
                        ds.physical_reference.sel(mode=m),
                        ds.source_reference.sel(mode=m),
                    )[0, 1]
                ),
            )
            for m in MODES
        ]
    ).to_csv(output / "training_reference_comparison.csv", index=False)
    windows = {
        "years01_05": slice(0, 60),
        "years06_15": slice(60, 180),
        "years11_30": slice(120, 360),
    }
    for kind in ("physical", *COMPONENTS):
        for mode in MODES:
            ref = ds[kind + "_reference"].sel(mode=mode).values
            for start in range(0, len(ref) - 240 + 1, 12):
                window_rows.append(
                    dict(
                        kind=kind,
                        mode=mode,
                        first_year=1966 + start // 12,
                        **scalar_statistics(ref[start : start + 240], ref),
                    )
                )
            for period, sl in windows.items():
                for method in DISPLAY:
                    for member in ds.member.values:
                        x = (
                            ds[kind]
                            .sel({"mode": mode, "method": method, "member": member})
                            .values[sl]
                        )
                        for treatment in (
                            "fixed_training_month",
                            "own_month",
                            "own_month_detrended",
                        ):
                            xx, rr = x, ref
                            if treatment != "fixed_training_month":
                                treatment_name = (
                                    "window_monthly"
                                    if treatment == "own_month"
                                    else "window_monthly_detrend"
                                )
                                xx = anomalies(
                                    x[:, None], ds.time.values[sl], treatment_name
                                )[:, 0]
                                rr = anomalies(
                                    ref[:, None],
                                    ds.training_time.values,
                                    treatment_name,
                                )[:, 0]
                            rows.append(
                                dict(
                                    kind=kind,
                                    mode=mode,
                                    period=period,
                                    method=method,
                                    member=int(member),
                                    anomalies=treatment,
                                    **scalar_statistics(xx, rr),
                                )
                            )
    for mode in MODES:
        for method in ("reference", *DISPLAY):
            for member in [0] if method == "reference" else ds.member.values:
                if method == "reference":
                    a = ds.atmosphere_reference.sel(mode=mode).values
                    o = ds.ocean_reference.sel(mode=mode).values
                else:
                    a = ds.atmosphere.sel(
                        {"mode": mode, "method": method, "member": member}
                    ).values[120:]
                    o = ds.ocean.sel(
                        {"mode": mode, "method": method, "member": member}
                    ).values[120:]
                for lag in range(-24, 25):
                    # Positive lag: atmosphere leads ocean.
                    x, y = (
                        (a[:-lag], o[lag:])
                        if lag > 0
                        else ((a[-lag:], o[:lag]) if lag < 0 else (a, o))
                    )
                    coupling.append(
                        dict(
                            mode=mode,
                            method=method,
                            member=int(member),
                            lag_months=lag,
                            correlation=float(np.corrcoef(x, y)[0, 1]),
                        )
                    )
    table = pd.DataFrame(rows)
    table.to_csv(output / "member_metrics.csv", index=False)
    table.groupby(["kind", "mode", "period", "anomalies", "method"]).mean(
        numeric_only=True
    ).drop(columns="member").reset_index().to_csv(output / "summary.csv", index=False)
    pd.DataFrame(window_rows).to_csv(
        output / "reference_20year_windows.csv", index=False
    )
    pd.DataFrame(coupling).to_csv(
        output / "atmosphere_ocean_lag_correlations.csv", index=False
    )
    pd.DataFrame(
        [
            dict(
                kind=k,
                mode=m,
                **scalar_statistics(
                    ds[k + "_reference"].sel(mode=m).values,
                    ds[k + "_reference"].sel(mode=m).values,
                ),
            )
            for k in ("physical", *COMPONENTS)
            for m in MODES
        ]
    ).to_csv(output / "reference_metrics.csv", index=False)
    return table


def render(ds, table, output):
    skill = pd.read_csv(output / "readout_skill.csv")
    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    curve_records = []
    for kind in ("physical", *COMPONENTS):
        fig, axes = plt.subplots(4, 3, figsize=(14, 13), layout="constrained")
        for row, mode in enumerate(MODES):
            ref = ds[kind + "_reference"].sel(mode=mode).values
            scale = ref.std()
            curves = {"reference": ref[:, None] / scale}
            curves.update(
                {
                    m: ds[kind]
                    .sel({"mode": mode, "method": m})
                    .isel(time=slice(120, None))
                    .values
                    / scale
                    for m in DISPLAY
                }
            )
            if kind == "physical":
                curves["source"] = (
                    ds.source_reference.sel(mode=mode).values[:, None] / scale
                )
            pooled = np.concatenate([x.ravel() for x in curves.values()])
            lim = max(4.0, np.quantile(np.abs(pooled), 0.999) * 1.1)
            support = np.linspace(-lim, lim, 401)
            for name, x in curves.items():
                color = {"reference": "black", "source": ".55"}.get(
                    name, COLORS.get(name)
                )
                label = {
                    "reference": "Encoded training reference",
                    "source": "Actual training data",
                }.get(name, LABELS.get(name))
                ls = "--" if name == "source" else "-"
                axes[row, 0].plot(
                    support,
                    gaussian_kde(x.ravel())(support),
                    color=color,
                    ls=ls,
                    label=label,
                )
                da = xr.DataArray(x.T, dims=("member", "time"))
                acf = lag_autocorrelation(da, max_lag=60, detrend="constant")
                psd = temporal_spectrum(da, nperseg=120).isel(frequency=slice(1, None))
                axes[row, 1].plot(
                    np.arange(61) / 12, acf.mean("member"), color=color, ls=ls
                )
                axes[row, 2].loglog(
                    1 / psd.frequency / 12, psd.mean("member"), color=color, ls=ls
                )
                for member in range(len(x.T)):
                    curve_records.extend(
                        dict(
                            kind=kind,
                            mode=mode,
                            method=name,
                            member=member + 1,
                            statistic="acf",
                            coordinate=float(lag),
                            value=float(val),
                        )
                        for lag, val in zip(acf.lag_months, acf.isel(member=member))
                    )
                    curve_records.extend(
                        dict(
                            kind=kind,
                            mode=mode,
                            method=name,
                            member=member + 1,
                            statistic="psd",
                            coordinate=float(f),
                            value=float(val),
                        )
                        for f, val in zip(psd.frequency, psd.isel(member=member))
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
                title += f" · validation r={r:.2f}" + (
                    " · weak readout" if r < 0.5 else ""
                )
            axes[row, 0].set(
                title=title,
                xlabel="Encoded-reference SD; offsets retained",
                ylabel="Density",
            )
            axes[row, 1].set(xlabel="Lag (years)", ylabel="ACF", ylim=(-0.8, 1))
            axes[row, 1].axhline(0, color=".6", lw=0.6)
            axes[row, 2].set(
                xlabel="Period (years)",
                ylabel="PSD (reference SD² × month)",
                xlim=(1 / 6, 10),
            )
            for ax in axes[row]:
                ax.grid(alpha=0.16)
        fig.suptitle(
            f"{kind.title()} indices · cycling duration · simulation years 11–30\nFixed training-month anomalies; no detrending or ensemble-mean subtraction",
            fontsize=12,
        )
        fig.legend(
            *axes[0, 0].get_legend_handles_labels(),
            loc="outside lower center",
            ncol=4,
            frameon=False,
        )
        for ext in ("png", "pdf"):
            fig.savefig(output / f"{kind}_mode_dynamics.{ext}", dpi=180)
        plt.close(fig)
    pd.DataFrame(curve_records).to_csv(output / "spectra_and_acf.csv", index=False)
    # All physical trajectories, so a favorable ACF cannot conceal long offsets.
    fig, axes = plt.subplots(
        4, 5, figsize=(18, 10), sharex=True, sharey="row", layout="constrained"
    )
    for row, mode in enumerate(MODES):
        scale = float(ds.physical_reference.sel(mode=mode).std())
        for col, method in enumerate(DISPLAY):
            ax = axes[row, col]
            ax.plot(
                np.arange(360) / 12,
                ds.physical.sel({"mode": mode, "method": method}) / scale,
                lw=0.7,
                alpha=0.65,
            )
            ax.axvspan(0, 10, color=".8", alpha=0.2)
            ax.axhline(0, color="black", lw=0.6)
            ax.set_ylim(auto=True)
            ax.grid(alpha=0.1)
            if row == 0:
                ax.set_title(LABELS[method])
            if col == 0:
                ax.set_ylabel(f"{mode}\nReference SD")
            if row == 3:
                ax.set_xlabel("Simulation years")
    fig.suptitle(
        "Decoded indices · five individual trajectories · gray = excluded from late-period comparison"
    )
    for ext in ("png", "pdf"):
        fig.savefig(output / f"physical_all_trajectories.{ext}", dpi=170)
    plt.close(fig)


def render_seasonal_sensitivity(ds, output):
    """Separate a persistent seasonal mismatch from nonseasonal variability.

    Unlike the extra detrended table, these panels remove ONLY each member's
    own 12 monthly means. Slow nonseasonal drift is deliberately retained.
    """
    fig, axes = plt.subplots(4, 3, figsize=(14, 12), layout="constrained")
    dates = ds.time.values[120:]
    months = pd.DatetimeIndex(dates).month.to_numpy()
    rows = []
    for row, mode in enumerate(MODES):
        ref = ds.physical_reference.sel(mode=mode).values
        scale = ref.std()
        curves = {"reference": ref[:, None] / scale}
        curves.update(
            {
                m: ds.physical.sel({"mode": mode, "method": m}).values[120:] / scale
                for m in DISPLAY
            }
        )
        for method, x in curves.items():
            color = "black" if method == "reference" else COLORS[method]
            label = "Encoded reference" if method == "reference" else LABELS[method]
            if method == "reference":
                centered = x
            else:
                monthly = np.stack([x[months == m].mean(0) for m in range(1, 13)])
                # Remove each member's annual offset only for this seasonal-shape panel.
                # Original offsets remain in the primary PDFs, trajectories and CSVs.
                seasonal = monthly - monthly.mean(0)
                axes[row, 0].plot(
                    range(1, 13), seasonal.mean(1), color=color, label=label
                )
                axes[row, 0].fill_between(
                    range(1, 13),
                    seasonal.min(1),
                    seasonal.max(1),
                    color=color,
                    alpha=0.07,
                )
                rows.extend(
                    dict(
                        mode=mode,
                        method=method,
                        member=j + 1,
                        month=m + 1,
                        monthly_bias_reference_sd=float(monthly[m, j]),
                        seasonal_shape_reference_sd=float(seasonal[m, j]),
                    )
                    for m in range(12)
                    for j in range(5)
                )
                centered = anomalies(x, dates, "window_monthly")
            da = xr.DataArray(centered.T, dims=("member", "time"))
            acf = lag_autocorrelation(da, max_lag=60, detrend="constant").mean("member")
            psd = (
                temporal_spectrum(da, nperseg=120)
                .mean("member")
                .isel(frequency=slice(1, None))
            )
            axes[row, 1].plot(np.arange(61) / 12, acf, color=color, label=label)
            axes[row, 2].loglog(1 / psd.frequency / 12, psd, color=color)
        axes[row, 0].axhline(0, color="black", lw=0.8)
        axes[row, 0].set(
            title=mode,
            xlabel="Calendar month",
            ylabel="Seasonal mismatch / reference SD",
            xticks=[1, 4, 7, 10, 12],
        )
        axes[row, 1].set(
            xlabel="Lag (years)",
            ylabel="ACF · own monthly means removed",
            ylim=(-0.8, 1),
        )
        axes[row, 1].axhline(0, color=".5", lw=0.6)
        axes[row, 2].set(
            xlabel="Period (years)",
            ylabel="PSD · own monthly means removed",
            xlim=(1 / 6, 10),
        )
        for ax in axes[row]:
            ax.grid(alpha=0.15)
    fig.suptitle(
        "Decoded modes · seasonal-climatology sensitivity · years 11–30\nNo linear detrending or ensemble-mean subtraction; shading shows member range"
    )
    fig.legend(
        *axes[0, 1].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    for ext in ("png", "pdf"):
        fig.savefig(output / f"physical_seasonal_sensitivity.{ext}", dpi=180)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(output / "seasonal_mismatch.csv", index=False)


def render_summary(ds, table, output):
    skill = pd.read_csv(output / "readout_skill.csv")
    late = table[
        (table.kind == "physical")
        & (table.period == "years11_30")
        & (table.anomalies == "fixed_training_month")
    ]
    ref_windows = pd.read_csv(output / "reference_20year_windows.csv")
    fig, axes = plt.subplots(4, 3, figsize=(13, 11), layout="constrained")
    for row, mode in enumerate(MODES):
        for col, metric in enumerate(("sd_ratio", "mean_reference_sd", "acf24")):
            ax = axes[row, col]
            r = ds.physical_reference.sel(mode=mode).values
            ax.axhline(scalar_statistics(r, r)[metric], color="black", lw=1)
            w = ref_windows[
                (ref_windows.kind == "physical") & (ref_windows["mode"] == mode)
            ][metric]
            ax.axhspan(w.min(), w.max(), color=".5", alpha=0.12)
            for i, method in enumerate(DISPLAY):
                v = late[(late["mode"] == mode) & (late.method == method)][metric]
                ax.scatter(
                    i + np.linspace(-0.1, 0.1, len(v)), v, s=18, color=COLORS[method]
                )
                ax.plot(
                    [i - 0.2, i + 0.2], [v.mean()] * 2, color=COLORS[method], lw=2.5
                )
            ax.set_xticks(range(5), ["0", "1", "12", "30", "Observed"])
            ax.set(
                ylabel=f"{mode} · "
                + {
                    "sd_ratio": "SD / reference",
                    "mean_reference_sd": "Mean / reference SD",
                    "acf24": "24-month ACF",
                }[metric],
                xlabel="Cycling months / initialization",
            )
            ax.grid(axis="y", alpha=0.15)
    fig.suptitle(
        "Decoded mode summary · years 11–30\nDots: individual members; bars: averages; gray: range of overlapping 20-year reference windows"
    )
    for ext in ("png", "pdf"):
        fig.savefig(output / f"mode_summary.{ext}", dpi=180)
    plt.close(fig)
    coupling = pd.read_csv(output / "atmosphere_ocean_lag_correlations.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, mode in zip(axes.flat, MODES):
        for method in ("reference", *DISPLAY):
            curve = (
                coupling[(coupling["mode"] == mode) & (coupling.method == method)]
                .groupby("lag_months")
                .correlation.mean()
            )
            ax.plot(
                curve.index,
                curve,
                color="black" if method == "reference" else COLORS[method],
                label="Encoded reference" if method == "reference" else LABELS[method],
            )
        rs = (
            skill[(skill["mode"] == mode) & (skill.split == "validation")]
            .set_index("component")
            .correlation
        )
        ax.set(
            title=f"{mode} · readout validation r: atm {rs.atmosphere:.2f}, ocn {rs.ocean:.2f}",
            xlabel="Lag (months); positive = atmosphere leads",
            ylabel="Latent readout correlation",
            ylim=(-1, 1),
        )
        ax.axhline(0, color=".5", lw=0.6)
        ax.grid(alpha=0.15)
    fig.suptitle(
        "Atmosphere–ocean mode relationships · years 11–30\nWeak readouts are exploratory; correlation is not causal coupling"
    )
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    for ext in ("png", "pdf"):
        fig.savefig(output / f"latent_mode_relationships.{ext}", dpi=180)
    plt.close(fig)
