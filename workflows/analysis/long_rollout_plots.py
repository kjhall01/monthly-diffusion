"""Plots from small retained products, never from a full physical rollout."""
from pathlib import Path
import calendar
import logging
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import gaussian_kde
from workflows.analysis.spectra import lag_autocorrelation, temporal_spectrum
from workflows.analysis.climate import annual_means
from workflows.analysis.plotting import plot_map_comparison, plot_response_map
from workflows.analysis.long_rollout import save_dataset
from monthly_diffusion.preprocessing import mdv1_variables

log = logging.getLogger(__name__)
COLORS = {"reference": "#262626", "model": "#2378b5"}
LABELS = {"reference": "Held-out E3SMv2", "model": "Monthly Diffusion v1.0"}


def display(variable, field):
    """Presentation units only; retained calculations remain in native units."""
    factor, units = {"mtpr": (86400., "mm day⁻¹"), "mslp": (.01, "hPa"),
        "ohc300": (1.e-9, "GJ m⁻²"), "ohc700_minus_ohc300": (1.e-9, "GJ m⁻²")}.get(
        variable, (1., field.attrs.get("units", "")))
    return field*factor, units


def finish(fig, path):
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_indices(root, config):
    import matplotlib.pyplot as plt
    datasets = {name: xr.load_dataset(root/name/"indices.nc") for name in COLORS}
    rows = []
    labels = dict(LABELS, reference=config.get("reference_label", LABELS["reference"]))
    for mode in datasets["reference"]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 7.8), constrained_layout=True)
        unit = "hPa" if mode == "NAO" else "m s⁻¹" if mode == "QBO" else "°C"
        factor = .01 if mode == "NAO" else 1.
        series = {name: ds[mode]*factor for name, ds in datasets.items()}
        bounds = np.nanpercentile(np.concatenate([x.values.ravel() for x in series.values()]), [.1, 99.9])
        density_x = np.linspace(*bounds, 400)
        for name, value in series.items():
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite {name} {mode} index; cannot silently interpolate")
            color, label = COLORS[name], labels[name]
            nperseg = min(config.get("spectrum_segment_months", 600), value.sizes["time"])
            psd = temporal_spectrum(value, nperseg=nperseg)
            acf = lag_autocorrelation(value, max_lag=min(120, value.sizes["time"]-1), detrend="linear")
            phase = value.groupby("time.month").std(("member", "time"), ddof=0)
            summary = xr.Dataset(dict(psd=psd, acf=acf, seasonal_std=phase))
            summary.attrs["display_units"] = unit
            save_dataset(summary, root/"indices"/f"{name}_{mode.lower()}_statistics.nc")
            period = 1/(12*psd.frequency.values[1:])
            axes[0, 0].plot(period, psd.mean("member").values[1:], color=color, label=label)
            axes[0, 0].fill_between(period, psd.min("member").values[1:], psd.max("member").values[1:], color=color, alpha=.12)
            axes[0, 1].plot(acf.lag_months/12, acf.mean("member"), color=color)
            axes[0, 1].fill_between(acf.lag_months/12, acf.min("member"), acf.max("member"), color=color, alpha=.12)
            if float(value.std()) > 0:
                axes[0, 2].plot(density_x, gaussian_kde(value.values.ravel())(density_x), color=color)
            axes[1, 0].plot(phase.month, phase, marker="o", color=color)
            year = config.get("sample_start_year", 1980)
            sample = value.sel(time=slice(str(year), str(year+9))).isel(member=0)
            sample_years = sample.time.dt.year+(sample.time.dt.month-.5)/12
            axes[1, 1].plot(sample_years, sample, color=color, lw=1.)
            periods = ({"Training": config["training"]} if "index_period" in config else
                       {"Early": config["early"], "Training": config["training"], "Late": config["late"]})
            amplitudes = []
            for key, (lo, hi) in periods.items():
                part = value.sel(time=slice(str(lo), str(hi)))
                std = float(part.std(("member", "time")))
                amplitudes.append(std)
                rows.append(dict(ensemble=name, mode=mode, period=key, std=std, units=unit))
            if "index_period" in config:
                xpos = 0 if name == "reference" else 1
                axes[1, 2].bar(xpos, amplitudes[0], color=color, alpha=.7)
                axes[1, 2].scatter(np.full(value.sizes["member"], xpos), value.std("time"), color=color, edgecolor="white", zorder=3)
            else:
                axes[1, 2].plot(list(periods), amplitudes, marker="o", color=color)
        axes[0, 0].set(xscale="log", yscale="log", xlabel="Period (years)", ylabel=f"PSD ({unit}² month)", title="Spectrum")
        axes[0, 0].legend(frameon=False, fontsize=9)
        axes[0, 1].axhline(0, color=".6", lw=.8)
        axes[0, 1].set(xlabel="Lag (years)", ylabel="Autocorrelation", ylim=(-1.05, 1.05), title="Lag ACF")
        axes[0, 2].set(xlabel=unit, ylabel="Density", title="Pooled index distribution")
        axes[1, 0].set(xticks=range(1, 13), xticklabels=[calendar.month_abbr[i][0] for i in range(1, 13)], ylabel=f"Std ({unit})", title="Seasonal phase locking")
        axes[1, 1].set(xlabel="Year", ylabel=unit, title="10-year samples (independent realizations)")
        axes[1, 2].set(ylabel=f"Std ({unit})", title="Training-period index amplitude" if "index_period" in config else "Variability across forcing regimes")
        if "index_period" in config:
            axes[1, 2].set(xticks=[0, 1], xticklabels=[labels["reference"], "Monthly Diffusion"])
        for ax in axes.ravel():
            ax.grid(alpha=.15)
        period = config.get("index_period", config["evaluation"])
        kind = "monthly anomalies, linearly detrended" if config.get("reference_kind") == "reanalysis" else "internal variability"
        fig.suptitle(f"{mode} · {kind} · {period[0]}–{period[1]}")
        finish(fig, root/"indices"/f"{mode.lower()}_comparison")
    pd.DataFrame(rows).to_csv(root/"indices"/"amplitudes.csv", index=False)


def plot_fields(root, config):
    import matplotlib.pyplot as plt
    groups = {group: [v.name for v in mdv1_variables() if v.component == group and v.name in config["variables"]]
              for group in ("atmosphere", "ocean")}
    labels = dict(LABELS, reference=config.get("reference_label", LABELS["reference"]))
    map_labels = (labels["reference"], labels["model"], "Difference")
    for variable in config["variables"]:
        log.info("Rendering forced/climatology/teleconnection maps: %s", variable)
        ref = xr.load_dataset(root/"reference"/f"{variable}.nc")
        model = xr.load_dataset(root/"model"/f"{variable}.nc")
        for key, title in (("forced_response", "Forced response"), ("climatology", "Training-period annual climatology")):
            if key == "forced_response" and key not in ref:
                field, units = display(variable, model[key])
                early, late = config["early"], config["late"]
                fig = plot_response_map(field, path=root/key/variable, units=units,
                    title=f"{variable} · {config.get('scenario_label', 'SSP3-7.0 projection')}\n"
                          f"{late[0]}–{late[1]} minus {early[0]}–{early[1]} · model only")
                plt.close(fig)
                continue
            r, m = ref[key], model[key]
            if key == "climatology":
                r, m = r.sel(period="training", drop=True), m.sel(period="training", drop=True)
            r, units = display(variable, r)
            m, _ = display(variable, m)
            fig = plot_map_comparison(r, m, path=root/key/variable, title=f"{variable} · {title}", units=units,
                                     symmetric=key != "climatology", labels=map_labels)
            plt.close(fig)
        for mode in ref.mode.values:
            r, units = display(variable, ref.regression.sel(mode=mode))
            m, _ = display(variable, model.regression.sel(mode=mode))
            fig = plot_map_comparison(r, m, path=root/"teleconnections"/f"{str(mode).lower()}_{variable}",
                title=f"{mode} · {variable}" + (f" · {config['index_period'][0]}–{config['index_period'][1]}" if "index_period" in config else ""),
                units=f"{units} per +1 SD {mode}", labels=map_labels)
            plt.close(fig)
    for group, variables in groups.items():
        if not variables:
            continue
        fig, axes = plt.subplots(int(np.ceil(len(variables)/3)), 3, figsize=(14, 2.8*int(np.ceil(len(variables)/3))), constrained_layout=True, squeeze=False)
        for ax, variable in zip(axes.ravel(), variables):
            for name in COLORS:
                with xr.open_dataset(root/name/f"{variable}.nc") as ds:
                    value, unit = display(variable, ds.global_mean.load())
                annual = annual_means(value)
                ax.fill_between(annual.year, annual.min("member"), annual.max("member"), color=COLORS[name], alpha=.15)
                ax.plot(annual.year, annual.mean("member"), color=COLORS[name], label=labels[name], lw=1.3)
                ax.set(title=variable, ylabel=unit, xlabel="Year")
            ax.axvspan(config["training"][0], config["training"][1]+1, color="#459957", alpha=.12)
            ax.grid(alpha=.15)
        for ax in axes.ravel()[len(variables):]:
            ax.set_visible(False)
        axes.ravel()[0].legend(frameon=False, fontsize=8)
        fig.suptitle("Annual global means · physical units · shading: member range; green: training")
        finish(fig, root/"time_series"/f"{group}_global_trajectories")


def plot_long_rollout(output, config):
    root = Path(output)
    plot_indices(root, config)
    plot_fields(root, config)
