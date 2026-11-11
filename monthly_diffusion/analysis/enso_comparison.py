"""Matched-period Niño3.4 diagnostics in physical units for four data sources.

Anomalization is supplied by enso_diversity.regional_trajectory_anomalies:
each individual trajectory loses a jointly fitted monthly mean and linear
trend over the common analysis interval. No ensemble mean is subtracted and
no index is standardized. This permits the same treatment of single-trajectory
reanalysis and ensemble simulations. It is descriptive, not forecast skill.
"""
from pathlib import Path
import calendar

import numpy as np
import xarray as xr
from scipy.stats import gaussian_kde

from .spectra import lag_autocorrelation, temporal_spectrum


def enso_statistics(index, *, density_grid, bandwidth_degC=.2, segment_months=240, max_lag=120):
    """Memberwise Welch/ACF; equal-member KDE; calendar-month amplitude ratio.

    Never concatenate member trajectories for temporal statistics. Welch uses
    Hann windows, 50% overlap, and dt=1/12 year, giving PSD in °C² year. No
    additional linear detrend is applied here. KDE uses the SAME absolute
    Gaussian bandwidth for every source/member, not a sample-count-dependent
    smoothing width. Seasonal amplification = pooled monthly SD / pooled SD;
    this is a measure of seasonal amplitude modulation, not a significance test.
    """
    index = index.transpose("member", "time")
    if not np.isfinite(index).all():
        raise ValueError("ENSO statistics require finite contiguous index data; no interpolation")
    keys = index.time.dt.year.values*12 + index.time.dt.month.values
    if np.any(np.diff(keys) != 1):
        raise ValueError("ENSO statistics require consecutive monthly samples")
    if float(index.std("time").min()) <= 0 or bandwidth_degC <= 0:
        raise ValueError("Need nonconstant trajectories and a positive KDE bandwidth")
    psd = temporal_spectrum(index, dt=1/12, nperseg=segment_months, noverlap=segment_months//2)
    acf = lag_autocorrelation(index, max_lag=max_lag, detrend="none")
    density = np.stack([gaussian_kde(x, bw_method=bandwidth_degC/np.std(x, ddof=1))(density_grid)
                        for x in index.values])
    seasonal = index.groupby("time.month").std("time", ddof=0)
    pooled_seasonal = index.groupby("time.month").std(("member", "time"), ddof=0)
    pooled_std = float(index.std(("member", "time"), ddof=0))
    result = xr.Dataset(dict(psd=psd, acf=acf,
        density=(("member", "anomaly"), density),
        seasonal_std=seasonal, pooled_seasonal_std=pooled_seasonal,
        seasonal_amplification=seasonal/index.std("time", ddof=0),
        pooled_seasonal_amplification=pooled_seasonal/pooled_std,
        member_std=index.std("time", ddof=0)),
        coords=dict(anomaly=density_grid))
    result.attrs.update(pooled_std_degC=pooled_std, kde_bandwidth_degC=bandwidth_degC,
        samples_per_member=index.sizes["time"], member_count=index.sizes["member"],
        anomaly_definition=index.attrs.get("anomaly_definition", "supplied already-anomalized index"))
    result.psd.attrs["units"] = "degC^2 year"
    result.frequency.attrs["units"] = "year-1"
    result.density.attrs["units"] = "degC-1"
    result.seasonal_std.attrs["units"] = "degC"
    result.pooled_seasonal_std.attrs["units"] = "degC"
    return result


def plot_enso_comparison(statistics, config, path, coverage):
    """Four-panel landscape, common colors, physical amplitudes retained.

    Shading is the 10--90% MEMBER range, not confidence intervals. Reanalysis
    has one trajectory and therefore no such band. PSD/ACF/KDE bold lines are
    equal-member means; the seasonal line uses pooled sample SDs.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import ScalarFormatter, MaxNLocator

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10.5,
                         "axes.labelsize": 11, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 4, figsize=config.get("figsize", [17.6, 4.9]))
        fig.subplots_adjust(left=.055, right=.987, top=.86, bottom=.25, wspace=.32)
        for j, ax in enumerate(axes):
            ax.set_title(f"{chr(97+j)})", loc="left", fontsize=12, weight="bold", pad=12)
            ax.tick_params(direction="out", length=3, labelsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(".4")
            ax.grid(axis="y", color=".90", lw=.6, zorder=0)
        for source, ds in statistics.items():
            color = config["sources"][source]["color"]
            # Decreasing frequency means increasing period; sort for fill_between.
            psd = ds.psd.isel(frequency=slice(1, None)).sortby("frequency", ascending=False)
            series = [(axes[0], 1/psd.frequency.values, psd),
                      (axes[1], ds.anomaly.values, ds.density),
                      (axes[2], ds.lag_months.values/12, ds.acf),
                      (axes[3], ds.month.values, ds.seasonal_amplification)]
            for ax, x, y in series:
                if ds.sizes["member"] > 1:
                    low, high = y.quantile([.1, .9], "member").values
                    ax.fill_between(x, low, high, color=color, alpha=.10, linewidth=0, zorder=1)
                mean = ds.pooled_seasonal_amplification if ax is axes[3] else y.mean("member")
                ax.plot(x, mean, color=color, lw=1.9, zorder=3,
                        marker="o" if ax is axes[3] else None, markersize=3.0)
        axes[0].set(xscale="log", yscale="log", xlabel="Period (years)",
                    ylabel="Power spectral density", xlim=(1/6, config["segment_months"]/12))
        axes[0].set_xticks([.25, .5, 1, 2, 5, 10, 20])
        axes[0].xaxis.set_major_formatter(ScalarFormatter())
        axes[0].set_xticklabels(["0.25", "0.5", "1", "2", "5", "10", "20"])
        axes[1].set(xlabel="Niño3.4 anomaly (°C)", ylabel="Probability density", ylim=(0, None))
        axes[1].xaxis.set_major_locator(MaxNLocator(5))
        axes[2].axhline(0, color=".45", lw=.8, zorder=0)
        axes[2].set(xlabel="Lag (years)", ylabel="Autocorrelation", xlim=(0, config["max_lag_months"]/12), ylim=(-.65, 1.03))
        axes[3].axhline(1, color=".45", ls=(0, (3, 3)), lw=.8, zorder=0)
        axes[3].set(xlabel="Calendar month", ylabel="Monthly SD / overall SD", xlim=(.7, 12.3))
        axes[3].set_xticks(range(1, 13), [calendar.month_abbr[i][0] for i in range(1, 13)])
        handles = [Line2D([], [], color=s["color"], lw=2.7, label=s["label"]) for s in config["sources"].values()]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.51, .055),
                   ncol=4, frameon=False, fontsize=11, columnspacing=2.0, handlelength=2.3)
        path = Path(path)
        for ext in ("png", "pdf"):
            fig.savefig(path.with_suffix("."+ext), dpi=config.get("dpi", 400), bbox_inches="tight", facecolor="white")
        plt.close(fig)
