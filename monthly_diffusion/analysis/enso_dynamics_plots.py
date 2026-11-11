"""Publication plots of retained ENSO diagnostics; no archive/model reads."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import gaussian_kde
from .lagged_enso import SEASONS

STYLE = {
    "sst": (1., "SST", "°C", "RdBu_r"),
    "ohc300": (1.e-9, "OHC 0–300 m", "GJ m⁻²", "RdBu_r"),
    "d20": (1., "D20", "m", "RdBu_r"),
    "tau_x": (1., "Eastward Surface Stress", "N m⁻²", "BrBG"),
    "ssh": (100., "Sea Surface Height", "cm", "RdBu_r"),
}
COLORS = {"model": "#2378b5", "e3sm": "#303030", "reanalysis": "#c67428"}


def symmetric_limit(arrays, percentile=99.5):
    values = np.concatenate([np.asarray(a).ravel() for a in arrays])
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise ValueError("No finite data to plot")
    bound = max(float(np.percentile(abs(finite), percentile)), 1.e-12)
    step = 10**np.floor(np.log10(bound))/2
    limit = float(np.ceil(bound/step)*step)
    return limit, dict(limit=limit, percentile=percentile,
        clipped_fraction=float(np.mean(abs(finite) > limit)))


def finish(fig, path):
    import matplotlib.pyplot as plt
    path = Path(path)
    fig.savefig(path.with_suffix(".png"), dpi=280, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), dpi=280, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_seasonal(root, config):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    metadata = {}
    for variable in config["map_variables"]:
        factor, title, units, cmap = STYLE[variable]
        arrays = [xr.load_dataset(root/name/f"seasonal_{variable}.nc").supported_slope.sel(anomaly_method="trajectory", drop=True)
                  .sel(lat=slice(-30, 30), lon=slice(120, 280))*factor for name in config["sources"]]
        arrays.append(arrays[0]-arrays[1])
        labels = [spec["label"] for spec in config["sources"].values()]+["MD − E3SMv2"]
        bound, metadata[variable] = symmetric_limit(arrays)
        metadata[variable].update(display_factor=factor, display_units=units+" per °C Niño3.4")
        fig, axes = plt.subplots(4, 4, figsize=(16, 8.7),
            subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=180)))
        fig.subplots_adjust(left=.085, right=.985, top=.88, bottom=.12, wspace=.09, hspace=.22)
        for row, values in enumerate(arrays):
            for col, season in enumerate(SEASONS):
                ax = axes[row, col]
                field = values.sel(season=season)
                image = ax.pcolormesh(field.lon, field.lat, field, transform=ccrs.PlateCarree(),
                    cmap=cmap, vmin=-bound, vmax=bound, rasterized=True, shading="auto")
                ax.set_extent((120, 280, -30, 30), crs=ccrs.PlateCarree())
                ax.set_facecolor(".9")
                ax.coastlines(resolution="110m", linewidth=.45, color=".25")
                ax.set_xticks([140, 180, 220, 260], crs=ccrs.PlateCarree())
                ax.set_yticks([-20, 0, 20], crs=ccrs.PlateCarree())
                ax.xaxis.set_major_formatter(LongitudeFormatter())
                ax.yaxis.set_major_formatter(LatitudeFormatter())
                ax.tick_params(labelsize=8, labelbottom=row == 3, labelleft=col == 0, length=2)
                if row == 0: ax.set_title(season.replace("-1", "−1"), fontsize=12)
        fig.canvas.draw()
        for row, label in enumerate(labels):
            b = axes[row, 0].get_position()
            fig.text(.025, (b.y0+b.y1)/2, label, rotation=90, ha="center", va="center", fontsize=11)
        cax = fig.add_axes([.33, .055, .4, .016])
        fig.colorbar(image, cax=cax, orientation="horizontal", extend="both").set_label(units+" per °C Niño3.4")
        fig.suptitle(f"{title} · Lagged ENSO Response", y=.975, fontsize=16)
        fig.text(.535, .935, f"DJF anchor {config['winter_years'][0]}–{config['winter_years'][1]} · regression per +1°C · monthly climatology + linear trend removed",
                 ha="center", fontsize=10, color=".3")
        finish(fig, root/f"lagged_{variable}")
    return metadata


def plot_hovmollers(root, config):
    import matplotlib.pyplot as plt
    variables = ["sst", "tau_x", "d20", "ssh", "ohc300"]
    names = list(config["sources"])
    fig, axes = plt.subplots(3, 5, figsize=(17, 9), sharex=True, sharey=True)
    fig.subplots_adjust(left=.085, right=.99, bottom=.16, top=.88, hspace=.13, wspace=.10)
    metadata = {}
    for col, variable in enumerate(variables):
        factor, title, unit, cmap = STYLE[variable]
        fields = [xr.load_dataset(root/name/f"hovmoller_{variable}.nc").slope
                  .sel(anomaly_method="trajectory", lon=slice(120, 280))*factor for name in names]
        limit, metadata[variable] = symmetric_limit(fields)
        for row, field in enumerate(fields):
            ax = axes[row, col]
            image = ax.pcolormesh(field.lon, field.lag, field, cmap=cmap, vmin=-limit, vmax=limit,
                                 shading="auto", rasterized=True)
            ax.axhline(0, color=".25", linewidth=.65, ls="--")
            ax.set(ylim=(config["lag_months"], -config["lag_months"]), xlim=(120, 280),
                   xticks=[140, 180, 220, 260], xticklabels=["140°E", "180°", "140°W", "100°W"],
                   yticks=[-12, -6, 0, 6, 12])
            ax.tick_params(labelsize=9)
            if row == 0: ax.set_title(title, fontsize=11)
            if col == 0: ax.set_ylabel("Months from January", fontsize=9)
        b = axes[2, col].get_position()
        cax = fig.add_axes([b.x0+.009, .095, b.width-.018, .015])
        bar = fig.colorbar(image, cax=cax, orientation="horizontal", extend="both")
        bar.set_label(unit+" / °C", fontsize=9)
        bar.ax.tick_params(labelsize=8)
    for row, name in enumerate(names):
        b = axes[row, 0].get_position()
        fig.text(.02, (b.y0+b.y1)/2, config["sources"][name]["label"], rotation=90,
                 ha="center", va="center", fontsize=11)
    fig.suptitle("Equatorial Pacific ENSO Evolution", fontsize=16, y=.97)
    fig.text(.535, .925, "5°S–5°N mean · regression per +1°C DJF Niño3.4 · negative lag precedes January",
             ha="center", fontsize=10)
    finish(fig, root/"equatorial_lagged_regressions")
    return metadata


def plot_event_composites(root, config):
    import matplotlib.pyplot as plt
    names = list(config["sources"])
    fields = [xr.load_dataset(root/name/"hovmoller_sst.nc").composite.sel(lon=slice(120, 280)) for name in names]
    limit, metadata = symmetric_limit(fields)
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True, sharey=True)
    fig.subplots_adjust(left=.12, right=.98, bottom=.13, top=.88, hspace=.20, wspace=.13)
    for row, name in enumerate(names):
        d20 = xr.load_dataset(root/name/"hovmoller_d20.nc").composite.sel(lon=slice(120, 280))
        events = pd.read_csv(root/name/"events.csv")
        for col, phase in enumerate(("El Niño", "La Niña")):
            ax = axes[row, col]
            field = fields[row].sel(phase=phase)
            image = ax.pcolormesh(field.lon, field.lag, field, cmap="RdBu_r", vmin=-limit, vmax=limit,
                                 shading="auto", rasterized=True)
            contours = ax.contour(d20.lon, d20.lag, d20.sel(phase=phase),
                levels=[-20, -10, 10, 20], colors=".15", linewidths=.6,
                linestyles=["--", "--", "-", "-"])
            ax.clabel(contours, inline=True, fontsize=7, fmt="%d")
            ax.axhline(0, color=".25", lw=.7, ls=":")
            n = len(events[(events.phase == phase) & events.used])
            ax.text(.02, .95, f"{n} events", transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=2))
            ax.set(ylim=(config["lag_months"], -config["lag_months"]), xlim=(120, 280),
                yticks=[-12, -6, 0, 6, 12], xticks=[120, 160, 200, 240, 280],
                xticklabels=["120°E", "160°E", "160°W", "120°W", "80°W"])
            if row == 0: ax.set_title(phase, fontsize=13)
            if col == 0: ax.set_ylabel("Months from event peak")
        b = axes[row, 0].get_position()
        fig.text(.025, (b.y0+b.y1)/2, config["sources"][name]["label"], rotation=90,
                 ha="center", va="center", fontsize=11)
    cax = fig.add_axes([.32, .06, .48, .017])
    fig.colorbar(image, cax=cax, orientation="horizontal", extend="both").set_label("SST anomaly (°C)")
    fig.suptitle("ENSO Event Evolution · SST and Thermocline", fontsize=16, y=.975)
    fig.text(.55, .93, "5°S–5°N · one sample per episode · contours: D20 anomaly (m; negative dashed)", ha="center", fontsize=10)
    finish(fig, root/"event_sst_d20_composites")
    return metadata


def plot_ec(root, config):
    import matplotlib.pyplot as plt
    all_events = {name: pd.read_csv(root/name/"events.csv") for name in config["sources"]}
    events = {name: table[table.used] for name, table in all_events.items()}
    combined = pd.concat(list(events.values()), ignore_index=True)
    bounds = []
    for name in ("E", "C"):
        lo, hi = combined[name].min(), combined[name].max()
        margin = max((hi-lo)*.12, .3)
        bounds.append((lo-margin, hi+margin))
    ex, cy = np.linspace(*bounds[0], 180), np.linspace(*bounds[1], 180)
    xx, yy = np.meshgrid(ex, cy)
    n_sources = len(events)
    fig = plt.figure(figsize=(max(11, 5.17*n_sources), 5.7))
    outer = fig.add_gridspec(1, n_sources, left=.06, right=.99, top=.82, bottom=.16, wspace=.21)
    densities = []
    for col, (name, table) in enumerate(events.items()):
        grid = outer[col].subgridspec(4, 4, hspace=.03, wspace=.03)
        ax = fig.add_subplot(grid[1:, :3])
        top = fig.add_subplot(grid[0, :3], sharex=ax)
        side = fig.add_subplot(grid[1:, 3], sharey=ax)
        for phase, color in (("El Niño", "#c44536"), ("La Niña", "#2878b5")):
            part = table[table.phase == phase]
            ax.scatter(part.E, part.C, s=19, color=color, alpha=.65, edgecolor="white", linewidth=.25,
                       label=f"{phase}: {len(part)}")
            if len(part) >= 5:
                top.plot(ex, gaussian_kde(part.E)(ex), color=color, lw=1.25)
                side.plot(gaussian_kde(part.C)(cy), cy, color=color, lw=1.25)
        if len(table) >= 6:
            density = gaussian_kde(table[["E", "C"]].values.T)(np.stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            sorted_density = np.sort(density.ravel())[::-1]
            probability = np.cumsum(sorted_density)/sorted_density.sum()
            levels = sorted([sorted_density[np.searchsorted(probability, q)] for q in (.5, .9)])
            ax.contour(xx, yy, density, levels=levels, colors=".4", linewidths=.8)
        else:
            density = np.full(xx.shape, np.nan)
        densities.append(density)
        ax.axhline(0, color=".7", lw=.6)
        ax.axvline(0, color=".7", lw=.6)
        ax.set(xlim=bounds[0], ylim=bounds[1], xlabel="E (°C)", ylabel="C (°C)")
        ax.legend(loc="lower right", frameon=False, fontsize=8, handletextpad=.2)
        ax.grid(alpha=.13)
        top.set_title(config["sources"][name]["label"], fontsize=12, pad=9)
        for margin_ax in (top, side):
            margin_ax.set_axis_off()
    fig.suptitle(config.get("diversity_title", "ENSO Diversity · E/C Event-Peak Distribution"), y=.965, fontsize=16)
    fig.text(.5, .895, "Takahashi Niño-box approximations · one centered 3-month peak per episode · no standardization",
             ha="center", fontsize=10)
    fig.text(.5, .035, "Warm: Niño3.4 ≥ +0.5°C; cold: ≤ −0.5°C · all episode lengths · contours: 50% / 90% joint KDE mass",
             ha="center", fontsize=10)
    finish(fig, root/"ec_event_distribution")
    from .long_rollout import save_dataset
    save_dataset(xr.Dataset(dict(density=(("source", "C", "E"), np.stack(densities))),
        coords=dict(source=list(events), E=ex, C=cy),
        attrs=dict(definition="Gaussian KDE of one peak per event; Scott bandwidth; not a significance test")),
        root/"ec_joint_density.nc")


def plot_lag_curves(root, config):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True, sharey=True, constrained_layout=True)
    for ax, variable in zip(axes.ravel(), config["map_variables"]):
        for name, spec in config["sources"].items():
            table = pd.read_csv(root/name/"regional_lag_correlations.csv")
            part = table[table.variable == variable]
            ax.plot(part.lag, part.correlation, lw=1.6, color=COLORS[name], label=spec["label"])
        ax.axhline(0, color=".6", lw=.7)
        ax.axvline(0, color=".6", lw=.7, ls=":")
        region = "160°E–150°W" if variable == "tau_x" else "120°E–80°W"
        ax.set(title=STYLE[variable][1]+f" · {region}", xlim=(-18, 18), ylim=(-1, 1),
               xlabel="Field lag relative to Niño3.4 (months)", ylabel="Correlation")
        ax.grid(alpha=.12)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("Recharge / Wind Coupling · 5°S–5°N\nNegative lag: field leads Niño3.4; positive lag: field follows", fontsize=13)
    finish(fig, root/"recharge_wind_lag_correlations")


def plot_dynamics(root, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(root)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42}):
        meta = dict(seasonal=plot_seasonal(root, config), hovmoller=plot_hovmollers(root, config),
                    event_composites=plot_event_composites(root, config))
        plot_ec(root, config)
        plot_lag_curves(root, config)
    (root/"display.json").write_text(json.dumps(meta, indent=2)+"\n")
