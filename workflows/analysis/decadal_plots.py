"""Plots of retained decadal-SST products; no raw-archive reads or inference."""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr


def finish(fig, path):
    import matplotlib.pyplot as plt
    path = Path(path)
    fig.savefig(path.with_suffix(".png"), dpi=260, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def map_panel(ax, field, limit, *, signed=True, bounds=None):
    import cartopy.crs as ccrs
    from cartopy.util import add_cyclic_point
    values, longitude = add_cyclic_point(field.transpose("lat", "lon").values, coord=field.lon)
    image = ax.pcolormesh(longitude, field.lat, values, transform=ccrs.PlateCarree(),
        cmap="RdBu_r" if signed else "YlOrRd", vmin=-limit if signed else 0., vmax=limit,
        rasterized=True, shading="auto")
    if bounds is None or bounds == [-90, 90, 0, 360]:
        ax.set_global()
    else:
        south, north, west, east = bounds
        if west > east:
            west -= 360
        ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
    ax.set_facecolor(".91")
    ax.coastlines(resolution="110m", linewidth=.45, color=".25")
    return image


def bound(arrays):
    value = max(float(np.nanmax(np.abs(a))) for a in arrays)
    step = 10**np.floor(np.log10(max(value, 1.e-8)))/2
    return max(step, np.ceil(value/step)*step)


def plot_decadal(root, config):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullFormatter
    root = Path(root)
    colors = {n: s["color"] for n, s in config["sources"].items()}
    labels = {n: s["label"] for n, s in config["sources"].items()}
    projection = ccrs.Robinson(central_longitude=180)
    cutoff = config["cutoff_years"]
    half = config["filter_half_width"]
    region_projections = {}
    for region, spec in config["regions"].items():
        south, north, west, east = spec["bounds"]
        center = ((west-360+east)/2 if west > east else (west+east)/2)
        region_projections[region] = (projection if spec["bounds"] == [-90, 90, 0, 360]
                                      else ccrs.PlateCarree(central_longitude=center))
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False}):
        # Filter transparency: the plotted rolloff prevents treating a finite
        # low-pass kernel as a perfect division at exactly ten years.
        kernel = xr.load_dataset(root/"filter.nc")
        fig, axes = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True)
        axes[0].stem(kernel.lag_years, kernel.coefficients, basefmt=" ")
        axes[0].set(xlabel="Lag (years)", ylabel="Filter weight", title="Annual Lanczos kernel")
        axes[1].plot(kernel.frequency, kernel.amplitude_response, color=".2")
        axes[1].axvline(1/config["cutoff_years"], color=".5", ls="--", lw=1)
        axes[1].set(xlabel="Frequency (cycles/year)", ylabel="Amplitude response", xlim=(0, .5),
                    title=f"{cutoff}-year nominal cutoff")
        finish(fig, root/"filter_response")
        bases = {r: xr.load_dataset(root/f"basis_{r}.nc") for r in config["regions"]}
        fig = plt.figure(figsize=(12.5, 8.2))
        axes = np.array([[fig.add_subplot(len(bases), config["modes"], row*config["modes"]+col+1,
            projection=region_projections[region]) for col in range(config["modes"])]
            for row, region in enumerate(bases)])
        fig.subplots_adjust(left=.08, right=.98, top=.90, bottom=.08, wspace=.08, hspace=.32)
        limit = bound([b.loading for b in bases.values()])
        for row, (region, basis) in enumerate(bases.items()):
            for col in range(config["modes"]):
                image = map_panel(axes[row, col], basis.loading.isel(mode=col), limit, bounds=config["regions"][region]["bounds"])
                axes[row, col].set_title(f"EOF {col+1} · {float(basis.explained_fraction[col])*100:.1f}%", fontsize=11)
            box = axes[row, 0].get_position()
            fig.text(.025, (box.y0+box.y1)/2, config["regions"][region]["label"],
                rotation=90, va="center", ha="center", fontsize=10)
        fig.suptitle("E3SMv2 · historical low-frequency SST patterns", y=.975, fontsize=15)
        first_basis, last_basis = np.asarray(config["basis"]["years"])+[half, -half]
        fig.text(.53, .932, f"{first_basis}–{last_basis} retained · covariance EOFs, no local variance scaling",
            ha="center", fontsize=10, color=".3")
        bar = fig.colorbar(image, cax=fig.add_axes([.35, .04, .35, .014]), orientation="horizontal")
        bar.set_label("SST loading per +1 reference PC SD (°C)")
        finish(fig, root/"reference_eofs")

        for comparison, spec in config["comparisons"].items():
            folder = root/comparison
            names = spec["sources"]
            reference = spec["reference"]
            models = [n for n in names if n != reference]
            fields = {n: xr.load_dataset(folder/f"{n}_fields.nc") for n in names}
            first = int(fields[reference].filtered_year[0])
            last = int(fields[reference].filtered_year[-1])
            period = f"{first}–{last} retained · {cutoff}-year cutoff · {half} years trimmed at each end"
            subtitle = ("Per-trajectory linear detrend; NOT an isolated internal-variability estimate" if spec["method"] == "detrended"
                        else "Exact ensemble-mean removal + finite-ensemble correction; no linear detrend")
            height = 3*len(models)+1.6
            fig, axes = plt.subplots(len(models), 3, figsize=(12.5, height),
                subplot_kw=dict(projection=projection), squeeze=False)
            fig.subplots_adjust(left=.03, right=.97, top=1-.90/height,
                bottom=.85/height, hspace=.22, wspace=.07)
            limit = bound([fields[n].lowpass_std for n in names])
            diff_limit = bound([fields[n].lowpass_std-fields[reference].lowpass_std for n in models])
            for row, name in enumerate(models):
                arrays = [fields[reference].lowpass_std, fields[name].lowpass_std,
                          fields[name].lowpass_std-fields[reference].lowpass_std]
                for col, data in enumerate(arrays):
                    image = map_panel(axes[row, col], data, limit if col < 2 else diff_limit, signed=col == 2)
                    axes[row, col].set_title([labels[reference], labels[name], "Difference"][col], fontsize=11)
                    if row == 0 and col in (1, 2):
                        cax = fig.add_axes(([.12, .62/height, .44, .10/height] if col == 1
                                          else [.72, .62/height, .20, .10/height]))
                        fig.colorbar(image, cax=cax, orientation="horizontal",
                            label="Low-pass SST standard deviation (°C)" if col == 1 else "Difference (°C)")
            fig.suptitle(spec["label"]+" · SST variability", y=.98, fontsize=14)
            fig.text(.5, .98-.34/height, period, ha="center", fontsize=9)
            fig.text(.5, .008, subtitle+". Gray = unavailable/land.", ha="center", fontsize=9, color=".3")
            finish(fig, folder/"lowpass_std_maps")

            handles = [Line2D([0], [0], color=colors[n], lw=2, label=labels[n]) for n in names]
            for region in config["regions"]:
                pcs = {n: xr.load_dataset(folder/f"{n}_{region}_pcs.nc") for n in names}
                own = {n: xr.load_dataset(folder/f"{n}_{region}_independent_eofs.nc") for n in names}
                fig, axes = plt.subplots(config["modes"], 2, figsize=(13.3, 8.),
                    gridspec_kw=dict(width_ratios=[2.2, 1.]), squeeze=False)
                fig.subplots_adjust(left=.075, right=.98, bottom=.13, top=.85, hspace=.38, wspace=.23)
                for row, mode in enumerate(bases[region].mode.values):
                    for name in names:
                        ds = pcs[name].sel(mode=mode)
                        for j, member in enumerate(ds.member.values):
                            pc = ds.lowpass_pc.sel(member=member)
                            axes[row, 0].plot(pc.filtered_year, pc, color=colors[name], lw=1.35 if j == 0 else .8,
                                alpha=1. if j == 0 else .30)
                        psd = ds.psd.isel(frequency=slice(1, None))
                        period_axis = 1/psd.frequency
                        axes[row, 1].plot(period_axis, psd.mean("member"), color=colors[name], lw=1.7)
                        axes[row, 1].fill_between(period_axis, psd.min("member"), psd.max("member"), color=colors[name], alpha=.12)
                    axes[row, 0].axhline(0, color=".6", lw=.65)
                    axes[row, 0].set(xlim=(first, last), ylabel=f"PC {mode} (°C)")
                    axes[row, 1].axvline(cutoff, color=".5", ls="--", lw=.8)
                    axes[row, 1].set(xscale="log", yscale="log", xlim=(2, config["spectrum_segment_years"]), ylabel="PSD (°C² year)")
                    axes[row, 1].set_xticks([2, 5, 10, 20, 50], labels=["2", "5", "10", "20", "50"])
                    axes[row, 1].xaxis.set_minor_formatter(NullFormatter())
                    for ax in axes[row]:
                        ax.grid(alpha=.13)
                axes[0, 0].set_title("Low-pass PCs · individual members", fontsize=11)
                axes[0, 1].set_title("Unfiltered annual PC spectra", fontsize=11)
                axes[-1, 0].set_xlabel("Year")
                axes[-1, 1].set_xlabel("Period (years)")
                fig.suptitle(f"{config['regions'][region]['label']} · {spec['label']}", y=.98, fontsize=14)
                fig.text(.53, .934, "Fixed E3SMv2 historical basis; physical PC amplitudes, not independently standardized", ha="center", fontsize=10)
                fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .915), ncol=len(names), frameon=False, fontsize=9)
                fig.text(.52, .047, "Bold = first member; thin = other members. Spectral shading = member range, NOT confidence intervals.", ha="center", fontsize=9)
                fig.text(.52, .018, subtitle+". Independent simulations need not agree in phase.", ha="center", fontsize=9, color=".3")
                finish(fig, folder/f"{region}_pcs_spectra")

                fig, axes = plt.subplots(config["modes"], len(names), figsize=(4*len(names), 8.4),
                    subplot_kw=dict(projection=region_projections[region]), squeeze=False)
                fig.subplots_adjust(left=.035, right=.98, bottom=.14, top=.855, wspace=.06, hspace=.32)
                limit = bound([own[n].loading for n in names])
                for row, mode in enumerate(bases[region].mode.values):
                    for col, name in enumerate(names):
                        data = own[name].sel(mode=mode)
                        image = map_panel(axes[row, col], data.loading, limit, bounds=config["regions"][region]["bounds"])
                        rank, score, fraction = int(data.original_rank), float(data.reference_pattern_cosine), float(data.explained_fraction)
                        title = (labels[name]+"\n" if row == 0 else "")+f"EOF {rank} → ref {mode} · {fraction*100:.0f}% · cos {score:.2f}"
                        axes[row, col].set_title(title, fontsize=9)
                fig.suptitle(f"{config['regions'][region]['label']} · independently fitted low-frequency EOFs", y=.98, fontsize=14)
                fig.text(.51, .935, f"{spec['label']} · {first}–{last} · signs/ranks matched to historical reference", ha="center", fontsize=10)
                fig.colorbar(image, cax=fig.add_axes([.34, .083, .35, .015]), orientation="horizontal",
                    label="Loading per +1 own PC SD (°C)")
                fig.text(.51, .009, f"Matching is within the leading {config['modes']} EOFs only; similar eigenvalues can mix modes. Cos = wet-area pattern cosine.", ha="center", fontsize=9, color=".3")
                finish(fig, folder/f"{region}_independent_eofs")

        table = pd.read_csv(root/"variance_summary.csv")
        fig, axes = plt.subplots(2, 3, figsize=(13, 6.7))
        fig.subplots_adjust(left=.07, right=.985, bottom=.17, top=.87, wspace=.27, hspace=.14)
        groups = list(config["comparisons"])
        short = [config["comparisons"][g].get("short_label", g.replace("_", "\n")) for g in groups]
        for col, region in enumerate(config["regions"]):
            for i, name in enumerate(config["sources"]):
                data = table.loc[(table.source == name) & (table.region == region)].set_index("comparison")
                x = np.arange(len(groups))+(i-1.5)*.17
                std = [data.loc[g, "lowpass_rms_std_degC"] if g in data.index else np.nan for g in groups]
                fraction = [data.loc[g, "lowpass_variance_fraction"] if g in data.index else np.nan for g in groups]
                axes[0, col].bar(x, std, width=.16, color=colors[name])
                axes[1, col].bar(x, fraction, width=.16, color=colors[name])
            axes[0, col].set_title(config["regions"][region]["label"], fontsize=11)
            axes[0, col].set_xticks(np.arange(len(groups)), labels=[])
            axes[1, col].set_xticks(np.arange(len(groups)), labels=short, fontsize=9)
            for row in (0, 1):
                axes[row, col].grid(axis="y", alpha=.13)
        axes[0, 0].set_ylabel("Low-pass RMS std (°C)")
        axes[1, 0].set_ylabel("Low-pass / annual variance")
        fig.suptitle("Decadal SST variability · amplitude and fraction", y=.98, fontsize=15)
        fig.legend(handles=[Line2D([0], [0], color=colors[n], lw=5, label=labels[n]) for n in config["sources"]],
            loc="upper center", bbox_to_anchor=(.52, .947), ncol=4, frameon=False, fontsize=9)
        windows = "   |   ".join(f"{config['comparisons'][g].get('short_label', g)}: "
            f"{config['comparisons'][g]['years'][0]+half}–{config['comparisons'][g]['years'][1]-half}" for g in groups)
        fig.text(.53, .05, windows.replace("\n", " ")+" (retained)", ha="center", fontsize=8)
        fig.text(.53, .019, "Compare models within a group. Detrended trajectories are an approximate observational check, not isolated internal variability.", ha="center", fontsize=9, color=".3")
        finish(fig, root/"variance_summary")
