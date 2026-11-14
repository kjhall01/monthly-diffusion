"""Small optional plotting functions; importing the package needs no Cartopy."""
from pathlib import Path
import numpy as np


def plot_response_map(field, *, path, title="", units=""):
    """Model-only signed response; no fabricated future observational panel."""
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from cartopy.util import add_cyclic_point
    limit = max(float(np.nanmax(np.abs(field))), 1.e-12)
    fig, ax = plt.subplots(figsize=(7, 4), subplot_kw=dict(projection=ccrs.Robinson()))
    values, lon = add_cyclic_point(field.transpose("lat", "lon").values, coord=field.lon)
    image = ax.pcolormesh(lon, field.lat, values, transform=ccrs.PlateCarree(),
                         cmap="RdBu_r", vmin=-limit, vmax=limit, rasterized=True)
    ax.coastlines(linewidth=.5)
    ax.set_global()
    ax.set_title(title, fontsize=10)
    fig.colorbar(image, ax=ax, orientation="horizontal", shrink=.8, pad=.05, label=units)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    return fig


def plot_map_comparison(reference, model, *, path, title="", units=None,
                        labels=("Held-out E3SMv2", "Monthly Diffusion v1.0", "Difference"), symmetric=True):
    """Shared reference/model scale and separate symmetric difference scale."""
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from cartopy.util import add_cyclic_point
    from matplotlib.colors import TwoSlopeNorm, Normalize
    reference, model = __import__("xarray").align(reference, model, join="exact")
    arrays = [reference, model, model-reference]
    limit = max(float(np.nanmax(np.abs(reference))), float(np.nanmax(np.abs(model))), 1.e-12)
    difference_limit = max(float(np.nanmax(np.abs(arrays[2]))), 1.e-12)
    low = min(float(reference.min()), float(model.min()))
    high = max(float(reference.max()), float(model.max()), low+1.e-12)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), subplot_kw=dict(projection=ccrs.Robinson()))
    for i, (ax, data, label) in enumerate(zip(axes, arrays, labels)):
        values, lon = add_cyclic_point(data.transpose("lat", "lon").values, coord=data.lon)
        bound = limit if i < 2 else difference_limit
        norm = TwoSlopeNorm(0., -bound, bound) if symmetric or i == 2 else Normalize(low, high)
        image = ax.pcolormesh(lon, data.lat, values, transform=ccrs.PlateCarree(),
                             cmap="RdBu_r" if symmetric or i == 2 else "viridis",
                             norm=norm, rasterized=True)
        ax.coastlines(linewidth=.5)
        ax.set_global()
        ax.set_title(label, fontsize=10)
        fig.colorbar(image, ax=ax, orientation="horizontal", shrink=.8, pad=.05,
                     label=units or data.attrs.get("units", ""))
    fig.suptitle(title)
    fig.tight_layout()
    base = Path(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    return fig


def plot_training(history, path):
    import pandas as pd
    import matplotlib.pyplot as plt
    table = pd.read_csv(history)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(table.epoch, table.train_loss, label="Train (online)")
    ax.plot(table.epoch, table.val_loss, label="Validation (EMA)")
    ax.set(xlabel="Epoch", ylabel="Objective")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    return fig


def plot_response_matrix(reference, model, *, variables, labels, units, path,
                         column_titles=("E3SMv2", "MDv1-E3SMv2", "Difference"),
                         percentile=99., dpi=450):
    """Publication Robinson rows; physical maps supplied by the caller.

    No inference, averaging, or unit conversion occurs here. Reference/model
    share a symmetric scale per row; model-minus-reference has a separate
    symmetric scale. Rounded absolute percentiles set display limits only;
    colorbar extensions disclose saturation and metadata records its extent.
    Contours interpolate visually between native cell centers; stored maps
    remain unchanged. Letters run left-to-right, then top-to-bottom.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from cartopy.util import add_cyclic_point
    import xarray as xr
    from matplotlib.ticker import MaxNLocator
    reference, model = xr.align(reference, model, join="exact")
    if not variables or len(set(variables)) != len(variables):
        raise ValueError("Select at least one unique variable")
    if len(column_titles) != 3 or not 0 < percentile <= 100:
        raise ValueError("Need three column titles and a percentile in (0, 100]")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(column_titles=list(column_titles), difference="model minus reference",
                    title=None, percentile=percentile, panels=[])
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(len(variables), 3, figsize=(13.8, 3.15*len(variables)),
            subplot_kw=dict(projection=ccrs.Robinson()), squeeze=False, layout="constrained")
        fig.get_layout_engine().set(w_pad=.08, h_pad=.10, hspace=.08, wspace=.025)
        for column, title in enumerate(column_titles):
            axes[0, column].set_title(title, fontsize=13, weight="bold", pad=12)
        for row, variable in enumerate(variables):
            fields = [reference[variable], model[variable], model[variable]-reference[variable]]
            if any(set(f.dims) != {"lat", "lon"} for f in fields):
                raise ValueError("Supply already-averaged latitude/longitude maps")
            limits = []
            for group in (fields[:2], fields[2:]):
                values = np.concatenate([np.asarray(f).ravel() for f in group])
                values = np.abs(values[np.isfinite(values)])
                if not len(values):
                    raise ValueError(f"No finite data for {variable}")
                value = max(float(np.percentile(values, percentile)), 1.e-12)
                step = 10**np.floor(np.log10(value))/2
                limits.append(float(np.ceil(value/step)*step))
            images = []
            cmap = "BrBG" if variable == "mtpr" else "RdBu_r"
            for col, (ax, field) in enumerate(zip(axes[row], fields)):
                limit = limits[col == 2]
                values, lon = add_cyclic_point(field.transpose("lat", "lon").values, coord=field.lon)
                lat = field.lat.values
                # Same visual pole closure as the established publication plots.
                if lat[0] > -89.99:
                    lat, values = np.r_[-89.99, lat], np.concatenate((values[:1], values))
                if lat[-1] < 89.99:
                    lat, values = np.r_[lat, 89.99], np.concatenate((values, values[-1:]))
                images.append(ax.contourf(lon, lat, values, levels=np.linspace(-limit, limit, 25),
                    cmap=cmap, extend="both", transform=ccrs.PlateCarree(), antialiased=True))
                ax.set_global()
                ax.set_facecolor("#f7f7f7")
                ax.coastlines(resolution="110m", color=".22", linewidth=.42)
                letter = chr(ord("a")+3*row+col)+")"
                ax.text(.015, .975, letter, transform=ax.transAxes, ha="left", va="top",
                    fontsize=13, weight="bold", zorder=6)
                finite = np.asarray(field)[np.isfinite(field)]
                metadata["panels"].append(dict(variable=variable, panel=letter,
                    column=column_titles[col], units=units[variable], colorbar_label=f"{labels[variable]} ({units[variable]})",
                    limit=limit, cmap=cmap, clipped_fraction=float(np.mean(np.abs(finite) > limit))))
            for image, group, aspect in ((images[0], axes[row, :2].tolist(), 40), (images[2], axes[row, 2], 19)):
                bar = fig.colorbar(image, ax=group, orientation="horizontal", fraction=.065,
                    pad=.055, aspect=aspect, label=f"{labels[variable]} ({units[variable]})")
                # BoundaryNorm colorbars can clamp out-of-range locator ticks
                # to their ends, mislabeling the limits and crowding labels.
                ticks = MaxNLocator(5).tick_values(image.norm.vmin, image.norm.vmax)
                tolerance = 1.e-10 * max(abs(image.norm.vmin), abs(image.norm.vmax))
                ticks = ticks[(ticks >= image.norm.vmin-tolerance) & (ticks <= image.norm.vmax+tolerance)]
                bar.set_ticks(np.clip(ticks, image.norm.vmin, image.norm.vmax))
                bar.ax.tick_params(labelsize=9, length=2.5)
        for extension in ("png", "pdf"):
            fig.savefig(path.with_suffix("."+extension), dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return metadata
