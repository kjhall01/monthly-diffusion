"""Two compact figures per variable for the main evaluation scripts."""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point


def plot_series(series, path):
    """Plot monthly anomalies plus raw reconstruction bias/RMSE when available."""
    rows = 3 if "global_rmse" in series else 1
    fig, axes = plt.subplots(
        rows, 1, figsize=(12, 3 * rows), squeeze=False, sharex=True
    )
    ax = axes[0, 0]
    year = series.time.dt.year + (series.time.dt.month - 0.5) / 12
    for name, label, color in (
        ("reference_global_anomaly", "Reference", "black"),
        ("generated_global_anomaly", "Generated", "tab:blue"),
    ):
        field = series[name]
        if "member" in field.dims:
            for member in field.member:
                ax.plot(year, field.sel(member=member), color=color, alpha=0.18, lw=0.5)
            field = field.mean("member", skipna=False)
            label += " ensemble mean"
        ax.plot(year, field, color=color, lw=0.8, label=label)
    ax.set_title(
        f"{series.attrs['variable']}: global-mean monthly anomalies ({series.attrs['baseline']} baseline)"
    )
    ax.legend(frameon=False)
    if rows == 3:
        for ax, name, label in zip(
            axes[1:, 0],
            ("global_bias", "global_rmse"),
            ("Global-mean bias (generated − reference)", "Area-weighted spatial RMSE"),
        ):
            ax.plot(year, series[name], lw=0.8)
            ax.set_title(label)
    for ax in axes[:, 0]:
        ax.set_ylabel(series.attrs["units"])
        ax.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Year")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def color_limit(*fields):
    finite = np.concatenate([np.asarray(field).ravel() for field in fields])
    finite = finite[np.isfinite(finite)]
    return max(float(np.max(np.abs(finite))) if len(finite) else 0.0, 1e-12)


def draw_map(fig, position, field, title, units, *, limit, positive=False):
    ax = fig.add_subplot(position, projection=ccrs.Robinson())
    field = field.transpose("lat", "lon")
    values, longitude = add_cyclic_point(field.values, coord=field.lon.values)
    artist = ax.pcolormesh(
        longitude,
        field.lat,
        values,
        transform=ccrs.PlateCarree(),
        cmap="viridis" if positive else "RdBu_r",
        vmin=0 if positive else -limit,
        vmax=limit,
        shading="auto",
    )
    ax.set_global()
    ax.coastlines(linewidth=0.5)
    ax.set_title(title, fontsize=10)
    fig.colorbar(
        artist, ax=ax, orientation="horizontal", pad=0.05, shrink=0.85, label=units
    )


def plot_maps(maps, path):
    reconstruction = "rmse" in maps
    responses = "reference_response" in maps
    fig = plt.figure(figsize=(15, 8 if responses else 4), layout="constrained")
    grid = fig.add_gridspec(2 if responses else 1, 6)
    units, baseline = maps.attrs["units"], maps.attrs["baseline"]
    if reconstruction:
        draw_map(
            fig,
            grid[0, :3],
            maps.rmse,
            "Reconstruction RMSE (all months)",
            units,
            limit=color_limit(maps.rmse),
            positive=True,
        )
        draw_map(
            fig,
            grid[0, 3:],
            maps.climatological_bias,
            f"Mean bias, {baseline}",
            units,
            limit=color_limit(maps.climatological_bias),
        )
    else:
        draw_map(
            fig,
            grid[0, :2],
            maps.climatological_bias,
            f"Mean bias, {baseline}",
            units,
            limit=color_limit(maps.climatological_bias),
        )
        limit = color_limit(maps.reference_anomaly_sd, maps.generated_anomaly_sd)
        for position, name, label in (
            (grid[0, 2:4], "reference_anomaly_sd", "Reference"),
            (grid[0, 4:], "generated_anomaly_sd", "Generated"),
        ):
            draw_map(
                fig,
                position,
                maps[name],
                f"{label} anomaly SD, {baseline}",
                units,
                limit=limit,
                positive=True,
            )
    if responses:
        limit = color_limit(maps.reference_response, maps.generated_response)
        for position, name, label in (
            (grid[1, :2], "reference_response", "Reference response"),
            (grid[1, 2:4], "generated_response", "Generated response"),
            (grid[1, 4:], "response_difference", "Response difference"),
        ):
            draw_map(
                fig,
                position,
                maps[name],
                label,
                units,
                limit=color_limit(maps[name])
                if name == "response_difference"
                else limit,
            )
    fig.suptitle(
        f"{maps.attrs['variable']} | generated − reference | response: {maps.attrs['response']}"
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
