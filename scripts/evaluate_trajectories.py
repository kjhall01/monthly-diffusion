
## Evaluate Autoencoder: Can the decoder produce realistic future climate variability? 
## - time series of per-variable NRMSE between 1850-2100  
#  ---- shows global average RMSE as a function of time, highlights change between periods
## - RMSE reconstruction Maps in 1850-1958, 1958-2026, 2030-2100
## ---- highlights spatial patterns of Error  in different periods 
## - Annual-mean Bias Maps in 1850-1958, 1958-2026, 2030-2100
## ---- highlihgts spatial patterns of bias in different periods (expect cold bias in future)
## - Annual-mean "forced change" maps 2070-2100 minus 1850-1880 
## ---- shows change over time (due to CO2 but okay ya know) 

## Bar charts: Global-mean RMSE and Bias in the three periods for each variable (one bar per period-variable)
## Bar charts: Global-mean RMSE in the forced response spatial patterns per variable
## Bar chart:  global-mean forced response 

## maybe bar chart for ensemble variability? 

# trajectory postage stamps (time series)? 


import xarray as xr
import matplotlib.pyplot as plt 
import numpy as np 
import cartopy.crs as ccrs 
from pathlib import Path
import torch

component = "ocean"
project_dir = Path(__file__).resolve().parents[1]
run_dir = project_dir / "output/heptagon" / component
runs = {
    "Heptagon": run_dir,
    "Production": project_dir / "output/umd-production" / component,
}
models = list(runs)
output_dir = run_dir / "evaluation"
output_dir.mkdir(parents=True, exist_ok=True)

# Keep the requested periods, including the shared year 1958.
periods = {
    "1850-1958": slice("1850", "1958"),
    "1958-2026": slice("1958", "2026"),
    "2030-2100": slice("2030", "2100"),
}
early = slice("1850", "1880")
late = slice("2070", "2100")
colors = ["C0", "C1", "C2"]

trajectories = {
    model: xr.open_mfdataset(
        str(folder / "ae_trajectory/reconstruction_*.nc"),
        combine="by_coords", chunks={'time': 24},
    ) for model, folder in runs.items()
}
if component == "ocean":
    trajectories["Production"] = trajectories["Production"].rename(
        {"ohc700_minus_ohc300": "ohc300_700"},
    )
reference = xr.open_dataset(
    "/Users/kyle/Dev/downloads/e3smv2_monthly_1deg_r1i1p1f1_185001-210012.nc",
    chunks={'time': 24},
)
checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
scales = xr.DataArray(
    checkpoint["preprocessor"]["std_torch"].numpy().reshape(-1),
    dims="variable", coords={"variable": checkpoint["config"]["variables"]},
)
# Compare shared physical variables; production sivol is not the current model's sit.
variables = [name for name in trajectories["Heptagon"].data_vars
             if all(name in ds for ds in trajectories.values())]
weights = np.cos(np.deg2rad(reference.lat))

monthly_metrics = []
period_metrics = []
forced_metrics = []
annual_series = []

for variable in variables:
    print(f"Evaluating {component}: {variable}", flush=True)
    units = reference[variable].attrs.get("units", "")
    scale = float(scales.sel(variable=variable))

    # Both models use the same months, observed cells and Heptagon training SD.
    ae = xr.concat(
        [ds[variable] for ds in trajectories.values()],
        dim=xr.IndexVariable("model", models), join="inner",
    ).compute()
    ref = reference[variable].sel(time=ae.time).compute()
    observed = np.isfinite(ae).all("model") & np.isfinite(ref)
    ae = ae.where(observed)
    ref = ref.where(observed)
    error = ae - ref

    # Average squared error first, THEN take the root. NRMSE uses the training SD.
    spatial_mse = (error**2).weighted(weights).mean(['lat', 'lon'])
    monthly = xr.Dataset({
        "rmse": np.sqrt(spatial_mse),
        "nrmse": np.sqrt(spatial_mse) / scale,
        "bias": error.weighted(weights).mean(['lat', 'lon']),
    })
    monthly_metrics.append(monthly.expand_dims(variable=[variable]))

    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    for model, color in zip(models, colors):
        monthly.nrmse.sel(model=model).plot.line(
            ax=ax, x="time", color=color, label=model, linewidth=0.7,
        )
    for (label, period), color in zip(periods.items(), colors):
        ax.axvspan(np.datetime64(period.start), np.datetime64(str(int(period.stop) + 1)),
                   color=color, alpha=0.12, label=label)
    ax.set_title(f"{variable}: monthly global reconstruction NRMSE")
    ax.set_ylabel("RMSE / Heptagon training SD")
    ax.set_xlabel("")
    ax.legend()
    fig.savefig(output_dir / f"{variable}_nrmse_timeseries.png", dpi=150)
    plt.close(fig)

    # Annual means give each available month equal weight.
    annual_ae = ae.resample(time="YS").mean()
    annual_ref = ref.resample(time="YS").mean()
    annual_error = error.resample(time="YS").mean()
    series = xr.concat([
        annual_ae.weighted(weights).mean(['lat', 'lon']).rename(model="dataset"),
        annual_ref.weighted(weights).mean(['lat', 'lon']).expand_dims(dataset=["Reference"]),
    ], dim="dataset")
    # A shared reference baseline preserves any AE mean bias in the time series.
    series = series - series.sel(dataset="Reference", time=early, drop=True).mean("time")
    annual_series.append(series.expand_dims(variable=[variable]))

    rmse_maps = xr.concat([
        np.sqrt((error.sel(time=period)**2).mean("time"))
        for period in periods.values()
    ], dim=xr.IndexVariable("period", list(periods)))
    bias_maps = xr.concat([
        annual_error.sel(time=period).mean("time")
        for period in periods.values()
    ], dim=xr.IndexVariable("period", list(periods)))

    for name, maps, cmap, title in [
        ("rmse", rmse_maps, "viridis", "Monthly reconstruction RMSE"),
        ("bias", bias_maps, "RdBu_r", "Annual-mean bias: AE minus reference"),
    ]:
        fig, axes = plt.subplots(len(models), 3, figsize=(15, 4 * len(models)), layout="constrained",
                                 squeeze=False,
                                 subplot_kw={'projection': ccrs.PlateCarree()})
        limit = max(float(abs(maps).max()), 1e-12)
        for row, model in zip(axes, models):
            for ax, label in zip(row, periods):
                p = maps.sel(model=model, period=label).plot(
                    ax=ax, transform=ccrs.PlateCarree(), cmap=cmap,
                    vmin=0 if name == "rmse" else -limit, vmax=limit, add_colorbar=False,
                )
                ax.coastlines(resolution="110m")
                ax.set_title(f"{model}: {label}")
        fig.colorbar(p, ax=axes, label=f"{variable} [{units}]", shrink=0.7)
        fig.suptitle(f"{variable}: {title}")
        fig.savefig(output_dir / f"{variable}_{name}_maps.png", dpi=150)
        plt.close(fig)

    rows = []
    for label, period in periods.items():
        # Temporal variability of annual global means, with a linear trend removed.
        # This is one reference member, so it is not an ensemble-spread estimate.
        segment = series.sel(time=period)
        trend = xr.polyval(segment.time, segment.polyfit(dim="time", deg=1).polyfit_coefficients)
        variability = (segment - trend).std("time")
        ref_variability = variability.sel(dataset="Reference", drop=True)
        rows.append(xr.Dataset({
            "rmse": np.sqrt(spatial_mse.sel(time=period).mean("time")),
            "bias": monthly.bias.sel(time=period).mean("time"),
            "variability_ratio": variability.sel(dataset=models).rename(dataset="model")
                                 / ref_variability.where(ref_variability > 0),
        }).expand_dims(period=[label]))
    period_metrics.append(xr.concat(rows, dim="period").expand_dims(variable=[variable]))

    # Late-minus-early change on the same spatial support for both datasets.
    change = xr.concat([
        (annual_ae.sel(time=late).mean("time") - annual_ae.sel(time=early).mean("time"))
        .rename(model="dataset"),
        (annual_ref.sel(time=late).mean("time") - annual_ref.sel(time=early).mean("time"))
        .expand_dims(dataset=["Reference"]),
    ], dim="dataset")
    change = change.where(np.isfinite(change).all("dataset"))
    change_error = (change.sel(dataset=models).rename(dataset="model")
                    - change.sel(dataset="Reference", drop=True))
    forced_metrics.append(xr.Dataset({
        "pattern_rmse": np.sqrt((change_error**2).weighted(weights).mean(['lat', 'lon'])),
        "ae_change": change.sel(dataset=models).rename(dataset="model").weighted(weights).mean(['lat', 'lon']),
        "reference_change": change.sel(dataset="Reference", drop=True).weighted(weights).mean(['lat', 'lon']),
    }).expand_dims(variable=[variable]))

    fig, axes = plt.subplots(len(models), 3, figsize=(15, 4 * len(models)), layout="constrained",
                             squeeze=False,
                             subplot_kw={'projection': ccrs.PlateCarree()})
    limit = max(float(abs(change).max()), float(abs(change_error).max()), 1e-12)
    for row, model in zip(axes, models):
        for ax, field, title in zip(
            row, [change.sel(dataset=model), change.sel(dataset="Reference"), change_error.sel(model=model)],
            [model, "Reference", f"{model} minus reference"],
        ):
            p = field.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="RdBu_r",
                           vmin=-limit, vmax=limit, add_colorbar=False)
            ax.coastlines(resolution="110m")
            ax.set_title(title)
    fig.colorbar(p, ax=axes, label=f"{variable} [{units}]", shrink=0.7)
    fig.suptitle(f"{variable}: annual-mean change, 2070-2100 minus 1850-1880")
    fig.savefig(output_dir / f"{variable}_forced_change_maps.png", dpi=150)
    plt.close(fig)

monthly_metrics = xr.concat(monthly_metrics, dim="variable")
period_metrics = xr.concat(period_metrics, dim="variable")
forced_metrics = xr.concat(forced_metrics, dim="variable")
annual_series = xr.concat(annual_series, dim="variable")
forced_metrics["pattern_nrmse"] = forced_metrics.pattern_rmse / scales.sel(variable=variables)

# Save the plotted values too, without writing another full trajectory.
monthly_metrics.to_dataframe().to_csv(output_dir / "monthly_metrics.csv")
period_metrics.to_dataframe().to_csv(output_dir / "period_metrics.csv")
forced_metrics.to_dataframe().to_csv(output_dir / "forced_change_metrics.csv")
annual_series.rename("anomaly").to_dataframe().to_csv(output_dir / "annual_global_anomalies.csv")

# Separate panels keep each variable's physical units readable.
ncols = min(4, len(variables))
nrows = int(np.ceil(len(variables) / ncols))
for metric in ["rmse", "bias"]:
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             squeeze=False, layout="constrained")
    for ax, variable in zip(axes.flat, variables):
        values = period_metrics[metric].sel(variable=variable)
        x = np.arange(len(periods))
        for i, (model, color) in enumerate(zip(models, colors)):
            ax.bar(x + (i - 0.5) * 0.35, values.sel(model=model).values,
                   width=0.35, color=color, label=model)
        ax.set_xticks(x, list(periods))
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(variable)
        ax.set_ylabel(reference[variable].attrs.get("units", ""))
        ax.tick_params(axis="x", labelrotation=30)
    for ax in list(axes.flat)[len(variables):]:
        ax.set_visible(False)
    axes.flat[0].legend()
    fig.suptitle(f"{component}: global reconstruction {metric.upper()}")
    fig.savefig(output_dir / f"period_{metric}_bars.png", dpi=150)
    plt.close(fig)

fig, ax = plt.subplots(figsize=(max(10, len(variables)), 4), layout="constrained")
x = np.arange(len(variables))
for i, (model, color) in enumerate(zip(models, colors)):
    ax.bar(x + (i - 0.5) * 0.35, forced_metrics.pattern_nrmse.sel(model=model).values,
           width=0.35, color=color, label=model)
ax.set_xticks(x, variables)
ax.set_ylabel("Spatial RMSE of change / Heptagon training SD")
ax.set_title("Forced-change pattern error: 2070-2100 minus 1850-1880")
ax.tick_params(axis="x", labelrotation=45)
ax.legend()
fig.savefig(output_dir / "forced_pattern_nrmse_bars.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                         squeeze=False, layout="constrained")
for ax, variable in zip(axes.flat, variables):
    values = forced_metrics.sel(variable=variable)
    ax.bar(models + ["Reference"],
           [float(values.ae_change.sel(model=model)) for model in models] + [float(values.reference_change)],
           color=colors)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(variable)
    ax.set_ylabel(reference[variable].attrs.get("units", ""))
for ax in list(axes.flat)[len(variables):]:
    ax.set_visible(False)
fig.suptitle(f"{component}: global-mean change, 2070-2100 minus 1850-1880")
fig.savefig(output_dir / "global_forced_change_bars.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, len(periods), figsize=(18, 4), layout="constrained")
x = np.arange(len(variables))
for ax, label in zip(axes, periods):
    for i, (model, color) in enumerate(zip(models, colors)):
        ax.bar(x + (i - 0.5) * 0.35, period_metrics.variability_ratio.sel(period=label, model=model).values,
               width=0.35, color=color, label=model)
    ax.axhline(1, color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks(x, variables, rotation=90)
    ax.set_ylabel("AE SD / reference SD")
    ax.set_title(label)
axes[0].legend()
fig.suptitle("Detrended annual-global-mean temporal variability")
fig.savefig(output_dir / "temporal_variability_bars.png", dpi=150)
plt.close(fig)

# Trajectory postage stamps, relative to the same 1850-1880 reference mean.
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                         squeeze=False, sharex=True, layout="constrained")
for ax, variable in zip(axes.flat, variables):
    for dataset, color in zip(models + ["Reference"], colors):
        annual_series.sel(variable=variable, dataset=dataset).plot.line(
            ax=ax, x="time", label=dataset, color=color, linewidth=1,
        )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(variable)
    ax.set_xlabel("")
    ax.set_ylabel(reference[variable].attrs.get("units", ""))
for ax in list(axes.flat)[len(variables):]:
    ax.set_visible(False)
axes.flat[0].legend()
fig.suptitle(f"{component}: annual global-mean anomalies from the reference 1850-1880 mean")
fig.savefig(output_dir / "annual_trajectory_postage_stamps.png", dpi=150)
plt.close(fig)

for ds in trajectories.values():
    ds.close()
reference.close()
print(f"Saved evaluation to {output_dir}")
