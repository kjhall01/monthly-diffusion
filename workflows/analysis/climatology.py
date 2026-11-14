from __future__ import annotations

import numpy as np
import xarray as xr

from workflows.analysis.spatial import weighted_mean, weighted_rmse


def monthly_climatology(
    x: xr.DataArray | xr.Dataset,
    *,
    time_dim: str = "time",
) -> xr.DataArray | xr.Dataset:
    """Mean seasonal cycle grouped by calendar month."""

    return x.groupby(f"{time_dim}.month").mean(time_dim, skipna=True)


def remove_monthly_climatology(
    x: xr.DataArray | xr.Dataset,
    climatology: xr.DataArray | xr.Dataset | None = None,
    *,
    time_dim: str = "time",
) -> xr.DataArray | xr.Dataset:
    """Subtract a monthly climatology from a time series."""

    if climatology is None:
        climatology = monthly_climatology(x, time_dim=time_dim)
    return x.groupby(f"{time_dim}.month") - climatology


def climatology_rmse(
    pred: xr.DataArray | xr.Dataset,
    target: xr.DataArray | xr.Dataset,
    *,
    weights: xr.DataArray | None = None,
    spatial_dims: tuple[str, ...] = ("lat", "lon"),
    time_dim: str = "time",
    average_months: bool = True,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> xr.DataArray | xr.Dataset:
    """Area-weighted RMSE between monthly climatologies."""

    pred_clim = monthly_climatology(pred, time_dim=time_dim)
    target_clim = monthly_climatology(target, time_dim=time_dim)
    if not average_months:
        return weighted_rmse(
            pred_clim,
            target_clim,
            weights=weights,
            dims=spatial_dims,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
        )

    squared_error = (pred_clim - target_clim) ** 2
    spatial_mse = weighted_mean(
        squared_error,
        weights=weights,
        dims=spatial_dims,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
    )
    return np.sqrt(spatial_mse.mean("month", skipna=True))
