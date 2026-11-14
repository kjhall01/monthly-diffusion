from __future__ import annotations

import numpy as np
import xarray as xr

from workflows.analysis.helpers import apply_binary, apply_unary, require_dims


def area_weights(
    x: xr.DataArray | xr.Dataset,
    *,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    normalize: bool = True,
) -> xr.DataArray:
    """Cos-latitude area weights on the grid of ``x``."""

    require_dims(x, (lat_dim, lon_dim))
    if lat_dim not in x.coords:
        raise ValueError(f"Latitude dimension {lat_dim!r} must have coordinate values.")

    lat = x[lat_dim]
    if lat.ndim != 1:
        raise ValueError("Latitude coordinate must be one-dimensional.")

    values = np.cos(np.deg2rad(np.asarray(lat.values, dtype=np.float64))).clip(min=0.0)
    weights = xr.DataArray(values, dims=(lat_dim,), coords={lat_dim: lat})
    weights = weights * xr.ones_like(x[lon_dim], dtype=np.float64)
    weights = weights.transpose(lat_dim, lon_dim)

    if normalize:
        weights = weights / weights.sum((lat_dim, lon_dim), skipna=True)
    return weights.rename("area_weight")


def weighted_mean(
    x: xr.DataArray | xr.Dataset,
    *,
    weights: xr.DataArray | None = None,
    dims: tuple[str, ...] = ("lat", "lon"),
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> xr.DataArray | xr.Dataset:
    """Weighted mean with finite-value masking."""

    return apply_unary(
        x,
        _weighted_mean_array,
        weights=weights,
        dims=dims,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
    )


def weighted_rmse(
    pred: xr.DataArray | xr.Dataset,
    target: xr.DataArray | xr.Dataset,
    *,
    weights: xr.DataArray | None = None,
    dims: tuple[str, ...] = ("lat", "lon"),
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> xr.DataArray | xr.Dataset:
    """Weighted RMSE over ``dims``."""

    return apply_binary(
        pred,
        target,
        _weighted_rmse_array,
        weights=weights,
        dims=dims,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
    )


def weighted_pattern_correlation(
    x: xr.DataArray | xr.Dataset,
    y: xr.DataArray | xr.Dataset,
    *,
    weights: xr.DataArray | None = None,
    dims: tuple[str, ...] = ("lat", "lon"),
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> xr.DataArray | xr.Dataset:
    """Weighted Pearson correlation over spatial dimensions."""

    return apply_binary(
        x,
        y,
        _weighted_pattern_correlation_array,
        weights=weights,
        dims=dims,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
    )


def weighted_uncentered_pattern_correlation(
    x: xr.DataArray | xr.Dataset,
    y: xr.DataArray | xr.Dataset,
    *,
    weights: xr.DataArray | None = None,
    dims: tuple[str, ...] = ("lat", "lon"),
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> xr.DataArray | xr.Dataset:
    """Cosine-latitude-weighted uncentered spatial correlation.

    This is the cosine similarity of two maps on their common finite support.
    Unlike :func:`weighted_pattern_correlation`, neither map has its spatial
    mean removed.  It is therefore appropriate when the sign and magnitude of
    the spatially uniform component are part of the response being evaluated.
    """

    return apply_binary(
        x,
        y,
        _weighted_uncentered_pattern_correlation_array,
        weights=weights,
        dims=dims,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
    )


def reference_normalized_rmse(
    pred: xr.DataArray | xr.Dataset,
    reference: xr.DataArray | xr.Dataset,
    *,
    weights: xr.DataArray | None = None,
    dims: tuple[str, ...] = ("lat", "lon"),
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> xr.DataArray | xr.Dataset:
    """Weighted RMSE divided by the weighted RMS of the reference map.

    Both terms use the exact same common finite support.  The result is
    dimensionless and is undefined when the reference map has zero RMS.
    """

    return apply_binary(
        pred,
        reference,
        _reference_normalized_rmse_array,
        weights=weights,
        dims=dims,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
    )


def weighted_projection_ratio(
    pred: xr.DataArray | xr.Dataset,
    target: xr.DataArray | xr.Dataset,
    *,
    weights: xr.DataArray | None = None,
    dims: tuple[str, ...] = ("lat", "lon"),
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> xr.DataArray | xr.Dataset:
    """Weighted projection of ``pred`` onto ``target``."""

    return apply_binary(
        pred,
        target,
        _weighted_projection_ratio_array,
        weights=weights,
        dims=dims,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
    )


def _weighted_mean_array(
    x: xr.DataArray,
    *,
    weights: xr.DataArray | None,
    dims: tuple[str, ...],
    lat_dim: str,
    lon_dim: str,
) -> xr.DataArray:
    if weights is None:
        weights = area_weights(x, lat_dim=lat_dim, lon_dim=lon_dim)
    dims = tuple(dim for dim in dims if dim in x.dims)
    valid = x.notnull()
    w = weights.broadcast_like(x).where(valid)
    numerator = (x.where(valid) * w).sum(dims, skipna=True)
    denominator = w.sum(dims, skipna=True)
    return numerator / denominator.where(denominator > 0)


def _weighted_rmse_array(
    pred: xr.DataArray,
    target: xr.DataArray,
    *,
    weights: xr.DataArray | None,
    dims: tuple[str, ...],
    lat_dim: str,
    lon_dim: str,
) -> xr.DataArray:
    pred, target = xr.align(pred, target, join="inner")
    mse = _weighted_mean_array(
        (pred - target) ** 2,
        weights=weights,
        dims=dims,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
    )
    return np.sqrt(mse)


def _weighted_pattern_correlation_array(
    x: xr.DataArray,
    y: xr.DataArray,
    *,
    weights: xr.DataArray | None,
    dims: tuple[str, ...],
    lat_dim: str,
    lon_dim: str,
) -> xr.DataArray:
    x, y = xr.align(x, y, join="inner")
    if weights is None:
        weights = area_weights(x, lat_dim=lat_dim, lon_dim=lon_dim)
    dims = tuple(dim for dim in dims if dim in x.dims)
    valid = x.notnull() & y.notnull()
    w = weights.broadcast_like(x).where(valid)
    wsum = w.sum(dims, skipna=True)

    x_mean = (x.where(valid) * w).sum(dims, skipna=True) / wsum
    y_mean = (y.where(valid) * w).sum(dims, skipna=True) / wsum
    x_centered = x - x_mean
    y_centered = y - y_mean

    numerator = (w * x_centered * y_centered).sum(dims, skipna=True)
    x_ss = (w * x_centered**2).sum(dims, skipna=True)
    y_ss = (w * y_centered**2).sum(dims, skipna=True)
    denominator = np.sqrt(x_ss * y_ss)
    return numerator / denominator.where(denominator > 0)


def _weighted_uncentered_pattern_correlation_array(
    x: xr.DataArray,
    y: xr.DataArray,
    *,
    weights: xr.DataArray | None,
    dims: tuple[str, ...],
    lat_dim: str,
    lon_dim: str,
) -> xr.DataArray:
    x, y = xr.align(x, y, join="inner")
    if weights is None:
        weights = area_weights(x, lat_dim=lat_dim, lon_dim=lon_dim)
    dims = tuple(dim for dim in dims if dim in x.dims)
    valid = x.notnull() & y.notnull()
    w = weights.broadcast_like(x).where(valid)
    numerator = (w * x * y).sum(dims, skipna=True)
    x_ss = (w * x**2).sum(dims, skipna=True)
    y_ss = (w * y**2).sum(dims, skipna=True)
    denominator = np.sqrt(x_ss * y_ss)
    return numerator / denominator.where(denominator > 0)


def _reference_normalized_rmse_array(
    pred: xr.DataArray,
    reference: xr.DataArray,
    *,
    weights: xr.DataArray | None,
    dims: tuple[str, ...],
    lat_dim: str,
    lon_dim: str,
) -> xr.DataArray:
    pred, reference = xr.align(pred, reference, join="inner")
    if weights is None:
        weights = area_weights(reference, lat_dim=lat_dim, lon_dim=lon_dim)
    dims = tuple(dim for dim in dims if dim in reference.dims)
    valid = pred.notnull() & reference.notnull()
    w = weights.broadcast_like(reference).where(valid)
    wsum = w.sum(dims, skipna=True)
    mse = (w * (pred - reference) ** 2).sum(dims, skipna=True) / wsum.where(
        wsum > 0
    )
    reference_mean_square = (w * reference**2).sum(
        dims, skipna=True
    ) / wsum.where(wsum > 0)
    return np.sqrt(mse) / np.sqrt(reference_mean_square).where(
        reference_mean_square > 0
    )


def _weighted_projection_ratio_array(
    pred: xr.DataArray,
    target: xr.DataArray,
    *,
    weights: xr.DataArray | None,
    dims: tuple[str, ...],
    lat_dim: str,
    lon_dim: str,
) -> xr.DataArray:
    pred, target = xr.align(pred, target, join="inner")
    if weights is None:
        weights = area_weights(target, lat_dim=lat_dim, lon_dim=lon_dim)
    dims = tuple(dim for dim in dims if dim in target.dims)
    valid = pred.notnull() & target.notnull()
    w = weights.broadcast_like(target).where(valid)
    numerator = (w * pred * target).sum(dims, skipna=True)
    denominator = (w * target**2).sum(dims, skipna=True)
    return numerator / denominator.where(denominator > 0)
