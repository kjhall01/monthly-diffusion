from __future__ import annotations

import numpy as np
import xarray as xr


def spatial_eof(
    x: xr.DataArray,
    *,
    sample_dim: str = "time",
    feature_dims: tuple[str, str] = ("lat", "lon"),
    n_modes: int | None = None,
    mode_dim: str = "mode",
    center: bool = True,
    area_weight: bool = True,
    weights: xr.DataArray | np.ndarray | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Calculate spatial EOFs over two feature dimensions.

    All dimensions other than ``sample_dim`` and ``feature_dims`` are treated as
    independent slices. NaN feature points are excluded per slice when any
    sample is non-finite at that point. By default, EOFs are area-weighted with
    ``cos(lat)`` weights inferred from the latitude feature coordinate.
    """

    _check_spatial_eof_inputs(x, sample_dim, feature_dims, mode_dim, n_modes)

    extra_dims = tuple(dim for dim in x.dims if dim not in (sample_dim, *feature_dims))
    sample_size = int(x.sizes[sample_dim])
    feature_shape = tuple(int(x.sizes[dim]) for dim in feature_dims)
    n_features = int(np.prod(feature_shape))
    max_modes = min(sample_size, n_features) if n_modes is None else int(n_modes)

    ordered = x.transpose(sample_dim, *extra_dims, *feature_dims)
    values = np.asarray(ordered.values, dtype=np.float64)
    extra_shape = tuple(int(x.sizes[dim]) for dim in extra_dims)
    n_extra = int(np.prod(extra_shape)) if extra_shape else 1
    values = values.reshape(sample_size, n_extra, n_features)
    feature_weights = _feature_weights(
        x,
        ordered,
        sample_dim=sample_dim,
        feature_dims=feature_dims,
        extra_dims=extra_dims,
        extra_shape=extra_shape,
        feature_shape=feature_shape,
        area_weight=area_weight,
        weights=weights,
    )

    eofs_flat = np.full((max_modes, n_extra, n_features), np.nan, dtype=np.float64)
    pcs_values = np.full((sample_size, max_modes, n_extra), np.nan, dtype=np.float64)

    for extra_ndx in range(n_extra):
        slice_values = values[:, extra_ndx, :]
        valid_features = np.isfinite(slice_values).all(axis=0)
        slice_weights = None if feature_weights is None else feature_weights[extra_ndx]
        if slice_weights is not None:
            valid_features = valid_features & np.isfinite(slice_weights) & (slice_weights > 0.0)
        if not np.any(valid_features):
            continue

        if slice_weights is not None:
            mean_weight = np.nanmean(slice_weights[valid_features])
            if not np.isfinite(mean_weight) or mean_weight <= 0.0:
                continue
            slice_weights = slice_weights / mean_weight

        matrix = slice_values[:, valid_features]
        if center:
            matrix = matrix - matrix.mean(axis=0, keepdims=True)
        if slice_weights is not None:
            sqrt_weights = np.sqrt(slice_weights[valid_features])
            matrix = matrix * sqrt_weights[None, :]

        slice_modes = min(max_modes, matrix.shape[0], matrix.shape[1])
        if slice_modes == 0:
            continue

        u, s, vt = np.linalg.svd(matrix, full_matrices=False)
        if slice_weights is not None:
            vt = vt / sqrt_weights[None, :]
        eofs_flat[:slice_modes, extra_ndx, valid_features] = vt[:slice_modes]
        pcs_values[:, :slice_modes, extra_ndx] = u[:, :slice_modes] * s[:slice_modes]

    mode_coord = np.arange(1, max_modes + 1, dtype=np.int64)
    eofs = _build_eof_array(
        x,
        eofs_flat.reshape((max_modes, *extra_shape, *feature_shape)),
        sample_dim=sample_dim,
        feature_dims=feature_dims,
        extra_dims=extra_dims,
        mode_dim=mode_dim,
        mode_coord=mode_coord,
        center=center,
        area_weighted=feature_weights is not None,
    )
    pcs = _build_pc_array(
        x,
        pcs_values.reshape((sample_size, max_modes, *extra_shape)),
        sample_dim=sample_dim,
        extra_dims=extra_dims,
        mode_dim=mode_dim,
        mode_coord=mode_coord,
        center=center,
        area_weighted=feature_weights is not None,
    )
    return eofs, pcs


def _feature_weights(
    x: xr.DataArray,
    ordered: xr.DataArray,
    *,
    sample_dim: str,
    feature_dims: tuple[str, str],
    extra_dims: tuple[str, ...],
    extra_shape: tuple[int, ...],
    feature_shape: tuple[int, ...],
    area_weight: bool,
    weights: xr.DataArray | np.ndarray | None,
) -> np.ndarray | None:
    if weights is None:
        if not area_weight:
            return None
        weights = _infer_cos_lat_weights(x, feature_dims)

    template = ordered.isel({sample_dim: 0}, drop=True)
    if isinstance(weights, xr.DataArray):
        unsupported_dims = set(weights.dims).difference((*extra_dims, *feature_dims))
        if unsupported_dims:
            raise ValueError(f"weights have unsupported dimensions: {sorted(unsupported_dims)}")
        weight_values = weights.broadcast_like(template).transpose(*extra_dims, *feature_dims).values
    else:
        weight_values = np.asarray(weights, dtype=np.float64)
        expected_feature_shape = feature_shape
        expected_full_shape = (*extra_shape, *feature_shape)
        if weight_values.shape == expected_feature_shape:
            weight_values = np.broadcast_to(weight_values, expected_full_shape)
        elif weight_values.shape != expected_full_shape:
            raise ValueError(
                "weights must have shape matching feature_dims or extra_dims + feature_dims; "
                f"got {weight_values.shape}, expected {expected_feature_shape} or {expected_full_shape}."
            )

    weight_values = np.asarray(weight_values, dtype=np.float64)
    weight_values = np.where(weight_values > 1.0e-12, weight_values, np.nan)
    return weight_values.reshape((int(np.prod(extra_shape)) if extra_shape else 1, int(np.prod(feature_shape))))


def _infer_cos_lat_weights(x: xr.DataArray, feature_dims: tuple[str, str]) -> xr.DataArray:
    lat_dim = next((dim for dim in feature_dims if "lat" in dim.lower()), None)
    if lat_dim is None:
        raise ValueError("area_weight=True requires a latitude feature dimension or explicit weights.")
    if lat_dim not in x.coords:
        raise ValueError(f"area_weight=True requires coordinate values for latitude dimension {lat_dim!r}.")

    lat = x[lat_dim]
    if lat.ndim != 1:
        raise ValueError("area_weight=True requires a one-dimensional latitude coordinate.")
    values = np.cos(np.deg2rad(np.asarray(lat.values, dtype=np.float64)))
    values = np.where(values > 1.0e-12, values, np.nan)
    return xr.DataArray(values, dims=(lat_dim,), coords={lat_dim: lat})


def _check_spatial_eof_inputs(
    x: xr.DataArray,
    sample_dim: str,
    feature_dims: tuple[str, str],
    mode_dim: str,
    n_modes: int | None,
) -> None:
    missing = [dim for dim in (sample_dim, *feature_dims) if dim not in x.dims]
    if missing:
        raise ValueError(f"DataArray is missing required dimensions: {missing}")
    if len(feature_dims) != 2:
        raise ValueError("feature_dims must contain exactly two dimensions")
    if len(set((sample_dim, *feature_dims))) != 3:
        raise ValueError("sample_dim and feature_dims must be distinct")
    if mode_dim in x.dims:
        raise ValueError(f"mode_dim {mode_dim!r} already exists in input dimensions")
    if n_modes is not None and int(n_modes) <= 0:
        raise ValueError("n_modes must be positive")


def _dim_coords(x: xr.DataArray, dims: tuple[str, ...]) -> dict:
    return {dim: x.coords[dim] for dim in dims if dim in x.coords}


def _build_eof_array(
    original: xr.DataArray,
    values: np.ndarray,
    *,
    sample_dim: str,
    feature_dims: tuple[str, str],
    extra_dims: tuple[str, ...],
    mode_dim: str,
    mode_coord: np.ndarray,
    center: bool,
    area_weighted: bool,
) -> xr.DataArray:
    dims = (mode_dim, *extra_dims, *feature_dims)
    coords = {
        mode_dim: mode_coord,
        **_dim_coords(original, (*extra_dims, *feature_dims)),
    }
    out = xr.DataArray(
        values,
        dims=dims,
        coords=coords,
        name=f"{original.name}_eofs" if original.name is not None else "eofs",
    )
    eof_dims = tuple(mode_dim if dim == sample_dim else dim for dim in original.dims)
    out = out.transpose(*eof_dims)
    out.attrs["sample_dim"] = sample_dim
    out.attrs["feature_dims"] = tuple(feature_dims)
    out.attrs["centered"] = bool(center)
    out.attrs["area_weighted"] = bool(area_weighted)
    return out


def _build_pc_array(
    original: xr.DataArray,
    values: np.ndarray,
    *,
    sample_dim: str,
    extra_dims: tuple[str, ...],
    mode_dim: str,
    mode_coord: np.ndarray,
    center: bool,
    area_weighted: bool,
) -> xr.DataArray:
    dims = (sample_dim, mode_dim, *extra_dims)
    coords = {
        mode_dim: mode_coord,
        **_dim_coords(original, (sample_dim, *extra_dims)),
    }
    out = xr.DataArray(
        values,
        dims=dims,
        coords=coords,
        name=f"{original.name}_pcs" if original.name is not None else "pcs",
    )
    out.attrs["sample_dim"] = sample_dim
    out.attrs["centered"] = bool(center)
    out.attrs["area_weighted"] = bool(area_weighted)
    return out
