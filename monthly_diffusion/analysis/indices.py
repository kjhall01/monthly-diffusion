from __future__ import annotations

import numpy as np
import xarray as xr

from .climatology import remove_monthly_climatology
from .eofs import spatial_eof
from .spatial import area_weights, weighted_mean


def regional_mean(
    x: xr.DataArray,
    *,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> xr.DataArray:
    """Area-weighted regional mean with longitude wrap support."""

    region = _select_region(
        x,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
    )
    return weighted_mean(region, dims=(lat_dim, lon_dim), lat_dim=lat_dim, lon_dim=lon_dim)


def enso_index(
    ds: xr.Dataset,
    *,
    variable: str = "sst",
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    time_dim: str = "time",
    remove_monthly: bool = True,
) -> xr.DataArray:
    """Nino 3.4 SST anomaly index."""

    index = regional_mean(
        ds[variable],
        lat_bounds=(-5.0, 5.0),
        lon_bounds=(190.0, 240.0),
        lat_dim=lat_dim,
        lon_dim=lon_dim,
    )
    if remove_monthly:
        index = remove_monthly_climatology(index, time_dim=time_dim)
    return index.rename("ENSO")


def qbo_index(
    ds: xr.Dataset,
    *,
    variable: str = "u50",
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    time_dim: str = "time",
    remove_monthly: bool = True,
) -> xr.DataArray:
    """Equatorial lower-stratospheric zonal-wind anomaly index."""

    index = regional_mean(
        ds[variable],
        lat_bounds=(-5.0, 5.0),
        lon_bounds=(0.0, 360.0),
        lat_dim=lat_dim,
        lon_dim=lon_dim,
    )
    if remove_monthly:
        index = remove_monthly_climatology(index, time_dim=time_dim)
    return index.rename("QBO")


def nao_index(
    ds: xr.Dataset,
    *,
    variable: str = "mslp",
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    time_dim: str = "time",
    remove_monthly: bool = True,
) -> xr.DataArray:
    """Simple Azores-minus-Iceland sea-level-pressure difference.

    The returned series retains the input pressure units.  Seasonal-anomaly
    removal is optional, but this diagnostic deliberately does not centre or
    standardize the resulting index: its variance is a comparison target.
    """

    azores = regional_mean(
        ds[variable],
        lat_bounds=(35.0, 45.0),
        lon_bounds=(330.0, 360.0),
        lat_dim=lat_dim,
        lon_dim=lon_dim,
    )
    iceland = regional_mean(
        ds[variable],
        lat_bounds=(60.0, 70.0),
        lon_bounds=(330.0, 350.0),
        lat_dim=lat_dim,
        lon_dim=lon_dim,
    )
    index = azores - iceland
    if remove_monthly:
        index = remove_monthly_climatology(index, time_dim=time_dim)
    return index.rename("NAO")


def pdo_index(
    ds: xr.Dataset,
    *,
    variable: str = "sst",
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    time_dim: str = "time",
    remove_monthly: bool = True,
    loading: xr.DataArray | None = None,
) -> xr.DataArray:
    """Leading-EOF projection of North Pacific SST anomalies.

    With a supplied loading, this is the area-weighted projection onto that
    fixed EOF basis and retains the input SST units.  As with NAO, optional
    seasonal-anomaly removal does not imply temporal centring or
    standardization: the physical variance is retained for comparison.
    """

    region = _select_region(
        ds[variable],
        lat_bounds=(20.0, 70.0),
        lon_bounds=(110.0, 260.0),
        lat_dim=lat_dim,
        lon_dim=lon_dim,
    )
    anomaly = region
    if remove_monthly:
        anomaly = remove_monthly_climatology(region, time_dim=time_dim)
    if loading is None:
        _, pcs = spatial_eof(
            anomaly,
            sample_dim=time_dim,
            feature_dims=(lat_dim, lon_dim),
            n_modes=1,
        )
        index = pcs.sel(mode=1, drop=True)
    else:
        anomaly, loading = xr.align(anomaly, loading, join="inner")
        weights = area_weights(anomaly, lat_dim=lat_dim, lon_dim=lon_dim)
        index = (anomaly * loading * weights).sum((lat_dim, lon_dim), skipna=True)
    return index.rename("PDO")


def pdo_loading(
    ds: xr.Dataset,
    *,
    variable: str = "sst",
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    time_dim: str = "time",
    member_dim: str = "member",
    remove_monthly: bool = True,
) -> xr.DataArray:
    """Return one pooled North-Pacific SST EOF loading for comparable PDO projections."""

    region = _select_region(
        ds[variable],
        lat_bounds=(20.0, 70.0),
        lon_bounds=(110.0, 260.0),
        lat_dim=lat_dim,
        lon_dim=lon_dim,
    )
    anomaly = remove_monthly_climatology(region, time_dim=time_dim) if remove_monthly else region
    if member_dim in anomaly.dims:
        anomaly = anomaly.stack(sample=(time_dim, member_dim)).transpose("sample", lat_dim, lon_dim)
        sample_dim = "sample"
    else:
        sample_dim = time_dim
    eofs, _ = spatial_eof(
        anomaly,
        sample_dim=sample_dim,
        feature_dims=(lat_dim, lon_dim),
        n_modes=1,
    )
    return eofs.sel(mode=1, drop=True).rename("PDO_loading")


def climate_indices(
    ds: xr.Dataset,
    *,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    time_dim: str = "time",
    variables: dict[str, str] | None = None,
    remove_monthly: bool = True,
    pdo_loading: xr.DataArray | None = None,
) -> xr.Dataset:
    """Calculate available ENSO/PDO/NAO/QBO-style indices."""

    variables = {} if variables is None else dict(variables)
    out = {}
    if variables.get("enso", "sst") in ds:
        out["ENSO"] = enso_index(ds, variable=variables.get("enso", "sst"), lat_dim=lat_dim, lon_dim=lon_dim, time_dim=time_dim, remove_monthly=remove_monthly)
    if variables.get("pdo", "sst") in ds:
        out["PDO"] = pdo_index(ds, variable=variables.get("pdo", "sst"), lat_dim=lat_dim, lon_dim=lon_dim, time_dim=time_dim, remove_monthly=remove_monthly, loading=pdo_loading)
    if variables.get("nao", "mslp") in ds:
        out["NAO"] = nao_index(ds, variable=variables.get("nao", "mslp"), lat_dim=lat_dim, lon_dim=lon_dim, time_dim=time_dim, remove_monthly=remove_monthly)
    if variables.get("qbo", "u50") in ds:
        out["QBO"] = qbo_index(ds, variable=variables.get("qbo", "u50"), lat_dim=lat_dim, lon_dim=lon_dim, time_dim=time_dim, remove_monthly=remove_monthly)
    return xr.Dataset(out)


def _select_region(
    x: xr.DataArray,
    *,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
    lat_dim: str,
    lon_dim: str,
) -> xr.DataArray:
    lat0, lat1 = sorted(lat_bounds)
    out = x.where((x[lat_dim] >= lat0) & (x[lat_dim] <= lat1), drop=True)

    lon = _lon_360(out[lon_dim])
    lon0 = lon_bounds[0] % 360.0
    lon1 = lon_bounds[1] % 360.0
    if np.isclose((lon_bounds[1] - lon_bounds[0]) % 360.0, 0.0):
        keep_lon = xr.ones_like(lon, dtype=bool)
    elif lon0 <= lon1:
        keep_lon = (lon >= lon0) & (lon <= lon1)
    else:
        keep_lon = (lon >= lon0) | (lon <= lon1)
    return out.where(keep_lon, drop=True)


def _lon_360(lon: xr.DataArray) -> xr.DataArray:
    return (lon % 360.0).astype(np.float64)
