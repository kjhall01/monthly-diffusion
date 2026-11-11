"""Transparent climate diagnostics in physical units; no checkpoint dependencies."""
import numpy as np
import xarray as xr
from .indices import climate_indices, pdo_loading
from .spectra import lag_autocorrelation, temporal_spectrum
from .spatial import weighted_mean


def internal_variability(fields, member="member"):
    """Subtract the EXACT time-matched ensemble mean, then sqrt(N/(N-1)).

    No additional monthly-climatology removal. Require matching member support
    at a cell/month rather than silently changing N on missing members.
    """
    if member not in fields.dims:
        raise ValueError("Internal variability requires an explicit ensemble member dimension")
    n = fields.sizes[member]
    if n < 2:
        raise ValueError("Internal variability needs at least two ensemble members")
    return np.sqrt(n/(n-1))*(fields-fields.mean(member, skipna=False))


def annual_means(fields):
    """Equal-month annual means, consistent with the long-rollout response maps."""
    year = fields.time.dt.year
    years, counts = np.unique(year.values, return_counts=True)
    complete = years[counts == 12]
    return fields.where(year.isin(complete), drop=True).groupby("time.year").mean("time")


def forced_response(fields, early=(1850, 1880), late=(2070, 2100)):
    """Ensemble-mean late-minus-early annual climatology (inclusive endpoints)."""
    annual = annual_means(fields)
    if "member" in annual.dims:
        annual = annual.mean("member", skipna=False)
    for lo, hi in (early, late):
        if not np.isin(np.arange(lo, hi+1), annual.year.values).all():
            raise ValueError(f"Missing complete years in {lo}--{hi}")
    return annual.sel(year=slice(*late)).mean("year")-annual.sel(year=slice(*early)).mean("year")


def seasonal_means(fields, season="DJF"):
    """Day-weighted complete seasons, DJF labeled by January's year."""
    months = dict(DJF=(12, 1, 2), MAM=(3, 4, 5), JJA=(6, 7, 8), SON=(9, 10, 11))[season]
    selected = fields.where(fields.time.dt.month.isin(months), drop=True)
    label = selected.time.dt.year+(selected.time.dt.month == 12 if season == "DJF" else 0)
    selected = selected.assign_coords(season_year=("time", label.values))
    weights = selected.time.dt.days_in_month.astype(float)
    counts = selected.time.groupby(selected.season_year).count()
    numerator = (selected*weights).groupby("season_year").sum("time", skipna=False)
    denominator = weights.groupby(selected.season_year).sum("time")
    return (numerator/denominator).where(counts == 3, drop=True)


def teleconnection(fields, index, sample_dims=("member", "time")):
    """Native-unit regression coefficient per +1 index SD; independently standardize.

    Center over pooled samples to fit an intercept; this does not remove a
    second seasonal cycle. Fields should already be internal anomalies.
    """
    fields, index = xr.align(fields, index, join="exact")
    sample_dims = tuple(d for d in sample_dims if d in index.dims)
    z = (index-index.mean(sample_dims))/index.std(sample_dims, ddof=0)
    valid = np.isfinite(fields) & np.isfinite(z)
    x, y = z.where(valid), fields.where(valid)
    covariance = ((x-x.mean(sample_dims))*(y-y.mean(sample_dims))).mean(sample_dims)
    return covariance/x.var(sample_dims, ddof=0)


def climate_diagnostics(fields, *, pdo_basis=None, early=(1850, 1880), late=(2070, 2100)):
    """Return named xarray products; pass a REFERENCE PDO basis for comparisons.

    PDO basis fitting is explicit to avoid inadvertently using a different EOF
    for every model. No physical regridding or unit conversion is performed.
    """
    residuals = internal_variability(fields)
    if "sst" in fields and pdo_basis is None:
        raise ValueError("Supply a fixed reference PDO loading from pdo_loading(reference_internal, remove_monthly=False)")
    indices = climate_indices(residuals, remove_monthly=False, pdo_loading=pdo_basis)
    maps = xr.concat([teleconnection(residuals, index) for index in indices.data_vars.values()],
                     dim=xr.IndexVariable("mode", list(indices.data_vars)))
    return dict(indices=indices, teleconnections=maps,
                acf=lag_autocorrelation(indices), psd=temporal_spectrum(indices),
                forced_response=forced_response(fields, early, late),
                global_means=weighted_mean(fields))
