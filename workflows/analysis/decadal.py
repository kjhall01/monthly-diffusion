"""Decadal SST variability, without checkpoint or inference dependencies.

Work on annual SST in physical degrees Celsius. Ensemble departures use the
package's exact time-matched mean and finite-ensemble correction. Filtering is
along each member's time axis, NEVER along concatenated member trajectories.
The single-realization alternative removes a gridpoint linear trend instead;
it is an approximate sensitivity diagnostic, not a forced/internal separation.
"""
import logging
import numpy as np
import xarray as xr
from scipy.optimize import linear_sum_assignment
from scipy.signal import convolve, welch

from workflows.analysis.climate import annual_means, internal_variability
from workflows.analysis.eofs import spatial_eof

log = logging.getLogger(__name__)


def wet_coarsen(field, support, factor=5):
    """Average fine cells using cos(latitude), retaining their summed wet area.

    Only a fixed common ocean support enters numerator AND denominator. Missing
    values within that support invalidate the coarse cell; zeros never stand
    in for land. Requires divisible, regular, complete global lat/lon grids.
    Coarse-cell EOF weights are wet area, not merely cos(coarse latitude), so
    coastal fragments do not count as full ocean cells.
    """
    field, support = xr.align(field, support, join="exact")
    if factor < 1 or any(field.sizes[d] % factor for d in ("lat", "lon")):
        raise ValueError("Coarsening factor must divide both grid dimensions")
    for d in ("lat", "lon"):
        delta = np.diff(field[d])
        if not np.allclose(delta, delta[0]) or delta[0] <= 0:
            raise ValueError("Expected increasing regular latitude/longitude")
    weights = np.cos(np.deg2rad(support.lat)).broadcast_like(support).where(support, 0.)
    coarse = dict(lat=factor, lon=factor, boundary="exact")
    area = weights.coarsen(**coarse).sum()
    area.attrs.update(units="1", long_name="relative wet area (sum of fine-cell cosine-latitude weights)")
    numerator = (field.where(support, 0.)*weights).coarsen(**coarse).sum(skipna=False)
    out = numerator/area.where(area > 0)
    out.attrs.update(units="degC", spatial_reduction="fixed-support wet-area-weighted block mean")
    return out, area


def extract_annual_sst(archive, support, factor=5):
    """Read at most one year/one variable from a canonical or forecast archive."""
    if archive.units["sst"] not in ("degC", "degrees_Celsius", "degree_Celsius", "C", "°C"):
        raise ValueError(f"SST must already be Celsius, got {archive.units['sst']!r}")
    values, complete = [], []
    for year in np.unique(archive.time.dt.year):
        indices = np.flatnonzero(archive.time.dt.year == year)
        if len(indices) != 12:
            log.info("Skipping incomplete annual SST: %s (%d months)", year, len(indices))
            continue
        field = archive.read("sst", indices)
        # annual_means reuses the repository's equal-month convention. Reject
        # a missing month at a wet cell, rather than giving that year 11 months.
        annual = annual_means(field).where(np.isfinite(field).all("time"))
        reduced, area = wet_coarsen(annual, support, factor)
        values.append(reduced)
        complete.append(int(year))
        if len(complete) % 25 == 0:
            log.info("  SST annual reduction through %d", year)
    if not values:
        raise ValueError("No complete annual SST fields")
    sst = xr.concat(values, "year").transpose("member", "year", "lat", "lon")
    return xr.Dataset(dict(sst=sst, wet_area=area), attrs=dict(
        first_complete_year=complete[0], last_complete_year=complete[-1],
        annual_mean="equal weight of 12 complete months", coarsen_factor=factor))


def lanczos_weights(cutoff_years=10., half_width=10):
    """Normalized 21-point (default) annual Lanczos low-pass FIR coefficients.

    cutoff_years is a frequency cutoff, not a running-mean window. The nominal
    cutoff has roughly half-amplitude response; this is not a sharp spectral
    partition. Endpoints are discarded, never reflected/padded or interpolated.
    """
    if cutoff_years <= 2 or half_width < 2:
        raise ValueError("Need a cutoff above the annual Nyquist period and width >= 2")
    k = np.arange(-half_width, half_width+1, dtype=float)
    f = 1./cutoff_years
    kernel = 2*f*np.sinc(2*f*k)*np.sinc(k/half_width)
    return kernel/kernel.sum()


def lowpass_annual(field, kernel):
    """Valid FIR convolution independently for each member and grid cell."""
    field = field.transpose("member", "year", "lat", "lon")
    if np.any(np.diff(field.year) != 1):
        raise ValueError("Annual filter requires consecutive years")
    kernel = np.asarray(kernel)
    half = len(kernel)//2
    if len(kernel) % 2 != 1 or field.sizes["year"] <= len(kernel):
        raise ValueError("Need an odd filter shorter than the period")
    values = convolve(field.values, kernel[None, :, None, None], mode="valid", method="direct")
    return field.isel(year=slice(half, -half)).copy(data=values)


def variability_fields(annual, years, kernel, method="ensemble"):
    """Prepare one independently filtered interval; never borrow future edges.

    Remove each member's period mean after ensemble subtraction, but do not
    remove its linear trend. That avoids persistent member offsets entering
    the pooled covariance as apparent temporal variability. For the observation
    sensitivity, fit and remove a separate linear trend per member/gridpoint.
    """
    part = annual.sel(year=slice(*years)).transpose("member", "year", "lat", "lon")
    if not np.array_equal(part.year, np.arange(years[0], years[1]+1)):
        raise ValueError(f"Incomplete requested period {years}")
    if method == "ensemble":
        anomalies = internal_variability(part)
    elif method == "detrended":
        anomalies = part.copy()
    else:
        raise ValueError(f"Unknown anomaly method {method}")
    mean = anomalies.mean("year", skipna=False)
    anomalies = anomalies-mean
    t = part.year.astype(float)-float(part.year.mean())
    slope = (anomalies*t).sum("year", skipna=False)/(t*t).sum()
    if method == "detrended":
        anomalies = anomalies-slope*t
    low = lowpass_annual(anomalies, kernel)
    # Trimming can give each member a nonzero interior mean even after the
    # full-period mean was removed. Exclude that constant offset from the EOF
    # covariance too, so pooled EOF variance equals mean temporal variance.
    filtered_mean = low.mean("year", skipna=False)
    low = low-filtered_mean
    # Compare filtered and unfiltered variance on exactly the retained years.
    interior = anomalies.sel(year=low.year)
    annual_var = interior.var("year", ddof=0, skipna=False).mean("member", skipna=False)
    low_var = low.var("year", ddof=0, skipna=False).mean("member", skipna=False)
    result = xr.Dataset(dict(annual=anomalies, lowpass=low.rename(year="filtered_year"),
        removed_mean=mean, filtered_removed_mean=filtered_mean,
        linear_slope=slope, annual_variance=annual_var,
        lowpass_variance=low_var, lowpass_std=np.sqrt(low_var),
        lowpass_fraction=low_var/annual_var.where(annual_var > 0)), attrs=dict(
            method=method, input_first_year=years[0], input_last_year=years[1],
            retained_first_year=int(low.year[0]), retained_last_year=int(low.year[-1]),
            members=part.sizes["member"], slope_removed=int(method == "detrended"),
            finite_ensemble_factor=float(np.sqrt(part.sizes["member"]/(part.sizes["member"]-1)))
                if method == "ensemble" else 1.))
    for name in result.data_vars:
        result[name].attrs["units"] = ("degC year-1" if name == "linear_slope" else
            "1" if name == "lowpass_fraction" else "degC2" if name.endswith("variance") else "degC")
    return result


def region_weights(area, bounds):
    """Rectangular basin domains (degrees east), with the common wet mask."""
    south, north, west, east = bounds
    longitude = area.lon % 360
    longitude_mask = ((longitude >= west) & (longitude <= east)) if west < east else (
        (longitude >= west) | (longitude <= east))
    return area.where((area.lat >= south) & (area.lat <= north) & longitude_mask, 0.)


def fit_covariance_eofs(field, weights, n_modes=3):
    """Package covariance EOFs, rescaled to wet-area RMS-one spatial patterns.

    The patterns are dimensionless and PCs are degrees Celsius. A map times
    one reference PC standard deviation is the physical +1-SD loading. No
    per-gridpoint variance normalization (correlation EOF) is performed.
    """
    common = np.isfinite(field).all(("member", "year")) & (weights > 0)
    weight = weights.where(common, 0.)
    if int(common.sum()) < n_modes:
        raise ValueError("Too few common wet cells for EOF fit")
    stacked = field.where(common).stack(sample=("member", "year")).transpose("sample", "lat", "lon")
    eofs, _ = spatial_eof(stacked, sample_dim="sample", weights=weight, n_modes=n_modes)
    norm = np.sqrt((eofs**2*weight).sum(("lat", "lon"))/weight.sum())
    eofs = eofs/norm
    # A reproducible display sign, not a physical assertion about a named mode.
    for mode in eofs.mode.values:
        values = eofs.sel(mode=mode).values
        sign = np.sign(values.ravel()[np.nanargmax(abs(values))])
        eofs.loc[dict(mode=mode)] *= sign
    center = field.mean(("member", "year"), skipna=False)
    pc = project_eofs(field-center, eofs, weight)
    variance = pc.var(("member", "year"), ddof=0)
    total = ((field-center).var(("member", "year"), skipna=False)*weight).sum(("lat", "lon"))/weight.sum()
    return xr.Dataset(dict(pattern=eofs, center=center, weight=weight,
        pc_std=np.sqrt(variance), loading=eofs*np.sqrt(variance),
        explained_fraction=variance/total, total_variance=total), attrs=dict(
            method="wet-area weighted covariance EOF, pooled member-years",
            pattern_units="1 (wet-area RMS=1)", pc_units="degC",
            loading_units="degC per own PC standard deviation"))


def project_eofs(field, patterns, weights):
    """Project onto a fixed complete basis/support; preserve physical amplitude."""
    field, patterns, weights = xr.align(field, patterns, weights, join="exact")
    use = weights > 0
    if bool((~np.isfinite(field).all(("member", "year")) & use).any()):
        raise ValueError("Projection has missing values on fitted EOF support")
    return ((field.where(use, 0.)*patterns.fillna(0.)*weights).sum(("lat", "lon"))/weights.sum()).transpose(
        "member", "year", "mode")


def match_eofs(reference, other):
    """One-to-one maximum absolute weighted cosine matching of retained modes.

    Allows rank changes/sign flips, not a rotation of the independent EOFs.
    Near-degenerate modes may still mix: matched correlations are descriptive.
    """
    w = reference.weight
    a, b = reference.pattern.values.reshape(reference.sizes["mode"], -1), other.pattern.values.reshape(other.sizes["mode"], -1)
    weight = w.values.ravel()/float(w.sum())
    a, b = np.nan_to_num(a), np.nan_to_num(b)
    correlations = (a*weight)@b.T/np.sqrt(((a*a)@weight)[:, None]*((b*b)@weight)[None, :])
    rows, columns = linear_sum_assignment(-abs(correlations))
    columns = columns[np.argsort(rows)]
    signed = correlations[np.arange(len(columns)), columns]
    signs = np.where(signed < 0, -1., 1.)
    return columns, signs, abs(signed)


def pc_diagnostics(fields, basis, nperseg=64):
    """Common-reference PCs and spectra, without independently standardizing.

    Welch PSD is computed per member on unfiltered annual PCs, with only a
    constant detrend per segment. It illustrates how much power sits above
    10 years; it is not a claim of a resolved oscillatory mode or significance.
    """
    annual = project_eofs(fields.annual-basis.center, basis.pattern, basis.weight)
    low = project_eofs(fields.lowpass.rename(filtered_year="year")-basis.center, basis.pattern, basis.weight)
    frequency, psd = welch(annual.values, fs=1., axis=1,
        nperseg=min(nperseg, annual.sizes["year"]), detrend="constant", scaling="density")
    low_var = low.var("year", skipna=False)
    raw_var = annual.sel(year=low.year).var("year", skipna=False)
    result = xr.Dataset(dict(annual_pc=annual, lowpass_pc=low.rename(year="filtered_year"),
        psd=(("member", "frequency", "mode"), psd),
        lowpass_variance=low_var, annual_variance=raw_var,
        lowpass_fraction=low_var/raw_var.where(raw_var > 0)),
        coords=dict(frequency=frequency), attrs=dict(
            pc_units="degC in fixed historical reference EOF basis",
            psd_units="degC^2 year", psd_input="unfiltered annual common-basis PCs",
            filter_note="lowpass and annual variance use identical interior years"))
    for name in result.data_vars:
        result[name].attrs["units"] = ("degC2 year" if name == "psd" else
            "1" if name == "lowpass_fraction" else "degC2" if name.endswith("variance") else "degC")
    result.frequency.attrs["units"] = "year-1"
    return result
