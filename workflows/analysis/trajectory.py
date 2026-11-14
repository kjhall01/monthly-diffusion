"""Physical comparisons for reconstructions and free-running monthly trajectories."""

import numpy as np
import xarray as xr
from workflows.analysis.spatial import weighted_mean
from workflows.data.dataloader import validate_months, normalized_unit_name


def complete_period(field, years):
    """Select inclusive whole years, refusing silently shortened climatologies."""
    lo, hi = years
    if hi < lo:
        raise ValueError("Period end precedes its start")
    selected = field.sel(time=slice(str(lo), str(hi)))
    keys = selected.time.dt.year.values * 12 + selected.time.dt.month.values
    if not np.array_equal(keys, np.arange(lo * 12 + 1, (hi + 1) * 12 + 1)):
        raise ValueError(f"Missing complete years in {lo}--{hi}")
    return selected


def monthly_anomalies(field, baseline):
    """Remove each trajectory's own baseline monthly means; retain its trend."""
    climatology = complete_period(field, baseline).groupby("time.month").mean("time")
    return (field.groupby("time.month") - climatology).reset_coords(drop=True)


def ensemble_mean(field):
    return field.mean("member", skipna=False) if "member" in field.dims else field


def _divide(total, count):
    """A missing mean stays NaN, without warnings for unobserved cells."""
    result = np.full(total.shape, np.nan, dtype=np.float64)
    return np.divide(total, count, out=result, where=count > 0)


def _year_slices(field):
    years = field.time.dt.year.values
    for year in np.unique(years):
        indices = np.flatnonzero(years == year)
        yield int(year), slice(indices[0], indices[-1] + 1)


def _read_year(field, selection):
    # Slice BEFORE computing. Dask only reads native storage chunks for this
    # year; all subsequent arithmetic is on these small, in-memory arrays.
    return np.asarray(field.isel(time=selection).compute().values, dtype=np.float64)


def _map(field, values):
    dims = field.dims[1:]  # time is first; member, if present, is second
    return xr.DataArray(values, dims=dims, coords={d: field[d] for d in dims})


def _global_mean(values, weights):
    valid = ~np.isnan(values)
    total = (np.where(valid, values, 0.0) * weights).sum(axis=(-2, -1))
    count = (valid * weights).sum(axis=(-2, -1))
    return _divide(total, count)


def _baseline_statistics(reference, generated, baseline, reconstruction):
    """First pass: monthly sums/counts and the fixed AR comparison support."""
    ref = complete_period(reference, baseline)
    gen = complete_period(generated, baseline)
    ref_sum = np.zeros((12, *ref.shape[1:]), dtype=np.float64)
    gen_sum = np.zeros((12, *gen.shape[1:]), dtype=np.float64)
    ref_count = np.zeros(ref_sum.shape, dtype=np.int64)
    gen_count = np.zeros(gen_sum.shape, dtype=np.int64)
    support = np.ones(ref.shape[-2:], dtype=bool)
    # Both baseline selections have exactly January--December in every year.
    # Their timestamp conventions and member counts can differ for AR data.
    for (_, ref_slice), (_, gen_slice) in zip(_year_slices(ref), _year_slices(gen)):
        r, g = _read_year(ref, ref_slice), _read_year(gen, gen_slice)
        if reconstruction:
            valid = np.isfinite(r) & np.isfinite(g)
            r, g = np.where(valid, r, np.nan), np.where(valid, g, np.nan)
        else:
            support &= np.isfinite(r).all(axis=tuple(range(r.ndim - 2)))
            support &= np.isfinite(g).all(axis=tuple(range(g.ndim - 2)))
        for values, total, count in ((r, ref_sum, ref_count), (g, gen_sum, gen_count)):
            valid = ~np.isnan(values)
            total += np.where(valid, values, 0.0)
            count += valid
    ref_monthly = _divide(ref_sum, ref_count)
    gen_monthly = _divide(gen_sum, gen_count)
    # Weight the annual climatology by observed months, not equally by the
    # twelve monthly means: those can have different counts in reconstructions.
    ref_clim = _divide(ref_sum.sum(axis=0), ref_count.sum(axis=0))
    gen_clim = _divide(gen_sum.sum(axis=0), gen_count.sum(axis=0))
    if not reconstruction:
        ref_monthly = np.where(support, ref_monthly, np.nan)
        gen_monthly = np.where(support, gen_monthly, np.nan)
        ref_clim = np.where(support, ref_clim, np.nan)
        gen_clim = np.where(support, gen_clim, np.nan)
    return ref_monthly, gen_monthly, ref_clim, gen_clim, support


class _TrajectoryStatistics:
    """Small running totals for one trajectory; never retain its full fields."""

    def __init__(self, field, monthly_climatology, baseline, early, late, weights):
        self.field = field
        self.monthly_climatology = monthly_climatology
        self.baseline, self.early, self.late = baseline, early, late
        self.weights = weights
        self.months = field.time.dt.month.values - 1
        self.global_mean = np.full(field.shape[:-2], np.nan)
        self.global_anomaly = np.full(field.shape[:-2], np.nan)
        self.anomaly_sum = np.zeros(field.shape[1:])
        self.anomaly_square_sum = np.zeros(field.shape[1:])
        self.anomaly_count = np.zeros(field.shape[1:], dtype=np.int64)
        self.early_sum = np.zeros(field.shape[-2:])
        self.late_sum = np.zeros(field.shape[-2:])
        self.early_count = np.zeros(field.shape[-2:], dtype=np.int64)
        self.late_count = np.zeros(field.shape[-2:], dtype=np.int64)

    def add_year(self, values, year, selection):
        self.global_mean[selection] = _global_mean(values, self.weights)
        anomaly = values - self.monthly_climatology[self.months[selection]]
        self.global_anomaly[selection] = _global_mean(anomaly, self.weights)
        if self.baseline[0] <= year <= self.baseline[1]:
            valid = ~np.isnan(anomaly)
            centered = np.where(valid, anomaly, 0.0)
            self.anomaly_sum += centered.sum(axis=0)
            self.anomaly_square_sum += (centered**2).sum(axis=0)
            self.anomaly_count += valid.sum(axis=0)
        if self.early is None or len(values) != 12:
            return
        # Preserve the original order: mean of valid months within each year,
        # then mean across members (requiring all), then equal-weight years.
        valid = ~np.isnan(values)
        annual = _divide(np.where(valid, values, 0.0).sum(axis=0), valid.sum(axis=0))
        if "member" in self.field.dims:
            annual = annual.mean(axis=0)
        for period, total, count in (
            (self.early, self.early_sum, self.early_count),
            (self.late, self.late_sum, self.late_count),
        ):
            if period[0] <= year <= period[1]:
                valid = ~np.isnan(annual)
                total += np.where(valid, annual, 0.0)
                count += valid

    def series(self, label):
        dims = self.field.dims[:-2]
        coords = {d: self.field[d] for d in dims}
        return xr.Dataset(
            {
                f"{label}_global_mean": xr.DataArray(
                    self.global_mean, dims=dims, coords=coords
                ),
                f"{label}_global_anomaly": xr.DataArray(
                    self.global_anomaly, dims=dims, coords=coords
                ),
            }
        )

    def anomaly_sd(self):
        mean = _divide(self.anomaly_sum, self.anomaly_count)
        variance = _divide(self.anomaly_square_sum, self.anomaly_count) - mean**2
        # These sums use already-centered anomalies, avoiding subtraction of
        # large physical climatologies. Clip only negative roundoff at zero.
        return np.sqrt(ensemble_mean(_map(self.field, variance.clip(min=0))))

    def response(self):
        values = _divide(self.late_sum, self.late_count) - _divide(
            self.early_sum, self.early_count
        )
        return xr.DataArray(
            values,
            dims=("lat", "lon"),
            coords={"lat": self.field.lat, "lon": self.field.lon},
        )


def compare_trajectories(
    reference,
    generated,
    *,
    reconstruction,
    baseline=(1966, 2006),
    early=(1850, 1880),
    late=(2070, 2100),
):
    """Return computed time series/maps using two bounded passes, one year at a time.

    Reconstruction errors use matching months and common finite support.
    AR statistics use a common baseline, without treating stochastic trajectories
    as predictions of the reference's individual weather realizations.
    Pass early=None to omit the late-minus-early response comparison. The first
    pass fits monthly baseline means/support; the second accumulates diagnostics.
    Only small output time series grow with the chronology; working field arrays
    hold at most one year plus the twelve baseline climatology maps per member.
    """
    ref_series_dims = tuple(d for d in reference.dims if d not in ("lat", "lon"))
    gen_series_dims = tuple(d for d in generated.dims if d not in ("lat", "lon"))
    reference, generated = (
        reference.reset_coords(drop=True).transpose("time", ..., "lat", "lon"),
        generated.reset_coords(drop=True).transpose("time", ..., "lat", "lon"),
    )
    validate_months(reference.time)
    validate_months(generated.time)
    xr.align(reference, generated, join="exact", exclude={"time", "member"})
    if normalized_unit_name(reference.attrs.get("units", "")) != normalized_unit_name(
        generated.attrs.get("units", "")
    ):
        raise ValueError("Reference and generated units differ")
    if reconstruction:
        if "member" in reference.dims or "member" in generated.dims:
            raise ValueError(
                "Reconstruction comparison requires one matched trajectory"
            )
        reference, generated = xr.align(reference, generated, join="exact")
    # Reject incomplete requested periods before reading any physical data.
    for field in (reference, generated):
        complete_period(field, baseline)
        if early is not None:
            complete_period(field, early)
            complete_period(field, late)
    ref_monthly, gen_monthly, ref_clim, gen_clim, support = _baseline_statistics(
        reference, generated, baseline, reconstruction
    )
    weights = np.cos(np.deg2rad(reference.lat.values.astype(np.float64))).clip(min=0)[
        :, None
    ]
    ref_stats = _TrajectoryStatistics(
        reference, ref_monthly, baseline, early, late, weights
    )
    gen_stats = _TrajectoryStatistics(
        generated, gen_monthly, baseline, early, late, weights
    )
    if reconstruction:
        square_error_sum = np.zeros(reference.shape[-2:])
        valid_months = np.zeros(reference.shape[-2:], dtype=np.int64)
        global_bias = np.full(len(reference.time), np.nan)
        global_rmse = np.full(len(reference.time), np.nan)
        for year, selection in _year_slices(reference):
            r = _read_year(reference, selection)
            g = _read_year(generated, selection)
            valid = np.isfinite(r) & np.isfinite(g)
            r, g = np.where(valid, r, np.nan), np.where(valid, g, np.nan)
            ref_stats.add_year(r, year, selection)
            gen_stats.add_year(g, year, selection)
            error = g - r
            global_bias[selection] = _global_mean(error, weights)
            global_rmse[selection] = np.sqrt(_global_mean(error**2, weights))
            square_error_sum += np.where(valid, error**2, 0.0).sum(axis=0)
            valid_months += valid.sum(axis=0)
    else:
        # AR reference and generated chronologies need not have the same dates.
        for stats in (ref_stats, gen_stats):
            for year, selection in _year_slices(stats.field):
                values = np.where(support, _read_year(stats.field, selection), np.nan)
                stats.add_year(values, year, selection)
    ref_clim = ensemble_mean(_map(reference, ref_clim))
    gen_clim = ensemble_mean(_map(generated, gen_clim))
    # Explicitly outer-join time coordinates for AR sources with different coverage.
    series = xr.merge(
        [
            ref_stats.series("reference").transpose(*ref_series_dims),
            gen_stats.series("generated").transpose(*gen_series_dims),
        ],
        join="outer",
    )
    maps = xr.Dataset(
        {
            "reference_climatology": ref_clim,
            "generated_climatology": gen_clim,
            "climatological_bias": gen_clim - ref_clim,
        }
    )
    series["global_climatological_bias"] = weighted_mean(maps.climatological_bias)
    if reconstruction:
        series["global_bias"] = ("time", global_bias)
        series["global_rmse"] = ("time", global_rmse)
        maps["rmse"] = _map(reference, np.sqrt(_divide(square_error_sum, valid_months)))
        maps["valid_months"] = _map(reference, valid_months)
    else:
        # RMS across member temporal SDs, computed on the SAME baseline period.
        # This includes residual forced variability; it is not an IV decomposition.
        maps["reference_anomaly_sd"] = ref_stats.anomaly_sd()
        maps["generated_anomaly_sd"] = gen_stats.anomaly_sd()
    if early is not None:
        maps["reference_response"] = ref_stats.response()
        maps["generated_response"] = gen_stats.response()
        maps["response_difference"] = maps.generated_response - maps.reference_response
    attrs = dict(
        variable=reference.name or "field",
        units=reference.attrs.get("units", ""),
        comparison="reconstruction" if reconstruction else "autoregressive climate",
        baseline=f"{baseline[0]}--{baseline[1]}",
        anomalies="own baseline monthly climatology removed; trend retained",
        spatial_weights="cos(latitude), common finite support",
        response=(
            "omitted"
            if early is None
            else f"{late[0]}--{late[1]} minus {early[0]}--{early[1]}"
        ),
    )
    series.attrs, maps.attrs = attrs.copy(), attrs.copy()
    return series, maps
