"""Physical comparisons for reconstructions and free-running monthly trajectories."""

import numpy as np
import xarray as xr
from .climate import forced_response
from .spatial import weighted_mean
from ..data.dataloader import validate_months, normalized_unit_name


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


def compare_trajectories(
    reference,
    generated,
    *,
    reconstruction,
    baseline=(1966, 2006),
    early=(1850, 1880),
    late=(2070, 2100),
):
    """Return small time-series and map datasets; evaluate lazily per variable.

    Reconstruction errors use matching months and common finite support.
    AR statistics use a common baseline, without treating stochastic trajectories
    as predictions of the reference's individual weather realizations.
    Pass early=None to omit the late-minus-early response comparison.
    """
    reference, generated = (
        reference.reset_coords(drop=True),
        generated.reset_coords(drop=True),
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
        support = np.isfinite(reference) & np.isfinite(generated)
    else:
        # Fixed shared spatial support across the baseline. Keep all trajectory
        # members; do not mistake their ensemble mean variance for member variance.
        ref_valid = np.isfinite(complete_period(reference, baseline))
        gen_valid = np.isfinite(complete_period(generated, baseline))
        support = ref_valid.all([d for d in ("time", "member") if d in ref_valid.dims])
        support &= gen_valid.all([d for d in ("time", "member") if d in gen_valid.dims])
    reference, generated = reference.where(support), generated.where(support)
    ref_anomaly = monthly_anomalies(reference, baseline)
    gen_anomaly = monthly_anomalies(generated, baseline)
    ref_clim = ensemble_mean(complete_period(reference, baseline).mean("time"))
    gen_clim = ensemble_mean(complete_period(generated, baseline).mean("time"))
    # Explicitly outer-join time coordinates for AR sources with different coverage.
    series = xr.merge(
        [
            weighted_mean(reference).rename("reference_global_mean"),
            weighted_mean(generated).rename("generated_global_mean"),
            weighted_mean(ref_anomaly).rename("reference_global_anomaly"),
            weighted_mean(gen_anomaly).rename("generated_global_anomaly"),
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
        error = generated - reference
        series["global_bias"] = weighted_mean(error)
        series["global_rmse"] = np.sqrt(weighted_mean(error**2))
        maps["rmse"] = np.sqrt((error**2).mean("time"))
        maps["valid_months"] = support.sum("time")
    else:
        # RMS across member temporal SDs, computed on the SAME baseline period.
        # This includes residual forced variability; it is not an IV decomposition.
        maps["reference_anomaly_sd"] = np.sqrt(
            ensemble_mean(complete_period(ref_anomaly, baseline).var("time", ddof=0))
        )
        maps["generated_anomaly_sd"] = np.sqrt(
            ensemble_mean(complete_period(gen_anomaly, baseline).var("time", ddof=0))
        )
    if early is not None:
        maps["reference_response"] = forced_response(reference, early, late)
        maps["generated_response"] = forced_response(generated, early, late)
        maps["response_difference"] = maps.generated_response - maps.reference_response
    attrs = dict(
        variable=reference.name or "field",
        units=reference.attrs.get("units", ""),
        comparison="reconstruction" if reconstruction else "autoregressive climate",
        baseline=f"{baseline[0]}--{baseline[1]}",
        anomalies="own baseline monthly climatology removed; trend retained",
        spatial_weights="cos(latitude), common finite support",
        response="omitted"
        if early is None
        else f"{late[0]}--{late[1]} minus {early[0]}--{early[1]}",
    )
    series.attrs, maps.attrs = attrs.copy(), attrs.copy()
    return series, maps
