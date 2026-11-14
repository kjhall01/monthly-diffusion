"""Streaming diagnostics compared with the original whole-array definitions."""

import numpy as np
import pytest
import xarray as xr
from workflows.analysis.trajectory import (
    compare_trajectories,
    complete_period,
    monthly_anomalies,
    ensemble_mean,
)
from workflows.analysis.spatial import weighted_mean
from workflows.analysis.climate import forced_response
from workflows.data.dataloader import validate_months, normalized_unit_name


# Keep the old formula here as a small eager oracle, independent of the running
# sums used by production. Test arrays are deliberately small and irregular.
def eager_comparison(
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
        response=(
            "omitted"
            if early is None
            else f"{late[0]}--{late[1]} minus {early[0]}--{early[1]}"
        ),
    )
    series.attrs, maps.attrs = attrs.copy(), attrs.copy()
    return series, maps


def irregular_fields(dtype):
    dates = xr.date_range("1964-01", periods=96, freq="MS")
    rng = np.random.default_rng(42)
    time = np.arange(96)[:, None, None]
    reference = xr.DataArray(
        (280 + 0.2 * time + 5 * np.sin(time / 6) + rng.normal(size=(96, 3, 4))).astype(
            dtype
        ),
        dims=("time", "lat", "lon"),
        coords={
            "time": dates,
            "lat": [-70.0, 0.0, 40.0],
            "lon": [0.0, 90.0, 180.0, 270.0],
        },
        name="t2m",
        attrs={"units": "K"},
    )
    generated = (
        reference + rng.normal(size=reference.shape).astype(dtype)
    ).assign_attrs(reference.attrs)
    reference.values[:, 0, 0] = np.nan
    generated.values[:, 0, 0] = np.nan
    reference.values[13, 1, 0] = np.nan
    generated.values[15:21, 2, 2] = np.nan
    reference.values[20] = np.nan  # a whole month without paired observations
    generated.values[25, 0, 1] = np.nan  # changes the AR fixed baseline support
    generated.values[72:84, 1, 1] = np.nan  # an annual mean missing entirely
    return reference, generated


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("responses", [False, True])
def test_streamed_reconstruction_matches_eager_with_changing_masks(dtype, responses):
    reference, generated = irregular_fields(dtype)
    # Infinity is excluded by the reconstruction's paired finite mask.
    generated.values[22, 2, 3] = np.inf
    options = dict(
        reconstruction=True,
        baseline=(1966, 1967),
        early=(1964, 1965) if responses else None,
        late=(1970, 1971),
    )
    expected = eager_comparison(reference, generated, **options)
    actual = compare_trajectories(
        reference.chunk({"time": 12, "lat": 1, "lon": 2}),
        generated.chunk({"time": 1, "lat": 3, "lon": 4}),
        **options,
    )
    for left, right in zip(actual, expected):
        xr.testing.assert_allclose(
            left, right, rtol=1e-6, atol=5e-5 if dtype == "float32" else 1e-11
        )
        assert left.attrs == right.attrs
        assert all(
            not hasattr(v.data, "__dask_graph__") for v in left.data_vars.values()
        )
    xr.testing.assert_equal(actual[1].valid_months, expected[1].valid_months)


@pytest.mark.parametrize(
    "reference_members,generated_members",
    [(None, None), (None, [3, 8]), ([1, 2], [3, 8, 9])],
)
@pytest.mark.parametrize("responses", [False, True])
def test_streamed_ar_preserves_members_support_and_unequal_dates(
    reference_members, generated_members, responses
):
    reference, generated = irregular_fields("float64")
    if reference_members is not None:
        reference = xr.concat(
            [reference, 1.3 * reference],
            dim=xr.IndexVariable("member", reference_members),
        ).assign_attrs(units="K")
    if generated_members is not None:
        generated = xr.concat(
            [(1 + 0.8 * i) * generated + 4 * i for i in range(len(generated_members))],
            dim=xr.IndexVariable("member", generated_members),
        ).assign_attrs(units="K")
        # One missing member's entire year must invalidate that ensemble annual
        # mean, before the remaining years are averaged for the response map.
        generated.values[1, 72:84, 1, 2] = np.nan
    reference = reference.isel(time=slice(2, -3))
    options = dict(
        reconstruction=False,
        baseline=(1966, 1967),
        early=(1965, 1965) if responses else None,
        late=(1969, 1970),
    )
    expected = eager_comparison(reference, generated, **options)
    actual = compare_trajectories(reference, generated, **options)
    for left, right in zip(actual, expected):
        xr.testing.assert_allclose(left, right, rtol=1e-11, atol=1e-11)
        assert left.attrs == right.attrs


def test_reads_are_bounded_to_one_year(monkeypatch):
    reference, generated = irregular_fields("float64")
    reference = reference.chunk({"time": 12, "lat": 1, "lon": 2})
    generated = generated.chunk({"time": 1})
    calls = []
    compute = xr.DataArray.compute

    def bounded_compute(field, *args, **kwargs):
        if "time" in field.dims:
            calls.append(field.sizes["time"])
            assert field.sizes["time"] <= 12
        return compute(field, *args, **kwargs)

    monkeypatch.setattr(xr.DataArray, "compute", bounded_compute)
    compare_trajectories(
        reference,
        generated,
        reconstruction=True,
        baseline=(1966, 1967),
        early=(1964, 1965),
        late=(1970, 1971),
    )
    # Two sources, two baseline years, eight full-chronology years.
    assert calls == [12] * (2 * (2 + 8))
