import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monthly_diffusion.analysis.decadal import (
    wet_coarsen, lanczos_weights, lowpass_annual, variability_fields,
    fit_covariance_eofs, project_eofs, match_eofs, pc_diagnostics, extract_annual_sst,
)


def field(values, years=None):
    values = np.asarray(values, float)
    if values.ndim == 2:
        values = values[:, :, None, None]*np.ones((1, 1, 3, 4))
    return xr.DataArray(values, dims=("member", "year", "lat", "lon"),
        coords=dict(member=np.arange(values.shape[0]), year=np.arange(values.shape[1]) if years is None else years,
                    lat=[-60., 0., 60.], lon=[45., 135., 225., 315.]))


def test_wet_coarsen_and_missing_support():
    x = xr.DataArray([[2., np.nan], [6., 6.]], dims=("lat", "lon"), coords=dict(lat=[0., 60.], lon=[0., 60.]))
    mask = x.notnull()
    out, area = wet_coarsen(x, mask, 2)
    np.testing.assert_allclose(area, 2.)
    np.testing.assert_allclose(out, 4.)  # (2*1 + 6*.5 + 6*.5)/2
    x.values[1, 1] = np.nan
    bad, _ = wet_coarsen(x, mask, 2)
    assert np.isnan(bad).all()


def test_filter_members_not_concatenated_and_no_padded_edges():
    k = lanczos_weights()
    np.testing.assert_allclose(k.sum(), 1.)
    x = field(np.array([np.ones(70), 9*np.ones(70)]), np.arange(1959, 2029))
    y = lowpass_annual(x, k)
    np.testing.assert_allclose(y.isel(member=0), 1.)
    np.testing.assert_allclose(y.isel(member=1), 9.)
    assert int(y.year[0]) == 1969 and int(y.year[-1]) == 2018
    oscillations = field(np.stack([np.sin(2*np.pi*np.arange(150)/4), np.sin(2*np.pi*np.arange(150)/30)]))
    filtered = lowpass_annual(oscillations, k)
    assert float(filtered.isel(member=0).std()) < .02
    assert float(filtered.isel(member=1).std()) > .6
    with pytest.raises(ValueError, match="consecutive"):
        lowpass_annual(x.isel(year=[0, 1, 3, 4]), k)


def test_finite_ensemble_correction_and_period_separation():
    years = np.arange(1851, 2020)
    signal = np.sin(np.arange(len(years))*.2)
    x = field(np.array([signal, -signal])+np.arange(len(years))[None, :], years)
    k = lanczos_weights()
    out = variability_fields(x, [1851, 2014], k)
    expected = np.sqrt(2)*(signal[:164]-signal[:164].mean())
    np.testing.assert_allclose(out.annual.isel(member=0, lat=0, lon=0), expected, atol=1.e-13)
    assert out.attrs["retained_last_year"] == 2004
    np.testing.assert_allclose(out.lowpass.mean("filtered_year"), 0, atol=1.e-14)
    x.loc[dict(year=slice(2015, None))] = 1.e20
    changed = variability_fields(x, [1851, 2014], k)
    xr.testing.assert_equal(out, changed)
    with pytest.raises(ValueError, match="Incomplete"):
        variability_fields(x, [1850, 2014], k)


def test_single_trajectory_linear_detrend():
    x = field((2+np.arange(67)*.2)[None])
    out = variability_fields(x, [0, 66], lanczos_weights(), "detrended")
    np.testing.assert_allclose(out.annual, 0, atol=1.e-14)
    with pytest.raises(ValueError, match="at least two"):
        variability_fields(x, [0, 66], lanczos_weights(), "ensemble")


def test_covariance_eof_common_projection_preserves_amplitude_and_rank():
    rng = np.random.default_rng(42)
    x = field(rng.normal(size=(3, 80, 3, 4)))
    weights = np.cos(np.deg2rad(x.lat)).broadcast_like(x.isel(member=0, year=0, drop=True))
    basis = fit_covariance_eofs(x, weights, 3)
    w = basis.weight/float(basis.weight.sum())
    flat = basis.pattern.values.reshape(3, -1)
    np.testing.assert_allclose((flat*w.values.ravel())@flat.T, np.eye(3), atol=1.e-12)
    pc = project_eofs(x-basis.center, basis.pattern, basis.weight)
    np.testing.assert_allclose(pc.std(("member", "year")), basis.pc_std)
    doubled = project_eofs(2*(x-basis.center), basis.pattern, basis.weight)
    np.testing.assert_allclose(doubled, 2*pc)
    swapped = basis.isel(mode=[2, 0, 1]).assign_coords(mode=basis.mode)
    swapped["pattern"] = -swapped.pattern
    columns, signs, cosines = match_eofs(basis, swapped)
    np.testing.assert_array_equal(columns, [1, 2, 0])
    np.testing.assert_allclose(signs, -1)
    np.testing.assert_allclose(cosines, 1)
    bad = x.copy()
    bad.values[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        project_eofs(bad, basis.pattern, basis.weight)


def test_pc_spectra_and_variance_use_same_interior_years():
    rng = np.random.default_rng(6)
    x = field(rng.normal(size=(3, 100, 3, 4)))
    out = variability_fields(x, [0, 99], lanczos_weights())
    basis = fit_covariance_eofs(out.lowpass.rename(filtered_year="year"), xr.ones_like(x[0, 0]), 3)
    product = pc_diagnostics(out, basis)
    assert product.psd.dims == ("member", "frequency", "mode")
    expected = product.annual_pc.sel(year=product.filtered_year.values).var("year")
    xr.testing.assert_allclose(product.annual_variance, expected)
    assert product.sizes["filtered_year"] == 80


def test_annual_extraction_omits_partial_years_and_keeps_members():
    class Archive:
        time = xr.DataArray(pd.date_range("1958-03-01", "1960-12-01", freq="MS"), dims="time")
        units = {"sst": "degC"}

        def read(self, variable, indices):
            return xr.DataArray(np.ones((2, len(indices), 2, 2)), dims=("member", "time", "lat", "lon"),
                coords=dict(member=["a", "b"], time=self.time.values[indices], lat=[-45., 45.], lon=[90., 270.]))

    a = Archive()
    support = a.read("sst", [0]).isel(member=0, time=0, drop=True).astype(bool)
    result = extract_annual_sst(a, support, factor=2)
    assert list(result.year.values) == [1959, 1960]
    assert list(result.member.values) == ["a", "b"]
    np.testing.assert_allclose(result.sst, 1.)
