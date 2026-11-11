"""Matched four-source plotting retains ENSO amplitude and member timelines."""
import numpy as np
import pytest
import xarray as xr
from scipy.integrate import trapezoid

from monthly_diffusion.analysis.enso_comparison import enso_statistics
from monthly_diffusion.analysis.enso_diversity import regional_trajectory_anomalies


def index_series():
    time = np.arange("1960-01", "2000-01", dtype="datetime64[M]")
    wave = np.sin(2*np.pi*np.arange(len(time))/48)
    return xr.DataArray([wave, -wave], dims=("member", "time"),
                        coords=dict(member=[0, 1], time=time))


def test_temporal_statistics_are_memberwise_not_ensemble_mean():
    index = index_series()
    ds = enso_statistics(index, density_grid=np.linspace(-5, 5, 501))
    np.testing.assert_allclose(index.mean("member"), 0)
    assert float(ds.psd.max()) > 0
    peak = float(ds.psd.mean("member").isel(frequency=slice(1, None)).idxmax())
    assert 1/peak == pytest.approx(4.)
    np.testing.assert_allclose(ds.acf.sel(lag_months=0), 1)
    assert float(ds.acf.sel(lag_months=24).mean()) < -.9
    np.testing.assert_allclose(trapezoid(ds.psd, ds.frequency, axis=-1), .5, rtol=.01)


def test_physical_amplitudes_and_dimensionless_seasonal_ratios():
    index = index_series()
    grid = np.linspace(-8, 8, 1601)
    a = enso_statistics(index, density_grid=grid)
    b = enso_statistics(3*index, density_grid=grid)
    np.testing.assert_allclose(b.psd, 9*a.psd, atol=1e-12)
    np.testing.assert_allclose(b.acf, a.acf, atol=1e-12)
    np.testing.assert_allclose(b.pooled_seasonal_std, 3*a.pooled_seasonal_std)
    np.testing.assert_allclose(b.pooled_seasonal_amplification, a.pooled_seasonal_amplification)
    np.testing.assert_allclose(trapezoid(a.density, grid, axis=-1), 1, atol=1e-6)
    np.testing.assert_allclose(trapezoid(b.density, grid, axis=-1), 1, atol=1e-6)


def test_no_interpolation_of_missing_months_or_values():
    index = index_series()
    grid = np.linspace(-5, 5, 101)
    with pytest.raises(ValueError, match="consecutive"):
        enso_statistics(index.isel(time=np.r_[0:10, 11:480]), density_grid=grid)
    index[0, 20] = np.nan
    with pytest.raises(ValueError, match="finite"):
        enso_statistics(index, density_grid=grid)


def test_common_partial_year_monthly_and_linear_fit():
    time = np.arange("1958-03", "2026-01", dtype="datetime64[M]")
    dates = xr.DataArray(time, dims="time", coords=dict(time=time))
    t = np.arange(len(time))/12
    rng = np.random.default_rng(17)
    values = 27+np.sin(2*np.pi*(dates.dt.month.values-1)/12)+.03*t+rng.normal(size=(3, len(time)))
    raw = xr.Dataset({k: (("member", "time"), values) for k in ("nino34", "nino4", "nino12")},
                     coords=dict(member=[0, 1, 2], time=time))
    anomalies, fit = regional_trajectory_anomalies(raw)
    assert anomalies.sizes["time"] == 814
    np.testing.assert_allclose(anomalies.nino34.groupby("time.month").mean("time"), 0., atol=1e-12)
    np.testing.assert_allclose((anomalies.nino34*t).sum("time"), 0., atol=1e-7)
    assert fit.attrs["fit_start"] == "1958-03-01"
