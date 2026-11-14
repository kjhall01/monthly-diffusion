import numpy as np
import pytest
import xarray as xr
from workflows.analysis import *


def test_area_and_uncentered_metrics():
    from workflows.analysis.spatial import weighted_uncentered_pattern_correlation
    x = xr.DataArray([[1., 1.], [3., 3.]], dims=("lat", "lon"), coords=dict(lat=[0, 60], lon=[0, 180]))
    assert float(weighted_mean(x)) == pytest.approx(5/3)
    assert float(weighted_uncentered_pattern_correlation(x, 2*x)) == pytest.approx(1)
    assert float(weighted_rmse(x, 2*x)) == pytest.approx(np.sqrt(11/3))


def test_internal_regression_and_seasons():
    dates = xr.date_range("1999-12", periods=15, freq="MS", calendar="noleap", use_cftime=True)
    x = xr.DataArray(np.arange(30.).reshape(2, 15), dims=("member", "time"), coords=dict(member=[0, 1], time=dates))
    internal = internal_variability(x)
    np.testing.assert_allclose(internal.mean("member"), 0.)
    np.testing.assert_allclose(internal, (x-x.mean("member"))*np.sqrt(2))
    assert float(seasonal_means(x).isel(member=0, season_year=0)) == pytest.approx((0*31+1*31+2*28)/90)
    index = xr.DataArray(np.arange(15.), dims="time", coords=dict(time=dates))
    assert float(teleconnection(2*index+10, index)) == pytest.approx(2*float(index.std()))


def test_response_inclusive_periods():
    dates = xr.date_range("1850-01", "2100-12", freq="MS")
    years = dates.year.astype(float)
    x = xr.DataArray(np.tile(years, (2, 1)), dims=("member", "time"), coords=dict(time=dates))
    assert float(forced_response(x)) == pytest.approx(220.)
    with pytest.raises(ValueError): forced_response(x.isel(time=slice(1, None)))
