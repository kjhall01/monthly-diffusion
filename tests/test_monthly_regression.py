"""Streaming partial regression must equal explicit field residualization."""
import numpy as np
import pytest
import xarray as xr
from workflows.analysis.monthly_regression import regress_on_monthly_residual


class ArrayArchive:
    def __init__(self, fields):
        self.fields = fields
        self.lat, self.lon, self.members = fields.lat.values, fields.lon.values, fields.member.values
        self.units = {"t2m": "K"}
        self.keys = fields.time.dt.year.values*12+fields.time.dt.month.values

    def blocks(self, selection, size):
        for i in range(0, len(selection), size):
            yield selection[i:i+size]

    def read(self, variable, selection):
        return self.fields.isel(time=selection)


def example():
    time = np.arange("1958-03", "2026-01", dtype="datetime64[M]")
    n = len(time)
    months = (time.astype(int) % 12).astype(int)
    design = np.column_stack((np.eye(12)[months], np.arange(n)/12))
    rng = np.random.default_rng(23)
    x = rng.normal(size=(n, 3))
    x -= design@np.linalg.lstsq(design, x, rcond=None)[0]
    # Different climatologies/trends at each member and each point; field
    # variability not perfectly explained by the predictor.
    y = (design@rng.normal(size=(13, 18))).reshape(n, 3, 2, 3) + 3*x[:, :, None, None] + rng.normal(size=(n, 3, 2, 3))
    fields = xr.DataArray(y.transpose(1, 0, 2, 3), dims=("member", "time", "lat", "lon"),
        coords=dict(member=[0, 1, 2], time=time, lat=[-30, 30], lon=[0, 120, 240]), name="t2m")
    index = xr.DataArray(x.T, dims=("member", "time"), coords=dict(member=[0, 1, 2], time=time))
    residual = y.reshape(n, -1)-design@np.linalg.lstsq(design, y.reshape(n, -1), rcond=None)[0]
    expected = (residual.reshape(n, 3, 2, 3)*x[:, :, None, None]).sum((0, 1))/(x*x).sum()
    return fields, index, expected


def test_matches_explicit_detrending_with_intercepts_and_partial_edge_year():
    fields, index, expected = example()
    result = regress_on_monthly_residual(ArrayArchive(fields), index, "t2m", 17)
    np.testing.assert_allclose(result.slope, expected, atol=1e-12)
    np.testing.assert_array_equal(result.sample_count, 3*814)
    assert bool(result.complete_support.all())


def test_partial_support_is_masked_not_silently_reweighted():
    fields, index, _ = example()
    fields[0, 10, 0, 0] = np.nan
    result = regress_on_monthly_residual(ArrayArchive(fields), index, "t2m")
    assert np.isnan(result.slope[0, 0])
    assert result.sample_count[0, 0] == 3*814-1
    assert np.isfinite(result.slope[1, 1])


def test_raw_or_wrongly_matched_predictor_is_rejected():
    fields, index, _ = example()
    archive = ArrayArchive(fields)
    with pytest.raises(ValueError, match="monthly means and linear trend"):
        regress_on_monthly_residual(archive, index+2, "t2m")
    with pytest.raises(ValueError, match="members in order"):
        regress_on_monthly_residual(archive, index.isel(member=[2, 1, 0]), "t2m")
