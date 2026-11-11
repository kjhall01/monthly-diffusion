import numpy as np
import pytest
import xarray as xr
from monthly_diffusion.analysis.climate import internal_variability
from monthly_diffusion.analysis.lagged_enso import lagged_seasonal_means, lagged_enso_products, SEASONS


def test_month_mapping_native_calendar_and_missing_support():
    time = xr.date_range("1999-09", periods=12, freq="MS")
    x = xr.DataArray(np.arange(12.), dims="time", coords={"time": time})
    result = lagged_seasonal_means(x, 2000)
    np.testing.assert_allclose(result, [1., (3*31+4*31+5*28)/90,
                                       (6*31+7*30+8*31)/92, (9*30+10*31+11*31)/92])
    assert list(result.season.values) == list(SEASONS)
    # Year 2000 labels must NOT introduce a leap day into E3SM February.
    assert float(result.sel(season="DJF(0)")) != pytest.approx((3*31+4*31+5*29)/91)
    x[4] = np.nan
    result = lagged_seasonal_means(x, 2000)
    assert np.isnan(result.sel(season="DJF(0)"))
    assert np.isfinite(result.sel(season="SON(-1)"))
    with pytest.raises(ValueError, match="ordered without gaps"):
        lagged_seasonal_means(x.isel(time=slice(1, None)), 2000)
    with pytest.raises(ValueError, match="ordered without gaps"):
        lagged_seasonal_means(x.isel(time=list(range(11))+[10]), 2000)


def test_ensemble_removal_commutes_with_seasons():
    x = xr.DataArray(np.random.default_rng(11).normal(size=(5, 12)), dims=("member", "time"),
                    coords={"time": xr.date_range("1999-09", periods=12, freq="MS")})
    a = lagged_seasonal_means(internal_variability(x), 2000)
    b = internal_variability(lagged_seasonal_means(x, 2000))
    np.testing.assert_allclose(a, b, atol=1.e-14)


class SyntheticArchive:
    """Three winters with a huge common seasonal/forced signal to subtract."""
    def __init__(self):
        self.time = xr.DataArray(xr.date_range("1999-09", periods=36, freq="MS"), dims="time")
        self.keys = self.time.dt.year.values*12+self.time.dt.month.values
        self.members = ["a", "b", "c"]
        self.lat, self.lon = np.array([-2., 2.]), np.array([200., 220.])
        self.units = dict(sst="degC", z500="m2 s-2")
        amplitude = np.tile(np.repeat([1., 2., 4.], 12), (3, 1))*np.array([-1., 0., 1.])[:, None]
        self.sst = xr.DataArray(np.broadcast_to(amplitude[:, :, None, None], (3, 36, 2, 2)).copy(),
            dims=("member", "time", "lat", "lon"),
            coords=dict(member=self.members, time=self.time.values, lat=self.lat, lon=self.lon), name="sst")
        slopes = xr.DataArray(np.tile(np.repeat([2., -3., 4., -5.], 3), 3), dims="time",
                             coords={"time": self.time.values})
        background = xr.DataArray(np.arange(36)**2+5000., dims="time", coords={"time": self.time.values})
        self.z500 = self.sst*slopes+background
        self.z500[:, :, 0, 0] = np.nan
        self.sst = self.sst+background

    def read(self, variable, indices, *, lat=slice(None), lon=slice(None)):
        result = getattr(self, variable).isel(time=indices, lat=lat, lon=lon).copy()
        result.name = variable
        result.attrs["units"] = self.units[variable]
        return result


def test_streamed_regression_uses_degrees_not_standard_deviations(tmp_path):
    archive = SyntheticArchive()
    result = lagged_enso_products(archive, [2000, 2001, 2002], ["z500"], tmp_path)
    np.testing.assert_allclose(result.z500.sel(lat=2., lon=220.), [2., -3., 4., -5.], atol=1.e-12)
    assert np.isnan(result.z500.isel(lat=0, lon=0)).all()
    assert (result.count_z500.isel(lat=0, lon=0) == 0).all()
    assert (result.count_z500.isel(lat=1, lon=1) == 9).all()
    index = xr.load_dataset(tmp_path/"winter_indices.nc").nino34_djf
    assert float(index.std()) > 2  # Standardizing would yield quite different slopes.
    np.testing.assert_allclose(index.mean("member"), 0., atol=1.e-12)
    with pytest.raises(ValueError, match="Incomplete"):
        lagged_enso_products(archive, [1999, 2000], ["z500"], tmp_path/"invalid")
