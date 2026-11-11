import numpy as np
import pandas as pd
import xarray as xr
import pytest
from monthly_diffusion.analysis.enso_dynamics import box_ec, detect_events, djf_indices
from monthly_diffusion.analysis.lagged_enso import lagged_seasonal_means


def make_indices(values):
    data = np.asarray(values, dtype=float)
    if data.ndim == 1: data = data[None]
    coords = dict(member=np.arange(len(data)), time=pd.date_range("1990-01", periods=data.shape[1], freq="MS"))
    x = xr.DataArray(data, dims=("member", "time"), coords=coords)
    return xr.Dataset(dict(nino34=x, nino4=x*.6, nino12=x*1.2))


def test_box_ec_not_standardized():
    e, c = box_ec(np.array([2., -1.]), np.array([3., 0.]))
    np.testing.assert_allclose(e, [2., .5])
    np.testing.assert_allclose(c, [3.1, -1.7])


def test_short_episodes_and_event_peak_not_duration_weighted():
    x = np.zeros(72)
    x[15] = 2.1  # Exactly three active centered rolling windows, one episode.
    x[30:40] = -1.2
    events, smooth = detect_events(make_indices(x), peak_years=(1990, 1995), max_lag=12)
    assert len(events) == 2
    assert list(events.phase) == ["El Niño", "La Niña"]
    assert list(events.duration_months) == [3, 10]
    assert events.used.all()
    assert events.iloc[0].peak_index == 14  # Earliest of the tied rolling maxima.
    assert events.iloc[0].E == pytest.approx(.63)
    assert events.iloc[0].C == pytest.approx(.63)
    assert np.isnan(smooth.nino34.isel(time=0)).all()


def test_inclusive_threshold_and_member_boundaries():
    x = np.zeros((2, 60))
    x[0, 20:23] = .5
    x[1, 45:48] = -.5
    events, _ = detect_events(make_indices(x), peak_years=(1990, 1994), max_lag=0)
    assert len(events) == 2
    assert set(events.duration_months) == {1}
    assert set(events.member) == {"0", "1"}
    assert events.used.all()


def test_censored_events_and_lag_coverage():
    x = np.zeros(60)
    x[:5] = 2.
    x[20:23] = -1.
    x[-5:] = 2.
    events, _ = detect_events(make_indices(x), peak_years=(1990, 1994), max_lag=12)
    assert events.boundary_censored.sum() == 2
    assert events.used.sum() == 1
    assert events[events.used].iloc[0].phase == "La Niña"


def test_gregorian_and_noleap_seasons_are_explicit():
    x = xr.DataArray(np.arange(12.), dims="time", coords=dict(time=pd.date_range("1999-09", periods=12, freq="MS")))
    a = lagged_seasonal_means(x, 2000, calendar="standard").sel(season="DJF(0)")
    b = lagged_seasonal_means(x, 2000, calendar="noleap").sel(season="DJF(0)")
    assert float(a) == pytest.approx((3*31+4*31+5*29)/91)
    assert float(b) == pytest.approx((3*31+4*31+5*28)/90)


def test_djf_predictor_keeps_raw_amplitude_and_correct_year():
    dates = pd.date_range("1999-01", "2001-12", freq="MS")
    x = xr.DataArray(np.tile(np.arange(36.), (2, 1)), dims=("member", "time"),
                     coords=dict(member=[0, 1], time=dates))
    result = djf_indices(x.rename("nino34").to_dataset(), [2000, 2001], "noleap")
    assert result.dims == ("member", "winter_year")
    np.testing.assert_allclose(result[:, 0], (11*31+12*31+13*28)/90)
    np.testing.assert_allclose(result[:, 1], (23*31+24*31+25*28)/90)


def test_streamed_three_way_products_with_synthetic_archive(tmp_path):
    from monthly_diffusion.analysis.enso_dynamics import extract_source, lagged_products

    class Archive:
        members = [0, 1]
        lat = np.array([-7., -2., 2.])
        lon = np.array([170., 200., 230., 275.])
        units = dict(sst="degC", d20="m")

        def __init__(self):
            self.time = xr.DataArray(pd.date_range("1999-01", periods=48, freq="MS"), dims="time")
            self.keys = self.time.dt.year.values*12+self.time.dt.month.values
            t = np.arange(48)
            signal = np.stack([np.sin(t/5), np.cos(t/5)])[:, :, None, None]
            data = np.broadcast_to(signal, (2, 48, 3, 4)).copy()
            data += (t/30+2*np.sin(2*np.pi*t/12))[None, :, None, None]
            self.sst = xr.DataArray(data, dims=("member", "time", "lat", "lon"),
                coords=dict(member=self.members, time=self.time.values, lat=self.lat, lon=self.lon))

        def read(self, variable, indices, **kwargs):
            x = self.sst.isel(time=indices).copy()
            if variable == "d20": x = x*20
            return x.rename(variable).assign_attrs(units=self.units[variable])

        def blocks(self, selection, size):
            for i in range(0, len(selection), size):
                yield selection[i:i+size]

    config = dict(analysis_years=[1999, 2002], winter_years=[2000, 2001],
                  map_variables=["d20"], lag_months=2, threshold_degC=.1)
    spec = dict(calendar="noleap")
    extract_source(Archive(), config, spec, tmp_path)
    events = lagged_products(tmp_path, config, spec)
    maps = xr.load_dataset(tmp_path/"seasonal_d20.nc")
    assert maps.sizes["anomaly_method"] == 2
    assert (maps["count"] == 4).all()
    # Every box is identical in this fixture, so concurrent regression is 20 m/°C.
    np.testing.assert_allclose(maps.slope.sel(season="DJF(0)"), 20., atol=1.e-10)
    hov = xr.load_dataset(tmp_path/"hovmoller_d20.nc")
    assert hov.sizes["lag"] == 5
    assert hov.slope.dims == ("anomaly_method", "lag", "lon")
    assert (tmp_path/"regional_lag_correlations.csv").exists()
    assert set(events.member).issubset({"0", "1"})
