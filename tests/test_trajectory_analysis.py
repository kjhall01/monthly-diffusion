"""Known errors, responses and seasonal anomalies on a masked global grid."""

import numpy as np
import pytest
import xarray as xr
from workflows.analysis.trajectory import compare_trajectories, monthly_anomalies


@pytest.fixture
def fields():
    time = xr.date_range("1966-01-01", periods=24, freq="MS")
    values = np.arange(24) % 12 + np.repeat([0.0, 4.0], 12)
    field = xr.DataArray(
        np.broadcast_to(values[:, None, None], (24, 3, 4)).copy(),
        dims=("time", "lat", "lon"),
        coords={
            "time": time,
            "lat": [-60.0, 0.0, 60.0],
            "lon": [0.0, 90.0, 180.0, 270.0],
        },
        name="temperature",
        attrs={"units": "K"},
    )
    field.values[:, 0, 0] = np.nan
    return field


def test_reconstruction_metrics_and_periods(fields):
    generated = (fields + 2).assign_attrs(fields.attrs)
    series, maps = compare_trajectories(
        fields,
        generated,
        reconstruction=True,
        baseline=(1966, 1967),
        early=(1966, 1966),
        late=(1967, 1967),
    )
    np.testing.assert_allclose(series.global_bias, 2)
    np.testing.assert_allclose(series.global_rmse, 2)
    np.testing.assert_allclose(
        series.generated_global_mean - series.reference_global_mean, 2
    )
    xr.testing.assert_equal(
        series.generated_global_anomaly,
        series.reference_global_anomaly.rename("generated_global_anomaly"),
    )
    np.testing.assert_allclose(maps.rmse.values[1:], 2)
    np.testing.assert_allclose(maps.climatological_bias.values[1:], 2)
    np.testing.assert_allclose(maps.reference_response.values[1:], 4)
    np.testing.assert_allclose(maps.response_difference.values[1:], 0)
    assert np.isnan(maps.rmse.values[0, 0])
    # Use the same accepted unit spellings as the training boundary.
    compare_trajectories(
        fields.assign_attrs(units="degrees C"),
        generated.assign_attrs(units="degree_Celsius"),
        reconstruction=True,
        baseline=(1966, 1967),
        early=None,
    )
    with pytest.raises(ValueError, match="Missing complete years"):
        monthly_anomalies(fields, (1965, 1967))
    with pytest.raises(ValueError, match="Missing complete years"):
        compare_trajectories(
            fields, generated, reconstruction=True, baseline=(1966, 1967)
        )
    with pytest.raises(ValueError, match="align"):
        compare_trajectories(
            fields, generated.assign_coords(lon=generated.lon + 1), reconstruction=True
        )
    with pytest.raises(ValueError, match="units differ"):
        compare_trajectories(
            fields, generated.assign_attrs(units="degC"), reconstruction=True
        )


def test_ar_statistics_keep_member_variance_and_unmatched_coverage(fields):
    generated = xr.concat(
        [2 * fields + 3, 2 * fields + 7], dim=xr.IndexVariable("member", [0, 1])
    ).assign_attrs(fields.attrs)
    series, maps = compare_trajectories(
        fields,
        generated,
        reconstruction=False,
        baseline=(1966, 1967),
        early=(1966, 1966),
        late=(1967, 1967),
    )
    assert "global_rmse" not in series
    np.testing.assert_allclose(maps.reference_anomaly_sd.values[1:], 2)
    np.testing.assert_allclose(maps.generated_anomaly_sd.values[1:], 4)
    np.testing.assert_allclose(maps.response_difference.values[1:], 4)
    assert "member" in series.generated_global_anomaly.dims
    np.testing.assert_allclose(
        series.generated_global_anomaly.mean("time"), 0, atol=1e-12
    )
    assert np.isnan(maps.generated_anomaly_sd.values[0, 0])
    # A historical reference and a longer model trajectory need not have equal dates.
    short_series, short_maps = compare_trajectories(
        fields.sel(time="1966"),
        generated,
        reconstruction=False,
        baseline=(1966, 1966),
        early=None,
    )
    assert short_series.sizes["time"] == 24
    assert "reference_response" not in short_maps
    assert short_series.reference_global_mean.sel(time="1967").isnull().all()


def test_main_plots_use_robinson_and_coastlines(fields, tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    pytest.importorskip("cartopy")
    from cartopy.mpl.geoaxes import GeoAxes
    import cartopy.crs as ccrs
    from workflows.analysis.trajectory_plots import plot_series, plot_maps

    coastlines = []

    def offline_coastlines(axis, *args, **kwargs):
        assert isinstance(axis.projection, ccrs.Robinson)
        coastlines.append(axis)

    # Verify the coastline request without downloading Natural Earth in CI.
    monkeypatch.setattr(GeoAxes, "coastlines", offline_coastlines)
    for reconstruction in (True, False):
        series, maps = compare_trajectories(
            fields,
            (fields + 1).assign_attrs(fields.attrs),
            reconstruction=reconstruction,
            baseline=(1966, 1967),
            early=(1966, 1966),
            late=(1967, 1967),
        )
        plot_series(series, tmp_path / f"{reconstruction}_series.png")
        plot_maps(maps, tmp_path / f"{reconstruction}_maps.png")
    assert len(coastlines) == 11
    assert all(path.stat().st_size > 1000 for path in tmp_path.glob("*.png"))
