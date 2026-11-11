import numpy as np
import pandas as pd
import xarray as xr

from monthly_diffusion.analysis.enso_diversity import regional_trajectory_anomalies, event_summary
from monthly_diffusion.analysis.enso_dynamics import BOXES, detect_events
from monthly_diffusion.analysis.spatial import weighted_mean


def test_regional_joint_fit_commutes_with_fixed_mask_spatial_mean():
    # Partial first year matches the emulator's March 1850 start; no padding.
    time = pd.date_range("1850-03-01", "1855-12-01", freq="MS")
    rng = np.random.default_rng(456)
    field = xr.DataArray(rng.normal(size=(5, len(time), 2, 3)),
        dims=("member", "time", "lat", "lon"),
        coords=dict(member=np.arange(5), time=time, lat=[-4.5, .5], lon=[160.5, 161.5, 162.5]))
    field.values[:, :, 0, 0] = np.nan
    regional = weighted_mean(field)
    raw = xr.Dataset({name: regional for name in BOXES})
    indices, fit = regional_trajectory_anomalies(raw)
    t = (time.year*12+time.month-(1850*12+1)).to_numpy()/12
    design = np.column_stack((np.eye(12)[time.month-1], t))
    values = field.transpose("time", "member", "lat", "lon").values.reshape(len(time), -1)
    valid = np.isfinite(values).all(0)
    residual = np.full_like(values, np.nan)
    residual[:, valid] = values[:, valid] - design@np.linalg.lstsq(design, values[:, valid], rcond=None)[0]
    field_first = field.copy(data=residual.reshape(len(time), 5, 2, 3).transpose(1, 0, 2, 3))
    np.testing.assert_allclose(indices.nino34, weighted_mean(field_first), atol=2.e-14)
    assert indices.sizes["member"] == 5
    assert fit.attrs["fit_start"] == "1850-03-01"
    np.testing.assert_allclose(indices.nino34.groupby("time.month").mean(), 0, atol=2.e-14)
    np.testing.assert_allclose((indices.nino34*t).sum("time"), 0, atol=2.e-13)


def test_diversity_does_not_require_twelve_month_composite_window():
    time = pd.date_range("1850-01-01", periods=24, freq="MS")
    value = np.zeros((1, 24))
    value[:, 4:7] = 1.2
    value[:, 15:18] = -1.2
    ds = xr.Dataset({name: (("member", "time"), value.copy()) for name in BOXES},
                    coords=dict(member=["r1i1p1f1"], time=time))
    events, _ = detect_events(ds, peak_years=(1850, 2014), max_lag=0)
    assert events.used.sum() == 2
    assert not events.boundary_censored.any()
    summary = event_summary(events, ds)
    assert list(summary.events) == [1, 1]
    np.testing.assert_allclose(summary.events_per_decade_per_member, 5)
