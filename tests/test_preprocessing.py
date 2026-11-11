import numpy as np
import torch
import pytest
import xarray as xr
from monthly_diffusion.data.preprocessing import Preprocessor, LatentNormalizer, transform, inverse_transform
from monthly_diffusion.data.configs import Variable
from monthly_diffusion.data.dataloader import validate_months, co2_values, next_month


@pytest.mark.parametrize("name,values", [("identity", [-100, 0, 5]), ("softplus_inverse", [0, 1, 1000]), ("logit", [0, .25, 1])])
def test_guarded_roundtrips(name, values):
    spec = Variable("v", "ocean", transform=name, eps=1e-4)
    x = np.array(values, dtype=float)
    y = transform(x, spec)
    assert np.isfinite(y).all()
    expected = np.clip(x, spec.eps, 1-spec.eps) if name == "logit" else np.maximum(x, spec.eps) if name == "softplus_inverse" else x
    np.testing.assert_allclose(inverse_transform(y, spec), expected)
    assert np.isnan(transform(np.array([np.nan]), spec))[0]


def test_train_only_stats(tiny_data):
    a, o, static, co2, config = tiny_data
    pre = Preprocessor(config.variables)
    pre.fit(a.isel(time=slice(0, 12)), o.isel(time=slice(0, 12)), static, co2.isel(time=slice(0, 12)))
    scaled = pre.encode(a.isel(time=slice(0, 12)), "atmosphere").t2m
    np.testing.assert_allclose(scaled.mean(("time", "lon")), 0., atol=2e-5)
    assert float(scaled.std()) == pytest.approx(1., rel=2e-5)
    np.testing.assert_allclose(pre.decode(pre.encode(o, "ocean"), "ocean").sst, o.sst, atol=1e-5)
    with pytest.raises(RuntimeError): pre.fit(a, o, static, co2)
    z = torch.randn(20, 12, 4)
    norm = LatentNormalizer().fit(z)
    torch.testing.assert_close(norm.normalize(z).mean((0, 1)), torch.zeros(4), atol=1e-6, rtol=0)
    torch.testing.assert_close(norm.denormalize(norm.normalize(z)), z)
    with pytest.raises(RuntimeError): norm.fit(z+10)


def test_calendar_and_forcing():
    import cftime
    dates = xr.date_range("1999-12", periods=3, freq="MS", calendar="360_day", use_cftime=True)
    time = xr.DataArray(dates, dims="time", coords={"time": dates})
    validate_months(time)
    assert next_month(dates[0]) == cftime.Datetime360Day(2000, 1, 1)
    assert (co2_values(280, time) == 280).all()
    with pytest.raises(ValueError, match="lacks"):
        co2_values(xr.DataArray([300], dims="time", coords={"time": dates[:1]}), time)
    with pytest.raises(ValueError): validate_months(time.isel(time=[0, 2]))
