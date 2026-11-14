import numpy as np
import torch
import pytest
import xarray as xr
from monthly_diffusion.preprocessing import (
    Preprocessor,
    LatentNormalizer,
    transform,
    inverse_transform,
)
from monthly_diffusion.preprocessing import Variable
from workflows.data.dataloader import validate_months, co2_values, next_month


@pytest.mark.parametrize(
    "name,values",
    [
        ("identity", [-100, 0, 5]),
        ("log1p", [0, 1, 1000]),
        ("softplus_inverse", [0, 1, 1000]),
        ("logit", [0, 0.25, 1]),
    ],
)
def test_guarded_roundtrips(name, values):
    spec = Variable("v", "ocean", transform=name, eps=1e-4)
    x = np.array(values, dtype=float)
    y = transform(x, spec)
    assert np.isfinite(y).all()
    expected = (
        np.clip(x, spec.eps, 1 - spec.eps)
        if name == "logit"
        else np.maximum(x, spec.eps) if name == "softplus_inverse" else x
    )
    np.testing.assert_allclose(inverse_transform(y, spec), expected)
    assert np.isnan(transform(np.array([np.nan]), spec))[0]


def test_train_only_stats(tiny_data):
    a, o, static, co2, config = tiny_data
    pre = Preprocessor(config.variables)
    pre.fit(
        a.isel(time=slice(0, 12)),
        o.isel(time=slice(0, 12)),
        static,
        co2.isel(time=slice(0, 12)),
    )
    scaled = pre.encode(a.isel(time=slice(0, 12)), "atmosphere").t2m
    np.testing.assert_allclose(scaled.mean(("time", "lon")), 0.0, atol=2e-5)
    assert float(scaled.std()) == pytest.approx(1.0, rel=2e-5)
    np.testing.assert_allclose(
        pre.decode(pre.encode(o, "ocean"), "ocean").sst, o.sst, atol=1e-5
    )
    with pytest.raises(RuntimeError):
        pre.fit(a, o, static, co2)
    z = torch.randn(20, 12, 4)
    norm = LatentNormalizer().fit(z)
    torch.testing.assert_close(
        norm.normalize(z).mean((0, 1)), torch.zeros(4), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(norm.denormalize(norm.normalize(z)), z)
    with pytest.raises(RuntimeError):
        norm.fit(z + 10)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("mask_constants", [False, True])
@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_gridpoint_statistics_floor_masks_and_future_data(
    tiny_data, lazy, mask_constants
):
    atmosphere, _, statics, co2, config = tiny_data
    fields = atmosphere.copy(deep=True)
    nlat, nlon = fields.sizes["lat"], fields.sizes["lon"]
    center = 275 + np.arange(nlat)[:, None] + np.arange(nlon)[None, :] / 10
    scale = np.broadcast_to(np.linspace(1, 4, nlon), (nlat, nlon))
    time = np.arange(12, dtype=float)
    time = (time - time.mean()) / time.std()
    training = center + time[:, None, None] * scale
    training[:, 0, 0] = np.nan  # never observed
    training[1:, 0, 1] = np.nan  # only one observation: no fitted local variance
    training[:, 0, 2] = 7.0  # constant but valid
    training[:, 0, 3] = 7.0 + time * 1e-8  # nearly constant
    values = np.concatenate([training, np.full_like(training, 1000)])
    fields["t2m"] = xr.DataArray(
        values, dims=("time", "lat", "lon"), coords=fields.coords
    )
    fields.t2m.attrs["units"] = "K"
    source = fields.chunk({"time": 4, "lat": 6}) if lazy else fields
    pre = Preprocessor(
        [config.variables[0]],
        centering="gridpoint",
        gridpoint_std_floor=0.05,
        mask_constant_cells=mask_constants,
    )
    pre.fit_component(
        source.isel(time=slice(0, 12)),
        "atmosphere",
        statics,
        co2.isel(time=slice(0, 12)),
    )
    stats = pre.stats["t2m"]
    assert stats["center"].dims == stats["std"].dims == ("lat", "lon")
    np.testing.assert_allclose(stats["center"][2:], center[2:], atol=1e-12)
    np.testing.assert_allclose(stats["std"][2:], scale[2:], atol=1e-12)
    assert not bool(stats["mask"][0, 0]) and not bool(stats["mask"][0, 1])
    assert bool(stats["mask"][0, 2]) == (not mask_constants)
    assert bool(stats["mask"][0, 3])  # tiny but nonzero variability stays valid
    assert float(stats["std"][0, 2]) == float(stats["std"][0, 3]) == stats["std_floor"]
    assert stats["std_floor"] > 0

    encoded = pre.encode(fields, "atmosphere")
    normalized = encoded.t2m.values[:12, 2:]
    np.testing.assert_allclose(normalized.mean(0), 0, atol=1e-6)
    np.testing.assert_allclose(normalized.std(0), 1, atol=1e-6)
    # Future values do not refit the training center/std, and cannot revive
    # cells without enough training observations.
    assert np.isnan(encoded.t2m[:, 0, :2]).all()
    expected_future = (1000 - center[2:]) / scale[2:]
    np.testing.assert_allclose(encoded.t2m.values[12, 2:], expected_future, rtol=1e-6)
    expected = fields.t2m.where(stats.get("data_mask", stats["mask"]))
    if mask_constants:
        assert np.isnan(encoded.t2m[:, 0, 2]).all()
        expected.values[:, 0, 2] = 7.0  # fitted constant, including future months
    xr.testing.assert_allclose(
        pre.decode(encoded, "atmosphere").t2m,
        expected,
        atol=1e-5,
    )
    tensor = torch.tensor(encoded.t2m.values[:, None], requires_grad=True)
    decoded = pre.decode_tensor(tensor, "atmosphere")
    np.testing.assert_allclose(
        decoded.detach().numpy()[:, 0],
        expected,
        rtol=1e-6,
        equal_nan=True,
    )
    decoded[0, 0, 3, 4].backward()
    assert float(tensor.grad[0, 0, 3, 4]) == pytest.approx(
        float(stats["std"][3, 4]), rel=1e-6
    )


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_gridpoint_rejects_no_supported_training_cells(tiny_data):
    atmosphere, _, statics, co2, config = tiny_data
    pre = Preprocessor([config.variables[0]], centering="gridpoint")
    with pytest.raises(ValueError, match="two finite training values"):
        pre.fit_component(
            atmosphere.isel(time=slice(0, 1)),
            "atmosphere",
            statics,
            co2.isel(time=slice(0, 1)),
        )


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_constant_ice_exception_and_physical_metrics(tiny_data, tmp_path):
    from workflows.data.dataloader import ComponentStates
    from workflows.training.losses import reconstruction_mse

    _, ocean, statics, co2, _ = tiny_data
    fields = xr.Dataset({"sic": ocean.sst.copy(deep=True).astype("float64")})
    fields.sic.values[:, :, 1] = 0
    fields.sic.values[:, :, 2] = 1
    fields.sic.values[:, :, 3] = (
        0.3  # constant nonzero value can have roundoff variance
    )
    fields.sic.values[:, :, 4] = np.tile([0.0, 0.5], 12)[:, None]
    fields.sic.values[12:, :, 1] = 0.2  # future changes must not refit the mask
    pre = Preprocessor(
        [Variable("sic", "ocean")], centering="gridpoint", mask_constant_cells=True
    )
    pre.fit_component(fields.isel(time=slice(0, 12)), "ocean", statics, co2[:12])
    stats = pre.stats["sic"]
    assert not stats["mask"][:, [0, 1, 3]].any()
    assert stats["mask"][:, [2, 4]].all()
    state, target, _, _ = ComponentStates(fields, "ocean", pre, co2)[12]
    assert torch.isnan(state[0, :, 1]).all()
    assert torch.all(target[0, :, 1] == 0.2)
    # Retained SIC=1 gets a positive std floor and a finite normalized target.
    assert torch.all(state[0, :, 2] == 0)
    assert (stats["std"][:, 2] > 0).all()
    prediction = torch.nan_to_num(state)[None].requires_grad_()
    decoded = pre.decode_tensor(prediction, "ocean")
    assert torch.all(decoded[0, 0, :, 1] == 0)
    assert torch.all(decoded[0, 0, :, 2] == 1)
    assert torch.all(decoded[0, 0, :, 3] == 0.3)
    assert torch.isnan(decoded[0, 0, :, 0]).all()
    latitudes = torch.tensor(fields.lat.values)
    assert reconstruction_mse(prediction, state[None], latitudes) == 0
    assert reconstruction_mse(decoded, target[None], latitudes) > 0
    loss = reconstruction_mse(prediction + 1, state[None], latitudes).sum()
    loss.backward()
    assert torch.all(prediction.grad[0, 0, :, 1] == 0)
    assert torch.all(prediction.grad[0, 0, :, 2] > 0)
    pre.save(tmp_path / "preprocessor.pt")
    restored = Preprocessor.load(tmp_path / "preprocessor.pt").for_component("ocean")
    assert restored.mask_constant_cells
    torch.testing.assert_close(
        restored.decode_tensor(prediction, "ocean"), decoded, equal_nan=True
    )
    xr_decoded = restored.decode(restored.encode(fields, "ocean"), "ocean")
    np.testing.assert_allclose(
        xr_decoded.sic[12], decoded.detach().numpy()[0, 0], rtol=1e-6, equal_nan=True
    )


def test_scaled_log1p_tensor_inverse_and_gradient(tiny_data):
    atmosphere, _, statics, co2, _ = tiny_data
    fields = xr.Dataset({"mtpr": (atmosphere.t2m - 280).clip(min=0) / 86400})
    pre = Preprocessor(
        [Variable("mtpr", "atmosphere", transform="log1p", scale=1 / 86400)],
        centering="gridpoint",
        mask_constant_cells=True,
    )
    pre.fit_component(fields, "atmosphere", statics, co2_values(co2, fields.time))
    encoded = pre.encode(fields, "atmosphere").mtpr.values[:, None]
    tensor = torch.tensor(encoded, requires_grad=True)
    physical = pre.decode_tensor(tensor, "atmosphere")
    np.testing.assert_allclose(physical.detach().numpy()[:, 0], fields.mtpr, atol=1e-10)
    physical.sum().backward()
    expected = (fields.mtpr + 1 / 86400) * pre.stats["mtpr"]["std"]
    np.testing.assert_allclose(tensor.grad.numpy()[:, 0], expected, rtol=1e-6)


def test_old_preprocessor_without_mask_setting(tiny_data):
    atmosphere, _, statics, co2, config = tiny_data
    pre = Preprocessor([config.variables[0]], centering="gridpoint")
    del pre.mask_constant_cells  # old pickle has no new option or mask arrays
    fields = atmosphere.copy(deep=True)
    fields.t2m.values[:, 0, 0] = 7
    pre.fit_component(fields, "atmosphere", statics, co2_values(co2, fields.time))
    assert bool(pre.stats["t2m"]["mask"][0, 0])
    assert not pre.for_component("atmosphere").mask_constant_cells
    xr.testing.assert_allclose(
        pre.decode(pre.encode(fields, "atmosphere"), "atmosphere"), fields
    )


def test_calendar_and_forcing():
    import cftime

    dates = xr.date_range(
        "1999-12", periods=3, freq="MS", calendar="360_day", use_cftime=True
    )
    time = xr.DataArray(dates, dims="time", coords={"time": dates})
    validate_months(time)
    assert next_month(dates[0]) == cftime.Datetime360Day(2000, 1, 1)
    assert (co2_values(280, time) == 280).all()
    with pytest.raises(ValueError, match="lacks"):
        co2_values(xr.DataArray([300], dims="time", coords={"time": dates[:1]}), time)
    with pytest.raises(ValueError):
        validate_months(time.isel(time=[0, 2]))
