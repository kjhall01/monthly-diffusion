import io

import numpy as np
import pytest
import torch
import xarray as xr

from monthly_diffusion.model import (
    MeshStage, Preprocessor, RefinedIcosahedralMeshDecoder,
    RefinedIcosahedralMeshEncoder,
)


@pytest.fixture
def training_data():
    values = np.random.default_rng(12).normal(size=(4, 4, 8, 2)).astype("float32")
    values[:, :, 0, :] = np.nan  # Permanently unsupported grid cells.
    values[0, 1, 2, 0] = np.nan  # Seasonal observation: retain this cell.
    values[1, 2, 3, 1] = np.inf
    values[:, 0, 1, 1] = np.nan  # Other channel still supports this cell.
    return xr.DataArray(
        values, dims=("time", "lat", "lon", "variable"),
        coords={"time": np.arange(4), "lat": [-67.5, -22.5, 22.5, 67.5],
                "lon": np.arange(8) * 45., "variable": ["sst", "sit"]},
    )


def test_compact_grid_statistics_and_dynamic_validation(training_data):
    pp = Preprocessor(training_data.transpose("variable", "time", "lat", "lon"), compact=True)
    expected_indices = np.flatnonzero(np.tile(np.arange(8) != 0, 4))
    np.testing.assert_array_equal(pp.point_indices, expected_indices)
    assert pp.grid.n_nodes == 28
    np.testing.assert_array_equal(pp.grid.metadata["flat_index"], expected_indices)
    full_statics = torch.arange(64).reshape(32, 2)
    torch.testing.assert_close(pp.select_points(full_statics, dim=0), full_statics[expected_indices])

    # Fit all finite observations, including the seasonal cell's other months.
    expected_zonal = training_data.where(np.isfinite(training_data)).mean(("time", "lon"))
    xr.testing.assert_allclose(pp.zonal_mean_xr.isel(lon=0, drop=True), expected_zonal)
    state, observed = pp.preprocess_torch(pp.xr_to_torch(training_data), return_mask=True)
    assert torch.isfinite(state).all()
    assert (state[~observed] == 0).all()

    validation = training_data.isel(time=slice(0, 2)).copy(deep=True)
    validation.values[0, 1, 1, 0] = np.nan  # Newly missing at a previously complete cell.
    validation.values[0, 0, 1, 1] = 4.0  # Newly observed channel on a retained cell.
    validation.values[0, 0, 0, :] = 5.0  # Validation cannot expand the saved grid.
    validation.values[1, :, :, :] = np.nan
    state, observed = pp.preprocess_torch(pp.xr_to_torch(validation), return_mask=True)
    assert torch.isfinite(state).all()
    assert (state[~observed] == 0).all()
    assert observed[0, 0, 1]
    assert not observed[1].any()
    _, present = pp.masked_mse(state, state, observed)
    assert not present[1].any()

    restored = pp.torch_to_xr(pp.postprocess_torch(state), times=validation.time.values)
    assert np.isnan(restored.values[:, :, 0]).all()
    expected = validation.values.reshape(2, 32, 2)[:, expected_indices]
    actual = restored.values.reshape(2, 32, 2)[:, expected_indices]
    np.testing.assert_allclose(actual[observed.numpy()], expected[observed.numpy()], atol=1e-6)


def test_masked_loss_renormalizes_and_has_zero_missing_gradients(training_data):
    pp = Preprocessor(training_data, compact=True)
    observed = torch.zeros(2, pp.grid.n_nodes, 2, dtype=torch.bool)
    observed[0, [0, 7], 0] = True  # Two distinct latitude weights.
    target = torch.zeros(observed.shape)
    prediction = torch.full(observed.shape, float("nan"))
    prediction[0, 0, 0], prediction[0, 7, 0] = 2., 4.
    prediction.requires_grad_()
    mse, present = pp.masked_mse(prediction, target, observed)
    w = np.cos(np.deg2rad([-67.5, -22.5]))
    assert mse[0, 0].item() == pytest.approx((w[0] * 4 + w[1] * 16) / w.sum())
    assert present.tolist() == [[True, False], [False, False]]
    assert (mse[~present] == 0).all()
    mse.sum().backward()
    assert torch.isfinite(prediction.grad).all()
    assert (prediction.grad[~observed] == 0).all()
    assert (prediction.grad[observed] != 0).all()


@pytest.mark.parametrize("compact", [False, True])
def test_checkpoint_roundtrip_and_legacy_layout(training_data, compact):
    pp = Preprocessor(training_data, compact=compact)
    assert ("point_indices" in pp.state_dict()) == compact
    if not compact:
        assert pp.grid.n_nodes == 32
        np.testing.assert_array_equal(pp.valid_mask_xr, np.isfinite(training_data).all("time"))
    buffer = io.BytesIO()
    torch.save(pp.state_dict(), buffer)
    buffer.seek(0)
    restored = Preprocessor.from_state_dict(torch.load(buffer, weights_only=True), pp.coords)
    torch.testing.assert_close(restored.grid.xyz, pp.grid.xyz)
    for name, value in pp.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)
    state = pp.preprocess_torch(pp.xr_to_torch(training_data))
    torch.testing.assert_close(restored.preprocess_torch(restored.xr_to_torch(training_data)), state)
    xr.testing.assert_equal(
        restored.torch_to_xr(state, times=training_data.time.values),
        pp.torch_to_xr(state, times=training_data.time.values),
    )


def test_compact_autoencoder_forward_backward_and_restore(training_data):
    torch.manual_seed(12)
    pp = Preprocessor(training_data, compact=True)
    stages = [MeshStage(name="m0", refinement=0, width=8, refiners=1, kernel_mlp_hidden_dim=8)]
    kwargs = dict(hierarchy=stages, preprocessor=pp, channels=2, latent_channels=3,
                  grid_kernel_hidden_dim=8)
    encoder = RefinedIcosahedralMeshEncoder(**kwargs)
    decoder = RefinedIcosahedralMeshDecoder(**kwargs)
    statics = pp.select_points(torch.randn(32, 2), dim=0)
    state, observed = pp.preprocess_torch(pp.xr_to_torch(training_data), return_mask=True)
    prediction = decoder(encoder(state, statics), statics)
    assert prediction.shape == (4, 28, 2)
    assert torch.isfinite(prediction).all()
    mse, present = pp.masked_mse(prediction, state, observed)
    mse[present].mean().backward()
    assert all(torch.isfinite(p.grad).all() for m in (encoder, decoder)
               for p in m.parameters() if p.grad is not None)

    restored_pp = Preprocessor.from_state_dict(pp.state_dict(), pp.coords)
    kwargs["preprocessor"] = restored_pp
    restored_encoder = RefinedIcosahedralMeshEncoder(**kwargs)
    restored_decoder = RefinedIcosahedralMeshDecoder(**kwargs)
    restored_encoder.load_state_dict(encoder.state_dict(), strict=True)
    restored_decoder.load_state_dict(decoder.state_dict(), strict=True)
    torch.testing.assert_close(
        restored_decoder(restored_encoder(state, statics), statics), prediction,
    )
