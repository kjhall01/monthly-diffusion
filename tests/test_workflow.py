import numpy as np
import torch
import xarray as xr
import pytest
from workflows.legacy_model import MonthlyDiffusionV1
from workflows.inference.archive import open_rollout


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_staged_xarray_and_streaming(tiny_data, tmp_path, device, train_small_run):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    a, o, s, c, config = tiny_data
    model = train_small_run(tiny_data, tmp_path / "train", device=device)
    model.save(tmp_path / "train/model.pt")
    assert model.stages["autoencoder"]["atmosphere"]["epoch"] == 1
    assert model.stages["diffusion"]["epoch"] == 1
    assert all(not p.requires_grad for p in model.autoencoder.parameters())
    z, absent = model.encode(a.isel(time=slice(0, 2)), co2=c)
    assert absent is None and z.dims == ("time", "node", "channel")
    decoded, absent = model.decode(z, co2=c)
    assert absent is None and decoded.t2m.attrs["units"] == "K"
    restored = MonthlyDiffusionV1.load(tmp_path/"train/model.pt", device=device)
    zr, _ = restored.encode(a.isel(time=slice(0, 2)), co2=c)
    xr.testing.assert_equal(z, zr)
    initial_a, initial_o = a.isel(time=slice(0, 2)), o.isel(time=slice(0, 2))
    memory = model.rollout(initial_a, initial_o, co2=c, months=3, members=2, seed=41)
    disk = model.rollout(initial_a, initial_o, co2=c, months=3, members=2, seed=41, path=tmp_path/"forecast")
    xr.testing.assert_allclose(memory, disk.compute())
    assert np.isnan(disk.sst.values[:, :, :, 0]).all()
    before = (tmp_path/"forecast/month-000001.zarr/.zmetadata").stat().st_mtime_ns
    resumed = model.rollout(initial_a, initial_o, co2=c, months=3, members=2, seed=41, path=tmp_path/"forecast", resume=True)
    xr.testing.assert_allclose(memory, resumed.compute())
    assert before == (tmp_path/"forecast/month-000001.zarr/.zmetadata").stat().st_mtime_ns
    unopened = model.rollout(initial_a, initial_o, co2=c, months=3, members=2, seed=41,
                             path=tmp_path/"forecast", resume=True, open_result=False)
    assert unopened == tmp_path/"forecast"
    with pytest.raises(ValueError, match="differs"):
        model.rollout(initial_a, initial_o, co2=400, months=3, members=2, seed=41, path=tmp_path/"forecast", resume=True)
    # Restore an intermediate iterator and reproduce all subsequent draws.
    one = model.initialize(initial_a, initial_o, co2=c, members=2, seed=1)
    next(one)
    state = one.state_dict()
    expected = next(one)
    two = model.initialize(initial_a, initial_o, co2=c, members=2, seed=1)
    two.load_state_dict(state)
    actual = next(two)
    xr.testing.assert_equal(expected[0], actual[0])
    xr.testing.assert_equal(expected[1], actual[1])
