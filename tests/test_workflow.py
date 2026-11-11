import numpy as np
import torch
import xarray as xr
import pytest
from monthly_diffusion import MonthlyDiffusionV1, open_rollout


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_staged_xarray_and_streaming(tiny_data, tmp_path, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    a, o, s, c, config = tiny_data
    model = MonthlyDiffusionV1(statics=s, config=config, device=device)
    model.train(a.isel(time=slice(0, 12)), o.isel(time=slice(0, 12)),
        validation=(a.isel(time=slice(12, None)), o.isel(time=slice(12, None))), co2=c, directory=tmp_path/"train")
    assert model.stages["autoencoder"]["epoch"] == 1
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


def test_training_resume_matches_uninterrupted(tmp_path):
    from monthly_diffusion.training.common import fit
    from monthly_diffusion import TrainingConfig
    from torch.utils.data import Dataset
    class Data(Dataset):
        def __len__(self): return 8
        def __getitem__(self, i): return {"x": torch.tensor([float(i)])}
    def objective(model, batch):
        return {"loss": ((model(batch["x"])-2*batch["x"])**2).mean()}
    torch.manual_seed(1)
    a = torch.nn.Linear(1, 1)
    b = torch.nn.Linear(1, 1)
    b.load_state_dict(a.state_dict())
    config = TrainingConfig(max_epochs=2)
    fit(a, Data(), Data(), objective, config, tmp_path/"whole")
    config.max_epochs = 1
    fit(b, Data(), Data(), objective, config, tmp_path/"resume")
    config.max_epochs = 2
    fit(b, Data(), Data(), objective, config, tmp_path/"resume", resume=True)
    for x, y in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
