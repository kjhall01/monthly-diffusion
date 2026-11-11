import numpy as np
import pytest
import torch
import xarray as xr
from monthly_diffusion import ModelConfig, TrainingConfig, Variable

torch.set_num_threads(1)


@pytest.fixture
def tiny_data():
    lat = np.arange(-85, 90, 10.)
    lon = np.arange(5, 360, 10.)
    time = xr.date_range("1966-01-01", periods=24, freq="MS")
    coords = dict(time=time, lat=lat, lon=lon)
    rng = np.random.default_rng(4)
    shape = (24, len(lat), len(lon))
    a = xr.Dataset({"t2m": (("time", "lat", "lon"), (280+rng.normal(size=shape)).astype("float32"))}, coords=coords)
    o = xr.Dataset({"sst": (("time", "lat", "lon"), (20+rng.normal(size=shape)).astype("float32"))}, coords=coords)
    a.t2m.attrs["units"] = "K"
    o.sst.attrs["units"] = "degrees C"
    o.sst.values[:, :, 0] = np.nan
    static = xr.Dataset({"phi_s": (("lat", "lon"), rng.normal(size=shape[1:]).astype("float32")),
                         "lsm": (("lat", "lon"), np.zeros(shape[1:], dtype="float32"))}, coords=dict(lat=lat, lon=lon))
    co2time = xr.date_range("1966-01-01", periods=50, freq="MS")
    co2 = xr.DataArray(np.linspace(320, 350, 50), dims="time", coords={"time": co2time})
    train = TrainingConfig(batch_size=4, max_epochs=1, patience=5, rollout_steps=3)
    config = ModelConfig(variables=[Variable("t2m", "atmosphere", "K", weight=.5),
                                  Variable("sst", "ocean", "degrees C", weight=.5)],
        ae_width=8, mesh_levels=(1, 0), refiners=(1, 1), transition_width=8,
        transition_depth=1, transition_heads=2, time_embedding_channels=8,
        autoencoder_training=train, diffusion_training=train)
    return a, o, static, co2, config
