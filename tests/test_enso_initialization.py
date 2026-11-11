"""The readout is training-only and the initialization test changes history only."""
import importlib.util
from pathlib import Path
import numpy as np
import torch
import xarray as xr
from test_gaussian_forecast import ToyModel

spec = importlib.util.spec_from_file_location("enso_initialization",
    Path(__file__).parents[1]/"scripts/diagnose_enso_initialization.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_readout_recovers_known_latent_index():
    dates = xr.date_range("1966-01", periods=120, freq="MS")
    rng = np.random.default_rng(13)
    x = rng.normal(size=(120, 4))
    nuisance = diagnostic.design(dates)
    y = x@np.array([.5, 1., -.4, .1]) + nuisance@rng.normal(size=13)
    probe = diagnostic.fit_readout(x, y, dates)
    truth = y-nuisance@probe["physical_baseline"]
    pred = diagnostic.readout(x[:, None, :], dates, probe)[:, 0]
    assert np.corrcoef(pred, truth)[0, 1] > .999
    np.testing.assert_allclose(pred, truth, atol=.001)


def test_paired_initialization_preserves_order_dates_and_noise(tmp_path):
    model = ToyModel(nodes=32)
    cache = {"validation_time": np.array(["1965-11", "1965-12"], dtype="datetime64[M]"),
             "training_time": np.array(["1966-01", "1966-02"], dtype="datetime64[M]")}
    for split in ("validation", "training"):
        cache[split] = {c: torch.stack([torch.ones(32, 4), 2*torch.ones(32, 4)])
                        for c in ("atmosphere", "ocean")}
    probes = {c: dict(coef=np.ones(128), latent_baseline=np.zeros((13, 128)))
               for c in ("atmosphere", "ocean")}
    ds = diagnostic.compare_initialization(model, cache, probes, first="1966-01", months=2,
        members=2, seed=32, co2=350., output=tmp_path)
    np.testing.assert_array_equal(ds.time.values.astype("datetime64[M]"),cache["training_time"])
    observed = ds.sel(initialization="observed")
    np.testing.assert_array_equal(observed.initial_ocean.sel(history="current"), 2.)
    np.testing.assert_array_equal(observed.initial_ocean.sel(history="previous"), 1.)
    # Toy update is .8*current + .1*previous + innovation. Same RNG means
    # the difference between runs must be entirely due to their histories.
    delta = ds.sel(initialization="observed")-ds.sel(initialization="gaussian")
    expected = .8*delta.initial_ocean.sel(history="current")+.1*delta.initial_ocean.sel(history="previous")
    np.testing.assert_allclose(delta.latent_ocean.isel(time=0), expected, atol=5.e-7)
    assert model.decoded == 0
