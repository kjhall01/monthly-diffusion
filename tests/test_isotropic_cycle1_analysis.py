"""The paired initialization comparison must preserve offsets and member identity."""

import numpy as np
import pandas as pd
import xarray as xr

from monthly_diffusion.experiments import experiment_isotropic_cycle1 as experiment


def test_identical_arms_have_zero_paired_difference(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "OUTPUT", tmp_path)
    monkeypatch.setattr(experiment, "MODES", ("ENSO",))
    monkeypatch.setattr(experiment, "METHODS", ("isotropic_cycle1", "cycle1"))
    rng = np.random.default_rng(81)
    # A persistent nonzero offset must not disappear in the primary analysis.
    values = rng.normal(size=(360, 5, 1)) + 3
    reference = rng.normal(size=(492, 1))
    ds = xr.Dataset(coords=dict(
        time=pd.date_range("1800-01-01", periods=360, freq="MS"),
        training_time=pd.date_range("1966-01-01", periods=492, freq="MS"),
        method=list(experiment.METHODS), member=np.arange(1, 6), mode=["ENSO"],
    ))
    for kind in ("physical", "atmosphere", "ocean"):
        ds[kind] = (("time", "method", "member", "mode"), np.stack([values, values], axis=1))
        ds[kind + "_reference"] = (("training_time", "mode"), reference)
    metrics = experiment.analyze(ds)
    paired = pd.read_csv(tmp_path / "paired_comparison.csv")
    np.testing.assert_array_equal(paired.difference, 0)
    np.testing.assert_array_equal(paired.paired_bootstrap_low, 0)
    np.testing.assert_array_equal(paired.paired_bootstrap_high, 0)
    assert (metrics.loc[metrics.anomalies == "fixed_training_month", "mean_reference_sd"] > 2).all()
    np.testing.assert_allclose(
        metrics.loc[metrics.anomalies == "own_month", "mean_reference_sd"], 0, atol=1e-12,
    )
    assert set(metrics.member) == {1, 2, 3, 4, 5}
