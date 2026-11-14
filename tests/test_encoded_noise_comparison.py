"""Do not mistake a latent spatial climatology or bias for temporal spread."""

import numpy as np

from workflows.experiments import compare_encoded_noise as audit


def test_identical_latents_have_unit_spread_and_zero_mean_pattern_error():
    real = np.random.default_rng(6).normal(size=(12, 7, 4))
    metrics = audit.monthly_metrics(real, real)
    assert metrics["std_ratio"] == 1.
    assert metrics["mean_pattern_rmse"] == 0.


def test_fixed_spatial_pattern_does_not_inflate_temporal_variability():
    real = np.random.default_rng(7).normal(size=(20, 8, 4))
    fixed = np.arange(32).reshape(8, 4)*100.
    baseline = audit.monthly_metrics(real, real+3.)
    result = audit.monthly_metrics(real+fixed, real+fixed+3.)
    for key in ("std_ratio", "reference_temporal_rms_std", "mean_pattern_rmse"):
        np.testing.assert_allclose(result[key], baseline[key], rtol=1.e-11)
    np.testing.assert_allclose(result["mean_pattern_rmse"], 3.)


def test_collapsed_noise_spread_not_hidden_by_offset():
    real = np.random.default_rng(8).normal(size=(10, 8, 4))
    candidate = real.mean(0)+.1*(real-real.mean(0))+2.
    result = audit.monthly_metrics(real, candidate)
    np.testing.assert_allclose(result["std_ratio"], .1)
    np.testing.assert_allclose(result["mean_pattern_rmse"], 2.)
