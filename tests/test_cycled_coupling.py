"""Fixed anomaly baselines and synchronization must not hide level errors."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location("cycled_coupling",
    Path(__file__).resolve().parents[1]/"scripts/diagnose_cycled_coupling.py")
analysis=importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_fixed_monthly_baseline_preserves_offset():
    dates=pd.date_range("1966-01",periods=120,freq="MS")
    ref=np.column_stack((dates.month,2*dates.month)).astype(float)
    pilot=ref[:30]+[3.,-2.]
    clean,r=analysis.fixed_anomalies(pilot,dates[:30],ref,dates)
    np.testing.assert_allclose(r,0.)
    np.testing.assert_allclose(clean,np.broadcast_to([3.,-2.],clean.shape))


def test_perfectly_correlated_offset_still_penalized():
    rng=np.random.default_rng(1)
    x=rng.normal(size=200)
    ref=np.column_stack((x,.8*x+.2*rng.normal(size=200)))
    shifted=ref+[0.,2.]
    a,b=analysis.pair_metrics(ref,ref),analysis.pair_metrics(shifted,ref)
    np.testing.assert_allclose(a["correlation"],b["correlation"])
    np.testing.assert_allclose(b["relationship_bias"],2.)
    assert b["relationship_rmse_ratio"]>8
    np.testing.assert_allclose(b["relationship_centered_scatter_ratio"],1.)


def test_positive_lag_is_ocean_leading_atmosphere():
    x=np.random.default_rng(2).normal(size=300)
    y=np.r_[np.zeros(3),x[:-3]]
    curve=analysis.lag_curve(np.column_stack((x,y)))
    assert np.argmax(curve)-6==3
    np.testing.assert_allclose(curve[9],1.)
