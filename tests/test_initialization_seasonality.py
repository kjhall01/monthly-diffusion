"""Keep mean-state tests separate from descriptive anomaly diagnostics."""
import numpy as np
import pandas as pd
import xarray as xr

from monthly_diffusion.experiments import diagnose_initialization_seasonality as diagnostic


def test_own_month_trend_and_ensemble_subtraction(tmp_path,monkeypatch):
    dates=pd.date_range("1800-01",periods=480,freq="MS")
    rng=np.random.default_rng(205)
    values=np.repeat(rng.normal(size=(480,1,4,4))*.1,3,axis=1)
    months=dates.month.to_numpy()
    values+=np.sin(2*np.pi*months[:,None,None,None]/12)*np.arange(1,5)[None,None,:,None]
    values+=np.linspace(-1,1,480)[:,None,None,None]
    ds=xr.Dataset({c:(("time","method","member","mode"),values.copy()) for c in ("atmosphere","ocean")},
        coords=dict(time=dates,method=["observed","cycle12","cycle60"],member=range(4),mode=["ENSO","PDO","NAO","QBO"]))
    ds.to_netcdf(tmp_path/"mode_trajectories.nc")
    plotted={}
    monkeypatch.setattr(diagnostic,"plot",lambda ds,path,window: plotted.update({path.name:ds}))
    diagnostic.run(tmp_path,window=(0,40))
    own=plotted["own_month_and_trend"].ocean.values
    design=np.column_stack((np.eye(12)[months-1],np.linspace(-1,1,480)))
    np.testing.assert_allclose(design.T@own.reshape(480,-1),0,atol=1e-10)
    ensemble=plotted["ensemble_internal"].ocean
    np.testing.assert_allclose(ensemble.mean("member"),0,atol=1e-14)
    expected=np.sqrt(4/3)*(ds.ocean-ds.ocean.mean("member"))
    xr.testing.assert_allclose(ensemble,expected)
    # Fixed-baseline values were not altered to make mean-state agreement look better.
    xr.testing.assert_identical(plotted["fixed_training_month"].ocean,ds.ocean)
    scores=pd.read_csv(tmp_path/"seasonality_sensitivity/own_month_and_trend/agreement_metrics.csv")
    assert set(scores.metric)=={"sd_ratio","acf_rmse","spectral_cdf_distance","enso_coupling_difference"}
    assert scores.interval_pass.all()  # The synthetic methods are identical.
