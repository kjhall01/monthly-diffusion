"""Practical agreement metrics are not tests for nonsignificant differences."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from diagnose_isotropic_cycling_sweep import features, differences


def test_identical_series_and_level_shift():
    rng=np.random.default_rng(201)
    x=rng.normal(size=(480,6))
    dates=pd.date_range("1800-01-01",periods=480,freq="MS").values
    grid=np.linspace(-7,7,2049)
    f=features(x,dates,grid)
    weights=rng.multinomial(6,np.ones(6)/6,size=50)/6
    d=differences(f,f,weights,weights,grid[1]-grid[0])
    for metric,v in d.items():
        np.testing.assert_allclose(v,1 if metric=="sd_ratio" else 0,atol=1e-14)
    shifted=features(x+2,dates,grid)
    d=differences(shifted,f,weights,weights,grid[1]-grid[0])
    assert np.all(d["mean"]>1.5)
    assert np.all(d["wasserstein"]>1.5)
    np.testing.assert_allclose(d["sd_ratio"],1,atol=1e-14)
    np.testing.assert_allclose(d["acf_rmse"],0,atol=1e-14)


def test_amplitude_change_does_not_change_normalized_spectrum():
    rng=np.random.default_rng(202)
    dates=pd.date_range("1800-01-01",periods=480,freq="MS").values
    x=rng.normal(size=(480,4))
    grid=np.linspace(-10,10,2049)
    weights=np.full((1,4),.25)
    d=differences(features(2*x,dates,grid),features(x,dates,grid),weights,weights,grid[1]-grid[0])
    np.testing.assert_allclose(d["sd_ratio"],2)
    np.testing.assert_allclose(d["acf_rmse"],0)
    np.testing.assert_allclose(d["spectral_cdf_distance"],0)


def test_pooled_start_seasons_preserve_actual_calendar():
    jan=np.tile(np.arange(1,13),40)
    july=np.tile(np.r_[np.arange(7,13),np.arange(1,7)],40)
    months=np.column_stack((jan,july))
    values=np.sin(2*np.pi*(months-1)/12)
    f=features(values,months,np.linspace(-2,2,257))
    expected=np.sin(2*np.pi*np.arange(12)/12)
    np.testing.assert_allclose(f["monthly"],np.tile(expected,(2,1)),atol=1e-14)


def test_confirmation_pool_keeps_independent_cohort_and_window(tmp_path, monkeypatch):
    import xarray as xr
    import diagnose_isotropic_cycling_sweep as diagnostic
    folders=[]
    for label,start in (("january","1800-01-01"),("july","1800-07-01")):
        folder=tmp_path/label/"physical"
        folder.mkdir(parents=True)
        ds=xr.Dataset({"physical":(("time","member"),np.ones((24,2)))},
                      coords={"time":pd.date_range(start,periods=24,freq="MS"),"member":[0,1]})
        ds.to_netcdf(folder/"mode_trajectories.nc")
        folders.append(folder)
    calls=[]
    monkeypatch.setattr(diagnostic,"compare",lambda ds,folder,windows: calls.append(windows))
    monkeypatch.setattr(diagnostic,"plot",lambda *args,**kwargs: None)
    pooled=diagnostic.pool_confirmation(folders,tmp_path/"pooled",windows={"year2":(1,2)},
                  plot_window=(1,2),cohort_names=["january","july"])
    assert calls==[{"year2":(1,2)}]
    assert list(pooled.cohort.values)==["january","january","july","july"]
    np.testing.assert_array_equal(pooled.calendar_month.isel(time=0),[1,1,7,7])
