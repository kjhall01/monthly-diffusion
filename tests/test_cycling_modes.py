import numpy as np
import pandas as pd
import xarray as xr

from workflows.experiments.diagnose_cycling_modes import fixed_month_anomalies, scalar_statistics, index_fields
from workflows.experiments.diagnose_latent_attractor import anomalies
from workflows.experiments import diagnose_cycling_modes as diagnostic
import torch


def test_fixed_months_keep_forecast_offsets_and_spread():
    dates = pd.date_range("1966-01-01",periods=48,freq="MS")
    reference = np.tile(np.arange(12.),4)+np.repeat([-1.,1.,-1.,1.],12)
    forecast = np.broadcast_to(reference[:,None,None,None],(48,2,3,2)).copy()+7.
    x,r = fixed_month_anomalies(forecast,dates,np.repeat(reference[:,None],2,axis=1),dates)
    np.testing.assert_allclose(x.mean(0),7.)
    np.testing.assert_allclose(x.std(0),1.)
    np.testing.assert_allclose(r.mean(0),0.)


def test_statistics_measure_each_trajectory_not_ensemble_mean():
    x = np.sin(2*np.pi*np.arange(240)/24)
    first = scalar_statistics(x,x)
    opposite = scalar_statistics(-x,x)
    assert first["sd_ratio"]==opposite["sd_ratio"]==1.
    assert first["acf12"] < -.9
    assert first["acf24"] > .85
    assert abs(first["spectral_peak_months"]-24.)<1e-10
    assert scalar_statistics(x+10.,x)["mean_reference_sd"]>10.


def test_physical_indices_use_native_definitions_and_units():
    lat=np.array([-2.,2.,25.,40.,65.]); lon=np.array([115.,200.,235.,335.,345.,355.])
    coords=dict(time=pd.date_range("2000",periods=2,freq="MS"),lat=lat,lon=lon)
    a=xr.Dataset({v:(("time","lat","lon"),np.ones((2,5,6))*value)
                  for v,value in (("mslp",100000.),("u50",12.))},coords=coords)
    a["mslp"].loc[dict(lat=40.)]=101000.
    o=xr.Dataset(dict(sst=(("time","lat","lon"),np.ones((2,5,6))*24.)),coords=coords)
    loading=xr.ones_like(o.sst.isel(time=0,drop=True).sel(lat=slice(20,70),lon=slice(110,260)))
    out=index_fields(a,o,loading)
    np.testing.assert_allclose(out.ENSO,24.)
    np.testing.assert_allclose(out.NAO,10.)
    np.testing.assert_allclose(out.QBO,12.)
    assert np.isfinite(out.PDO).all()


def test_archive_methods_selected_by_label_not_position(tmp_path,monkeypatch):
    dates=pd.date_range("1800",periods=12,freq="MS").values
    order=list(reversed(diagnostic.DISPLAY))
    values=np.stack([np.full((12,5,162,4),diagnostic.DISPLAY.index(m),dtype=np.float32) for m in order],axis=1)
    torch.save(dict(time=dates,methods=order,predicted={c:values for c in diagnostic.COMPONENTS}),tmp_path/"year-001.pt")
    monkeypatch.setattr(diagnostic,"ARCHIVES",{m:tmp_path for m in diagnostic.DISPLAY})
    got_dates,got=diagnostic.read_annual_latents(1)
    np.testing.assert_array_equal(got_dates,dates)
    for i in range(5):
        for c in diagnostic.COMPONENTS: np.testing.assert_array_equal(got[c][:,i],i)


def test_monthly_sensitivity_does_not_remove_nonseasonal_trend():
    dates=pd.date_range("1800",periods=240,freq="MS")
    t=np.arange(240)
    values=(10*np.sin(2*np.pi*t/12)+.01*t)[:,None]
    residual=anomalies(values,dates,"window_monthly")[:,0]
    for month in range(1,13):
        np.testing.assert_allclose(residual[dates.month==month].mean(),0.,atol=1e-12)
    assert np.polyfit(t,residual,1)[0]>.009
    detrended=anomalies(values,dates,"window_monthly_detrend")[:,0]
    assert np.std(detrended)<1e-12
