"""Initialization moments, exact history order, mask support and paired restart."""
from types import SimpleNamespace
import numpy as np
import torch
import xarray as xr
from workflows.forecast import Forecast
from test_encoded_noise_forecast import NoiseModel
from test_cycled_initialization import CycleModel

from workflows.experiments import experiment_local_moment_spinup as experiment
from workflows.experiments import experiment_cycled_spinup as previous


def test_monthly_population_moments_and_constant_cells():
    times = np.arange("1966-01", "1968-01", dtype="datetime64[M]")
    values = np.tile(np.arange(12),2)[:,None,None]+np.repeat([0.,4.],12)[:,None,None]
    x = xr.DataArray(np.broadcast_to(values,(24,2,2)).copy(),dims=("time","lat","lon"),coords=dict(time=times))
    x[:,0,1] = 7.; x[:,1,0] = np.nan; x[:12,1,1] = np.nan
    out = experiment.monthly_moments(x)
    np.testing.assert_array_equal(out["mean"][:,0,0],np.arange(12)+2.)
    np.testing.assert_array_equal(out["std"][:,0,0],np.full(12,2.))
    assert (out["std"][:,0,1]==0).all()
    assert np.isnan(out["mean"][:,1,:]).all()
    assert (out["count"][:,1,1]==1).all()


def test_fit_excludes_nontraining_dates(tmp_path,monkeypatch):
    times=np.arange("1965-01","1969-01",dtype="datetime64[M]").astype("datetime64[ns]")
    training=times[12:36]
    values=np.full((48,2,2),1.e8)
    values[12:24]=2.; values[24:36]=6.
    source=xr.Dataset({"x":(("time","lat","lon"),values)},coords=dict(time=times))
    monkeypatch.setattr(experiment,"open_canonical_state",lambda *a,**k:(source,source,None))
    model=SimpleNamespace(names=dict(atmosphere=["x"],ocean=[]),
        preprocessor=SimpleNamespace(training_times=training,encode=lambda ds,c:ds))
    experiment.fit_moments(model,dict(state_zarr="fake"),tmp_path,training)
    result=xr.load_dataset(tmp_path/"monthly_moments/x.nc")
    assert (result["count"]==2).all()
    assert (result["mean"]==4).all()
    assert (result["std"]==2).all()


def test_affine_sampling_exact_epsilon_months_masks_and_normalization(tmp_path):
    model = NoiseModel()
    model.initialize = lambda **kw: Forecast(model,None,None,**kw)
    folder = tmp_path/"monthly_moments"; folder.mkdir()
    for names in model.names.values():
        for name in names:
            mu = np.broadcast_to(np.arange(1,13,dtype="float32")[:,None,None],(12,4,8)).copy()
            sd = np.full_like(mu,.2); sd[:,1,1]=0
            xr.Dataset(dict(mean=(("month","lat","lon"),mu),std=(("month","lat","lon"),sd)),
                coords=dict(month=np.arange(1,13))).to_netcdf(folder/f"{name}.nc")
    base=dict(co2_ppm=280.,members=2,seed=44,first="1800-01")
    state=experiment.matched_history(model,base,tmp_path)
    assert [str(t)[:7] for t in state["times"]]==["1799-12","1799-11"]
    for ci,component in enumerate(model.names):
        rng=np.random.default_rng(1044+ci)
        for mi,(values,months,force,mask) in enumerate(getattr(model.autoencoder,component).calls):
            eps=rng.standard_normal((2,2,4,8)).astype("float32")
            expected=np.array([12.,11.],dtype="float32")[:,None,None,None]+.2*eps
            expected[:,:,1,1]=np.array([12.,11.])[:,None]
            valid=np.stack([model.preprocessor.stats[n]["mask"].values for n in model.names[component]])
            if component=="ocean": valid &= model.preprocessor.ocean_support_mask.values[None]
            expected[:,~valid]=np.nan
            np.testing.assert_allclose(values,expected,rtol=0,atol=0)
            raw=values.nanmean((1,2,3))+months[:,0]+force[:,0]
            torch.testing.assert_close(state["history"][ci][mi,:,0,0],(raw-.2)/2)
            np.testing.assert_array_equal(mask,np.broadcast_to(valid[0],(2,4,8)))


def test_matched_rollout_switch_and_exact_resume(tmp_path,monkeypatch):
    physical=lambda a,o:xr.Dataset(dict(nino34=o.sst.mean(("lat","lon")),pressure_difference=a.t2m.mean(("lat","lon"))))
    monkeypatch.setattr(previous,"physical_indices",physical)
    monkeypatch.setattr(experiment,"physical_indices",physical)
    cache=dict(validation_time=np.array(["1958-11","1958-12"],dtype="datetime64[M]"),
        validation={c:torch.zeros(2,4,1) for c in previous.COMPONENTS})
    baseline=tmp_path/"baseline"; baseline.mkdir()
    base=dict(co2_ppm=284.,members=2,seed=44,first="1800-01",cycle_months=6)
    previous.run_experiment(CycleModel(),cache,baseline,years=2,members=2,seed=44,first="1800-01",ppm=284.,cycle_months=6)
    initial=CycleModel().initialize(co2=284.,members=2,seed=44,gaussian_start="1800-01").state_dict()
    for h in initial["history"]: h.add_(2.)
    a=tmp_path/"complete"; a.mkdir(); b=tmp_path/"resumed"; b.mkdir()
    experiment.run_experiment(CycleModel(),initial,base,baseline,a,2)
    experiment.run_experiment(CycleModel(),initial,base,baseline,b,1)
    experiment.run_experiment(CycleModel(),initial,base,baseline,b,2)
    for year in (1,2):
        x=torch.load(a/f"year-{year:03d}.pt",weights_only=False)
        y=torch.load(b/f"year-{year:03d}.pt",weights_only=False)
        for stage in ("predicted","fed_back"):
            for c in experiment.COMPONENTS: np.testing.assert_array_equal(x[stage][c],y[stage][c])
        np.testing.assert_array_equal(x["physical"],y["physical"])
        assert (x["correction"][:,0]==0).all()
        if year==1:
            assert (x["correction"][:6,1]>0).all()
            assert (x["correction"][6:,1]==0).all()
        else: assert (x["correction"][:,1]==0).all()
