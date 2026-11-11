"""Public one-cycle initializer: local moments, masks, ordering, RNG and resume."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import xarray as xr

from monthly_diffusion import MonthlyDiffusionV1
from monthly_diffusion.forecast import Forecast, rollout
from test_encoded_noise_forecast import NoiseModel


class StochasticModel(NoiseModel):
    def __init__(self, fail_at=None):
        super().__init__(fail_at=fail_at)
        self.projected = 0
        self.preprocessor.training_times = np.arange("1966-01","1968-01",dtype="datetime64[M]")
        self.initialization_moments = {}
        for names in self.names.values():
            for name in names:
                mean = np.broadcast_to(np.arange(1,13,dtype="float32")[:,None,None],(12,4,8)).copy()
                sd = np.full_like(mean,.2)
                sd[:,1,1] = 0
                # Monthly support, not just the ever-valid training mask.
                mean[11,1,2] = np.nan
                self.initialization_moments[name] = xr.Dataset(
                    dict(mean=(("month","lat","lon"),mean),std=(("month","lat","lon"),sd),
                         count=(("month","lat","lon"),np.full_like(mean,2,dtype="int16"))),
                    coords=dict(month=np.arange(1,13)))

    def encode(self, a, o, *, co2):
        self.projected += 1
        return tuple(xr.DataArray((ds[name].values.reshape(len(ds.time),1,1)+.5).repeat(32,1).repeat(4,2),
            dims=("time","node","channel"),coords=dict(time=ds.time)) for ds,name in ((a,"t2m"),(o,"sst")))


def test_exact_local_draws_and_masks():
    model = StochasticModel()
    before = torch.get_rng_state().clone()
    f = MonthlyDiffusionV1.initialize(model,co2=280.,members=2,seed=44,stochastic_start="1800-01")
    torch.testing.assert_close(torch.get_rng_state(),before,rtol=0,atol=0)
    assert [str(t)[:7] for t in f.times]==["1799-12","1799-11"]
    for ci,c in enumerate(model.names):
        rng = np.random.default_rng(1044+ci)
        for j,(x,months,forcing,mask) in enumerate(getattr(model.autoencoder,c).calls):
            eps = rng.standard_normal((2,2,4,8)).astype("float32")
            mean = np.stack([model.initialization_moments[n]["mean"].sel(month=[12,11]).values for n in model.names[c]],1)
            sd = np.stack([model.initialization_moments[n]["std"].sel(month=[12,11]).values for n in model.names[c]],1)
            valid = np.stack([model.preprocessor.stats[n]["mask"].values for n in model.names[c]])
            if c=="ocean":valid &= model.preprocessor.ocean_support_mask.values[None]
            valid = valid[None]&np.isfinite(mean)&np.isfinite(sd)
            np.testing.assert_array_equal(x,np.where(valid,mean+sd*eps,np.nan))
            np.testing.assert_array_equal(mask,valid[:,0])
            expected = (x.nanmean((1,2,3))+months[:,0]+forcing[:,0]-.2)/2.
            torch.testing.assert_close(f.history[ci][j,:,0,0],expected)


def test_one_cycle_matches_manual_experiment_and_restart():
    model = StochasticModel()
    f = Forecast(model,None,None,co2=280.,members=2,seed=44,stochastic_start="1800-01")
    control_model = StochasticModel()
    control = Forecast(control_model,None,None,co2=280.,members=2,seed=44,stochastic_start="1800-01")
    control.cycle_months = 0  # Identical moment-matched start, manual projection below.
    previous = [h[:,0].clone() for h in f.history]
    outside_rng = torch.get_rng_state().clone()
    got = next(f)
    expected = next(control)
    for a,b in zip(got,expected):xr.testing.assert_equal(a,b)
    fields = control_model.decode(*expected,co2=280.)
    projected = [[],[]]
    for member in range(2):
        for ci,z in enumerate(control_model.encode(*(ds.isel(member=member,drop=True) for ds in fields),co2=280.)):
            projected[ci].append(z.values[0])
    for ci in range(2):
        control.history[ci][:,0]=torch.tensor(np.stack(projected[ci]))
        torch.testing.assert_close(f.history[ci],control.history[ci],rtol=0,atol=0)
        torch.testing.assert_close(f.history[ci][:,1],previous[ci],rtol=0,atol=0)
        torch.testing.assert_close(f.noise[ci],control.noise[ci],rtol=0,atol=0)
    torch.testing.assert_close(outside_rng,torch.get_rng_state(),rtol=0,atol=0)
    saved = copy.deepcopy(f.state_dict())
    assert saved["cycle_months"]==1 and saved["step"]==1
    restart_model = StochasticModel()
    restarted = Forecast(restart_model,None,None,co2=280.,members=2,seed=44,stochastic_start="1800-01")
    restarted.load_state_dict(saved)
    for month in range(2,7):
        got,manual,resumed = next(f),next(control),next(restarted)
        assert str(got[0].time.values[0])[:7]==f"1800-{month:02d}"
        for a,b,c in zip(got,manual,resumed):
            xr.testing.assert_equal(a,b);xr.testing.assert_equal(a,c)
    assert model.projected==2 and restart_model.projected==0


@pytest.mark.parametrize("spinup",[0,12])
def test_rollout_resume_and_moment_identity(tmp_path,spinup):
    kw=dict(co2=280.,months=3,members=2,seed=44,stochastic_start="1800-01",spinup_months=spinup)
    fail_at = spinup+3
    with pytest.raises(RuntimeError,match="simulated"):
        rollout(StochasticModel(fail_at),None,None,path=tmp_path, resume=False,**kw)
    assert len(list(tmp_path.glob("month-*.zarr")))==2
    resumed = StochasticModel()
    result = rollout(resumed,None,None,path=tmp_path,resume=True,**kw).compute()
    expected = rollout(StochasticModel(),None,None,path=None,resume=False,**kw)
    xr.testing.assert_equal(result,expected)
    assert resumed.projected==0 and resumed.transition.calls==1
    info = json.loads((tmp_path/"forecast.json").read_text())
    assert info["cycle_months"]==1
    assert info["initialization_kind"]=="encoded_local_monthly_noise_cycle1"
    modified = StochasticModel()
    # Changing an unused history month still changes the recorded moments identity.
    modified.initialization_moments["o1"]["mean"].loc[dict(month=6)]+=1
    with pytest.raises(ValueError,match="differs"):
        rollout(modified,None,None,path=tmp_path,resume=True,**kw)


def test_missing_moments_and_conflicting_initializers():
    with pytest.raises(ValueError,match="fit_stochastic_initialization"):
        Forecast(NoiseModel(),None,None,co2=280.,members=1,seed=1,stochastic_start="1800-01")
    with pytest.raises(ValueError,match="one noise initializer"):
        Forecast(StochasticModel(),None,None,co2=280.,members=1,seed=1,stochastic_start="1800-01",gaussian_start="1800-01")
    with pytest.raises(ValueError,match="does not accept physical"):
        Forecast(StochasticModel(),object(),None,co2=280.,members=1,seed=1,stochastic_start="1800-01")
    with pytest.raises(ValueError,match="lacks"):
        Forecast(StochasticModel(),None,None,members=1,seed=1,stochastic_start="1800-01",
            co2=xr.DataArray([280.],dims="time",coords=dict(time=[np.datetime64("1800-01-01")])) )


def test_fitting_selects_only_saved_training_dates_and_preserves_zero_sd():
    dates=np.arange("1965-01","1969-01",dtype="datetime64[M]")
    values=np.full((48,2,2),1e8,dtype="float32");values[12:24]=2.;values[24:36]=6.
    values[12:36,0,1]=7.;values[:,1,0]=np.nan;values[12:24,1,1]=np.nan
    a=xr.Dataset(dict(x=(("time","lat","lon"),values)),coords=dict(time=dates))
    model=SimpleNamespace(preprocessor=SimpleNamespace(stats={"x":{}},training_times=dates[12:36],
        source_attrs={"atmosphere":{},"ocean":{}},encode=lambda ds,c:ds),
        names={"atmosphere":["x"],"ocean":[]},_validate_pair=lambda a,o:None)
    before=torch.get_rng_state().clone()
    MonthlyDiffusionV1.fit_stochastic_initialization(model,a,a)
    s=model.initialization_moments["x"]
    np.testing.assert_array_equal(s["mean"][:,0,0],4.)
    np.testing.assert_array_equal(s["std"][:,0,0],2.)
    np.testing.assert_array_equal(s["std"][:,0,1],0.)
    assert np.isnan(s["mean"][:,1,:]).all()
    torch.testing.assert_close(before,torch.get_rng_state(),rtol=0,atol=0)
