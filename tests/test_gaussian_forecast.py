from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
import xarray as xr
from monthly_diffusion.forecast import Forecast, rollout
from monthly_diffusion.data.sources import hold_first_co2_back_to


class ToyTransition:
    def __init__(self, fail_at=None):
        self.calls, self.fail_at = 0, fail_at

    def eval(self):
        return self

    def sample(self, calendar, ah, oh, previous_noise):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("simulated interruption")
        c = ah.shape[-1]//2
        a = .8*ah[...,:c]+.1*ah[...,c:]+.1*torch.randn_like(ah[...,:c])
        o = .8*oh[...,:c]+.1*oh[...,c:]+.1*torch.randn_like(oh[...,:c])
        return a,o,torch.zeros_like(a),torch.zeros_like(o)


class ToyModel:
    def __init__(self, fail_at=None, nodes=32):
        self.device = torch.device('cpu')
        self.config = SimpleNamespace(mesh_levels=[2],latent_channels=4,to_dict=lambda:dict(test=True))
        self.geometry = SimpleNamespace(graphs={'mesh_l0':SimpleNamespace(n_nodes=nodes,lat=torch.zeros(nodes),lon=torch.zeros(nodes))})
        self.latent_normalizers = dict(atmosphere=True,ocean=True)
        self.transition = ToyTransition(fail_at)
        self.decoded = 0

    def encode(self, *args, **kwargs):
        raise AssertionError("Gaussian mode must never encode a physical state")

    def weight_digest(self):
        return "toy_checkpoint"

    def decode(self, a, o, *, co2):
        self.decoded += 1
        return tuple(z.mean(('node','channel')).expand_dims(lat=[0.],lon=[0.]).transpose('time','member','lat','lon').to_dataset(name=name)
                     for z,name in ((a,'t2m'),(o,'sst')))


def test_gaussian_histories_independent_and_rng_preserved():
    torch.manual_seed(72)
    before = torch.get_rng_state().clone()
    model = ToyModel(nodes=1000)
    f = Forecast(model,None,None,co2=300.,members=20,seed=123,gaussian_start='1845-01')
    assert torch.equal(before,torch.get_rng_state())
    assert [str(t)[:7] for t in f.times] == ['1844-12','1844-11']
    for h in f.history:
        assert h.shape == (20,2,1000,4)
        assert abs(float(h.mean()))<.02 and abs(float(h.std())-1)<.02
        assert not torch.equal(h[0],h[1])
        assert not torch.equal(h[:,0],h[:,1])
    assert not torch.equal(f.history[0],f.history[1])
    g = Forecast(model,None,None,co2=300.,members=20,seed=123,gaussian_start='1845-01')
    for a,b in zip(f.history,g.history): torch.testing.assert_close(a,b,rtol=0,atol=0)
    a,o = next(f)
    assert str(a.time.values[0])[:7]=='1845-01'
    assert model.decoded == 0


def test_constant_prefix_preserves_every_original_value():
    dates = xr.date_range('1850-01',periods=24,freq='MS')
    source = xr.DataArray(np.linspace(285.,288.,24,dtype='float32'),dims='time',coords=dict(time=dates),attrs=dict(units='ppm'))
    source = source.assign_coords(year=source.time.dt.year,month=source.time.dt.month)
    extended = hold_first_co2_back_to(source,'1845-01')
    assert extended.sizes['time']==84
    np.testing.assert_array_equal(extended.isel(time=slice(0,60)),np.full(60,285.,dtype='float32'))
    np.testing.assert_array_equal(extended.isel(time=slice(60,None)),source)
    np.testing.assert_array_equal(extended.time.values[60:],source.time.values)
    with pytest.raises(ValueError): hold_first_co2_back_to(source,'1851-01')


def test_spinup_not_decoded_or_archived_and_saved_months_resume(tmp_path):
    kwargs = dict(co2=300.,months=3,members=2,seed=72,gaussian_start='1845-01',spinup_months=60)
    model = ToyModel()
    expected = rollout(model,None,None,path=None,resume=False,**kwargs)
    assert model.decoded==3 and str(expected.time.values[0])[:7]=='1850-01'
    broken = ToyModel(fail_at=63)
    with pytest.raises(RuntimeError,match='simulated'):
        rollout(broken,None,None,path=tmp_path,resume=False,**kwargs)
    assert len(list(tmp_path.glob('month-*.zarr')))==2
    model = ToyModel()
    actual = rollout(model,None,None,path=tmp_path,resume=True,**kwargs).compute()
    xr.testing.assert_allclose(expected,actual)
    assert model.decoded==1
    record = json.loads((tmp_path/'forecast.json').read_text())
    assert record['spinup_months']==60 and len(record['valid_times'])==3
    assert record['valid_times'][0].startswith('1850-01')
    assert record['initialization_kind']=='iid_standard_normal'
    initial = torch.load(tmp_path/'initial_state.pt',weights_only=False)
    assert initial['step']==0
    with pytest.raises(ValueError,match='differs'):
        rollout(model,None,None,path=tmp_path,resume=True,**dict(kwargs,seed=73))


def test_interrupted_spinup_resumes_without_physical_shards(tmp_path):
    kwargs = dict(co2=300.,months=2,members=2,seed=72,gaussian_start='1845-01',spinup_months=60)
    broken = ToyModel(fail_at=15)
    with pytest.raises(RuntimeError,match='simulated'):
        rollout(broken,None,None,path=tmp_path,resume=False,**kwargs)
    assert not list(tmp_path.glob('month-*.zarr')) and broken.decoded==0
    assert torch.load(tmp_path/'spinup_state.pt',weights_only=False)['step']==12
    model = ToyModel()
    actual = rollout(model,None,None,path=tmp_path,resume=True,**kwargs).compute()
    expected = rollout(ToyModel(),None,None,path=None,resume=False,**kwargs)
    xr.testing.assert_allclose(actual,expected)
    assert model.transition.calls==50 and model.decoded==2


def test_extend_preserves_prefix_and_exact_restart(tmp_path):
    kwargs = dict(co2=300., members=2, seed=72, gaussian_start='1958-03')
    expected = rollout(ToyModel(), None, None, months=5, path=None, resume=False, **kwargs)
    rollout(ToyModel(), None, None, months=2, path=tmp_path, resume=False, **kwargs)
    original_record = (tmp_path/'forecast.json').read_bytes()
    original_files = {p: (p.read_bytes(), p.stat().st_mtime_ns)
                      for p in (tmp_path/'month-000002.zarr').rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='differs'):
        rollout(ToyModel(), None, None, months=5, path=tmp_path, resume=True, **kwargs)
    for changes in ({'co2': 400.}, {'seed': 73}, {'members': 3}):
        with pytest.raises(ValueError, match='differs'):
            rollout(ToyModel(), None, None, months=5, path=tmp_path, resume=True,
                    extend=True, **dict(kwargs, **changes))
        assert (tmp_path/'forecast.json').read_bytes() == original_record
    model = ToyModel()
    actual = rollout(model, None, None, months=5, path=tmp_path, resume=True, extend=True, **kwargs).compute()
    xr.testing.assert_equal(actual, expected)
    assert model.transition.calls == 3
    assert (tmp_path/'forecast_through_000002.json').read_bytes() == original_record
    for p, (contents, mtime) in original_files.items():
        assert p.read_bytes() == contents and p.stat().st_mtime_ns == mtime
    # --extend remains idempotent when rerunning the same completed horizon.
    rollout(model, None, None, months=5, path=tmp_path, resume=True, extend=True, **kwargs)
    assert model.transition.calls == 3


def test_extend_rejects_missing_original_month(tmp_path):
    kwargs = dict(co2=300., members=2, seed=72, gaussian_start='1958-03')
    with pytest.raises(RuntimeError, match='simulated'):
        rollout(ToyModel(fail_at=2), None, None, months=2, path=tmp_path, resume=False, **kwargs)
    record = (tmp_path/'forecast.json').read_bytes()
    with pytest.raises(ValueError, match='Finish the original'):
        rollout(ToyModel(), None, None, months=5, path=tmp_path, resume=True, extend=True, **kwargs)
    assert (tmp_path/'forecast.json').read_bytes() == record


def test_extend_discards_roundoff_in_reinitialized_history(tmp_path, monkeypatch):
    kwargs = dict(co2=300., members=2, seed=72, gaussian_start='1958-03')
    expected = rollout(ToyModel(), None, None, months=4, path=None, resume=False, **kwargs)
    rollout(ToyModel(), None, None, months=2, path=tmp_path, resume=False, **kwargs)
    original = json.loads((tmp_path/'forecast.json').read_text())
    initialize = Forecast.__init__
    def roundoff(self, *args, **options):
        initialize(self, *args, **options)
        self.history = [h + 1e-7 for h in self.history]
    monkeypatch.setattr(Forecast, '__init__', roundoff)
    actual = rollout(ToyModel(), None, None, months=4, path=tmp_path,
                     resume=True, extend=True, **kwargs).compute()
    # The approximate new IC is discarded, not substituted into the trajectory.
    xr.testing.assert_equal(expected, actual)
    assert json.loads((tmp_path/'forecast.json').read_text())['initialization_sha256'] == original['initialization_sha256']
