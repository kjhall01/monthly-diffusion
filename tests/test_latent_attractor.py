"""Initialization pairing, exact resume, and anomaly/spectrum conventions."""
from types import SimpleNamespace

import numpy as np
import torch
import xarray as xr

from test_gaussian_forecast import ToyModel

from workflows.experiments import diagnose_latent_attractor as audit


def test_fixed_anomalies_preserve_offset_and_window_removal_does_not_mix_members():
    dates=xr.date_range("1959-01",periods=360,freq="MS")
    months=np.asarray(dates.month)-1
    season=np.arange(12.)
    trend=np.linspace(-3,3,360)
    values=season[months,None]+trend[:,None]+np.array([[1.,10.]])
    fixed=audit.anomalies(values,dates,"fixed_training_monthly",season)
    np.testing.assert_allclose(fixed.mean(0),[1.,10.])
    monthly=audit.anomalies(values,dates,"window_monthly")
    assert monthly.std()>.5
    removed=audit.anomalies(values,dates,"window_monthly_detrend")
    np.testing.assert_allclose(removed,0,atol=1.e-12)
    # Adding an offset to only member zero must not alter another member.
    perturbed=values.copy();perturbed[:,0]+=100
    np.testing.assert_allclose(audit.anomalies(perturbed,dates,"window_monthly")[:,1],monthly[:,1])


def test_spectrum_and_acf_do_not_concatenate_members():
    t=np.arange(360)
    x=np.column_stack([np.sin(2*np.pi*t/48),np.cos(2*np.pi*t/48)])
    frequency,psd,acf=audit.curve_statistics(x)
    np.testing.assert_allclose(acf[:,0],1)
    assert (acf[:,24]<-.8).all()
    np.testing.assert_allclose(frequency[np.argmax(psd,axis=1)],1/48)


def test_paired_history_order_noise_and_exact_resume(tmp_path):
    cache=dict(validation_time=np.array(["1958-11","1958-12"],dtype="datetime64[M]"),
        validation={c:torch.stack((torch.ones(32,4),2*torch.ones(32,4))) for c in audit.COMPONENTS})
    model=ToyModel();model.preprocessor=SimpleNamespace(co2_mean=350.)
    audit.paired_rollout(model,cache,tmp_path,years=1,members=2,seed=37)
    initial={k:torch.load(tmp_path/k/"initial.pt",weights_only=False) for k in audit.INITIALS}
    first={k:torch.load(tmp_path/k/"year-001.pt",weights_only=False) for k in audit.INITIALS}
    for c,component in enumerate(audit.COMPONENTS):
        torch.testing.assert_close(initial["encoded"]["history"][c][:,0],2*torch.ones(2,32,4))
        torch.testing.assert_close(initial["encoded"]["history"][c][:,1],torch.ones(2,32,4))
        delta=initial["encoded"]["history"][c]-initial["gaussian"]["history"][c]
        expected=.8*delta[:,0]+.1*delta[:,1]
        actual=first["encoded"]["latents"][component][0]-first["gaussian"]["latents"][component][0]
        np.testing.assert_allclose(actual,expected.numpy(),atol=5.e-7)
    audit.paired_rollout(model,cache,tmp_path,years=2,members=2,seed=37)
    direct=tmp_path/"direct";direct.mkdir()
    fresh=ToyModel();fresh.preprocessor=SimpleNamespace(co2_mean=350.)
    audit.paired_rollout(fresh,cache,direct,years=2,members=2,seed=37)
    for k in audit.INITIALS:
        a=torch.load(tmp_path/k/"year-002.pt",weights_only=False)
        b=torch.load(direct/k/"year-002.pt",weights_only=False)
        np.testing.assert_array_equal(a["time"],b["time"])
        assert str(a["time"][0])[:7]=="1960-01"
        for c in audit.COMPONENTS:np.testing.assert_array_equal(a["latents"][c],b["latents"][c])
    assert model.decoded==fresh.decoded==0
