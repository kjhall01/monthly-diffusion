"""Cycling cutoffs, exact shared prefixes, and resumability without RNG resets."""
import numpy as np
import torch
import xarray as xr
from test_cycled_initialization import CycleModel

from workflows.experiments import experiment_cycled_spinup as original
from workflows.experiments import experiment_local_moment_spinup as local
from workflows.experiments import experiment_cycle_duration as duration


def test_duration_cutoffs_common_prefix_and_resume(tmp_path,monkeypatch):
    indices=lambda a,o:xr.Dataset(dict(nino34=o.sst.mean(("lat","lon")),pressure_difference=a.t2m.mean(("lat","lon"))))
    for module in (original,local,duration): monkeypatch.setattr(module,"physical_indices",indices)
    cache=dict(validation_time=np.array(["1958-11","1958-12"],dtype="datetime64[M]"),
               validation={c:torch.zeros(2,4,1) for c in original.COMPONENTS})
    base=dict(co2_ppm=284.,members=2,seed=44,first="1800-01",cycle_months=30)
    old=tmp_path/"old"; old.mkdir()
    matched=tmp_path/"matched"; matched.mkdir()
    original.run_experiment(CycleModel(),cache,old,years=2,members=2,seed=44,first="1800-01",ppm=284.)
    initial=CycleModel().initialize(co2=284.,members=2,seed=44,gaussian_start="1800-01").state_dict()
    local.run_experiment(CycleModel(),initial,base,old,matched,2)
    whole=tmp_path/"whole"; whole.mkdir()
    resumed=tmp_path/"resumed"; resumed.mkdir()
    duration.run_experiment(CycleModel(),initial,base,matched,whole,2)
    duration.run_experiment(CycleModel(),initial,base,matched,resumed,1)
    duration.run_experiment(CycleModel(),initial,base,matched,resumed,2)
    for year in (1,2):
        a=torch.load(whole/f"year-{year:03d}.pt",weights_only=False)
        b=torch.load(resumed/f"year-{year:03d}.pt",weights_only=False)
        origin=torch.load(matched/f"year-{year:03d}.pt",weights_only=False)
        for stage in ("predicted","fed_back"):
            for c in original.COMPONENTS:
                np.testing.assert_array_equal(a[stage][c],b[stage][c])
                if year==1:
                    np.testing.assert_array_equal(a[stage][c][0,0],origin[stage][c][0,1])
                    np.testing.assert_array_equal(a[stage][c][:,1],origin[stage][c][:,1])
        np.testing.assert_array_equal(a["physical"],b["physical"])
        if year==1:
            assert (a["correction"][0,0]>0).all()
            assert (a["correction"][1:,0]==0).all()
            assert (a["correction"][:,1]>0).all()
            np.testing.assert_array_equal(a["physical"][:,1],origin["physical"][:,1])
        else: assert (a["correction"]==0).all()
