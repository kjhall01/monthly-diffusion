"""Paired multi-member cycling stops cleanly and resumes without RNG changes."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import torch
import xarray as xr
from test_cycled_initialization import CycleModel

scripts=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location("cycle_experiment",scripts/"experiment_cycled_spinup.py")
experiment=importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_paired_switch_off_and_resume(tmp_path,monkeypatch):
    monkeypatch.setattr(experiment,"physical_indices",lambda a,o:xr.Dataset(dict(
        nino34=o.sst.mean(("lat","lon")),pressure_difference=a.t2m.mean(("lat","lon")))))
    cache=dict(validation_time=np.array(["1958-11","1958-12"],dtype="datetime64[M]"),
               validation={c:torch.stack([torch.full((4,1),7.),torch.full((4,1),8.)])
                           for c in experiment.COMPONENTS})
    kwargs=dict(cache=cache,members=2,seed=44,first="1800-01",ppm=284.,cycle_months=6)
    complete=tmp_path/"complete"; complete.mkdir()
    resumed=tmp_path/"resumed"; resumed.mkdir()
    experiment.run_experiment(CycleModel(),output=complete,years=2,**kwargs)
    experiment.run_experiment(CycleModel(),output=resumed,years=1,**kwargs)
    experiment.run_experiment(CycleModel(),output=resumed,years=2,**kwargs)
    for year in (1,2):
        a=torch.load(complete/f"year-{year:03d}.pt",weights_only=False)
        b=torch.load(resumed/f"year-{year:03d}.pt",weights_only=False)
        np.testing.assert_array_equal(a["time"],b["time"])
        for stage in ("predicted","fed_back"):
            for component in experiment.COMPONENTS:
                np.testing.assert_array_equal(a[stage][component],b[stage][component])
        np.testing.assert_array_equal(a["physical"],b["physical"])
        if year==1:
            for component in experiment.COMPONENTS:
                np.testing.assert_array_equal(a["fed_back"][component][:6,1],a["fed_back"][component][:6,2])
                assert not np.array_equal(a["fed_back"][component][6:,1],a["fed_back"][component][6:,2])
            assert (a["correction"][:6,1]>0).all()
            assert (a["correction"][6:,1]==0).all()
        else:
            assert (a["correction"][:,1]==0).all()
        assert (a["correction"][:,2]>0).all()
        initial=torch.load(complete/"initial_states.pt",weights_only=False)
        for h in initial["observed"]["history"]:
            torch.testing.assert_close(h[:,0],torch.full_like(h[:,0],8.))
            torch.testing.assert_close(h[:,1],torch.full_like(h[:,1],7.))
