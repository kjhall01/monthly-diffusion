from pathlib import Path
import json
import os
import subprocess
import sys
import numpy as np
import pytest
import xarray as xr
import yaml

from workflows.analysis.trajectory_anomalies import TrainingTrajectoryAnomalies
from workflows.analysis.long_rollout import run_physical_diagnostics, variable_products
from workflows.analysis import teleconnection


class ArrayArchive:
    def __init__(self, field):
        self.field = field
        self.lat, self.lon = field.lat.values, field.lon.values
        self.time, self.members = field.time, field.member.values
        self.keys = field.time.dt.year.values*12+field.time.dt.month.values

    def read(self, variable, indices):
        return self.field.isel(time=indices).rename(variable)

    def blocks(self, indices, size):
        for start in range(0, len(indices), size):
            yield indices[start:start+size]


def test_training_anomalies_match_joint_ols_and_ignore_future(tmp_path):
    rng = np.random.default_rng(32)
    dates = xr.date_range("2000-01", periods=72, freq="MS")
    values = rng.normal(size=(2,72,2,2)) + np.arange(72)[None,:,None,None]*.8
    values[:,48:] += 10000  # Must never influence the fit.
    values[:,:,-1,-1] = np.nan
    field = xr.DataArray(values, dims=("member","time","lat","lon"),
        coords=dict(member=[0,1],time=dates,lat=[0,60],lon=[0,120]),name="sst")
    reader = ArrayArchive(field)
    fit = TrainingTrajectoryAnomalies(reader, [2000,2003], tmp_path, 7)
    got = fit.transform(field.isel(time=slice(0,48)))
    design = np.column_stack((np.eye(12)[np.arange(48)%12], np.arange(48)/12))
    for member in range(2):
        y = values[member,:48,0,0]
        expected = y-design@np.linalg.lstsq(design,y,rcond=None)[0]
        np.testing.assert_allclose(got.isel(member=member,lat=0,lon=0),expected,atol=1e-12)
    assert np.isnan(got.isel(lat=-1,lon=-1)).all()
    np.testing.assert_allclose(got.groupby("time.month").mean("time").isel(lat=0,lon=0),0,atol=1e-12)
    assert float(fit.transform(field.isel(time=slice(48,None))).isel(lat=0,lon=0).mean()) > 9990
    # Reuse saved coefficients, including for a regional SST read.
    regional = fit.transform(field.isel(time=slice(0,12),lat=slice(0,1),lon=slice(0,1)))
    xr.testing.assert_allclose(regional,got.isel(time=slice(0,12),lat=slice(0,1),lon=slice(0,1)))


def test_single_trajectory_regression_uses_training_only(tmp_path):
    rng = np.random.default_rng(53)
    dates = xr.date_range("2000-01",periods=60,freq="MS")
    field = xr.DataArray(rng.normal(size=(1,60,2,2)),dims=("member","time","lat","lon"),
        coords=dict(member=["reanalysis"],time=dates,lat=[0,60],lon=[0,180]),name="sst",attrs=dict(units="degC"))
    reader = ArrayArchive(field)
    anomaly = TrainingTrajectoryAnomalies(reader,[2000,2002],tmp_path/"anomalies",5)
    residual = anomaly.transform(field.isel(time=slice(0,36)))
    index = residual.isel(lat=0,lon=0,drop=True).rename("ENSO").to_dataset()
    result = xr.load_dataset(variable_products(reader,np.arange(60),index,"sst",
        dict(training=[2000,2002]),tmp_path,7,anomaly))
    assert "forced_response" not in result
    assert result.global_mean.sizes["time"] == 60
    np.testing.assert_allclose(result.regression.isel(mode=0),teleconnection(residual,index.ENSO),atol=1e-12)
    np.testing.assert_allclose(result.regression_count,36)


def test_reanalysis_configuration_is_matched_and_no_future_truth():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root/"examples/diagnostics-era5-oras5.yaml").read_text())
    source = yaml.safe_load((root/"examples/era5-oras5-local.yaml").read_text())
    assert config["references"] == [source["state_zarr"]]
    assert config["co2_sha256"] == source["co2_sha256"]
    assert config["index_period"] == config["training"] == [1966,2006]
    assert config["early"] == config["training"]
    assert config["reference_evaluation"][1] == 2025
    script = (root/"scripts/recipes/run_era5_oras5_long_rollout.sh").read_text()
    assert "--months 3012 --members 5" in script
    assert "--gaussian-start 1845-01 --spinup-months 60 --hold-first-co2-back-to 1845-01" in script
    assert config["evaluation"] == [1850,2100]
    assert "--device mps" in script and "--require-trained" in script
    assert "--constant-initial-co2" in script


def test_bash_workflow_dispatches_both_paths_without_running_inference(tmp_path):
    root = Path(__file__).resolve().parents[1]
    stub = tmp_path/"python_stub"
    record = tmp_path/"calls.jsonl"
    stub.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                    "with open(os.environ['MD_LAUNCH_LOG'], 'a') as stream:\n"
                    "    stream.write(json.dumps(sys.argv[1:])+'\\n')\n")
    stub.chmod(0o755)
    subprocess.run(["bash",str(root/"scripts/recipes/run_era5_oras5_long_rollout.sh")],check=True,
                   capture_output=True,text=True,
                   env=dict(os.environ,MD_PYTHON=str(stub),MD_LAUNCH_LOG=str(record)))
    calls = [json.loads(line) for line in record.read_text().splitlines()]
    assert calls[0] == ['-', 'both']  # Space preflight; stub prevents any inference.
    calls = calls[1:]
    assert len(calls)==4
    for i in (0,2):
        assert "scripts/legacy/forecast.py" in calls[i] and "--resume" in calls[i]
        assert calls[i][calls[i].index("--device")+1]=="mps"
    assert "--constant-initial-co2" not in calls[0] and "--constant-initial-co2" in calls[2]
    assert "--skip-decoder" not in calls[1] and "--skip-decoder" in calls[3]


def make_small_archives(root):
    rng = np.random.default_rng(92)
    dates = xr.date_range("2000-01",periods=60,freq="MS")
    lat = [-65.,-5.,5.,35.,45.,65.]
    lon = np.arange(5.,360.,10.)
    fields = {}
    for var, units in (("sst","degrees C"),("mslp","Pa"),("u50","m s^-1")):
        values = rng.normal(size=(60,2,len(lat),len(lon))) + np.arange(60)[:,None,None,None]*.1
        if var == "sst": values[:,:,-1,0] = np.nan
        fields[var] = xr.DataArray(values,dims=("time","member","lat","lon"),
            coords=dict(time=dates,member=[0,1],lat=lat,lon=lon),attrs=dict(units=units))
    forecast = xr.Dataset(fields,attrs=dict(checkpoint_sha256="pilot"))
    reference = forecast.isel(member=0,drop=True,time=slice(0,48)).copy(deep=True)
    reference.attrs["member_id"] = "era5_oras5"
    reference.u50.attrs["units"] = "m s-1"  # Canonical notation aliases are allowed.
    ref_path = root/"reference.zarr"
    reference.to_zarr(ref_path,consolidated=True)
    forecast_path = root/"forecast"
    forecast_path.mkdir()
    (forecast_path/"forecast.json").write_text(json.dumps(dict(months=60,valid_times=[str(t) for t in dates],checkpoint_sha256="pilot")))
    for i in range(60):
        forecast.isel(time=slice(i,i+1)).to_zarr(forecast_path/f"month-{i+1:06d}.zarr",consolidated=True)
    return dict(forecast=str(forecast_path),references=[str(ref_path)],reference_kind="reanalysis",
                reference_label="ERA5/ORAS5",anomaly_method="training_monthly_linear_detrend",
                evaluation=[2000,2004],reference_evaluation=[2000,2003],training=[2000,2002],
                index_period=[2000,2002],early=[2000,2002],late=[2004,2004],variables=list(fields),
                sample_start_year=2000,block_months=7,spectrum_segment_months=24)


def test_end_to_end_reanalysis_products_resume_and_identity(tmp_path):
    config = make_small_archives(tmp_path)
    out = tmp_path/"diagnostics"
    run_physical_diagnostics(config,out)
    ref = xr.load_dataset(out/"reference/sst.nc")
    model = xr.load_dataset(out/"model/sst.nc")
    indices = xr.load_dataset(out/"reference/indices.nc")
    assert "forced_response" not in ref
    assert np.isfinite(model.forced_response).any()
    assert indices.sizes["member"] == 1 and indices.sizes["time"] == 36
    assert model.global_mean.sizes["time"] == 60 and ref.global_mean.sizes["time"] == 48
    for variable in config["variables"]:
        ds = xr.load_dataset(out/"model"/f"{variable}.nc")
        assert np.isfinite(ds.regression).any()
    before = (out/"model/sst.nc").stat().st_mtime_ns
    run_physical_diagnostics(config,out)
    assert (out/"model/sst.nc").stat().st_mtime_ns == before
    with pytest.raises(ValueError,match="configuration changed"):
        run_physical_diagnostics(dict(config,sample_start_year=2001),out)
