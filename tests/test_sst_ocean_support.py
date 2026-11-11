"""SST support is fitted on training only and shared by targets and evaluation."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
import xarray as xr
from monthly_diffusion import MonthlyDiffusionV1, ModelConfig, Variable
from monthly_diffusion.data.preprocessing import Preprocessor
from monthly_diffusion.data.dataloader import PhysicalStates, co2_values
from monthly_diffusion.modules.autoencoder import reconstruction_mse
from monthly_diffusion.analysis.archives import ClimateArchive
from monthly_diffusion.analysis.spatial import weighted_mean


def with_extra_ocean_cells(tiny_data):
    a, o, s, c, config = tiny_data
    o = o.copy(deep=True)
    o["ohc300"] = o.sst.fillna(1.e8).copy()
    o.ohc300.attrs["units"] = "J m^-2"
    o.ohc300.values[:, 0, 1] = np.nan  # extra variable-specific hole inside SST
    o["sic"] = xr.zeros_like(o.sst).fillna(.7)
    o.sic.attrs["units"] = "dimensionless"
    # Future-only SST availability must not expand the trained support.
    o.sst.values[12:, :, 0] = 20.
    config = replace(config, ocean_mask_aware=True, ocean_support="sst",
        variables=config.variables+[Variable("ohc300", "ocean", "J m^-2", weight=.1),
                                     Variable("sic", "ocean", "dimensionless", "logit", eps=1.e-4, weight=.1)])
    return a, o, s, c, config


def test_common_support_precedes_stats_preserves_nan_and_real_zeros(tiny_data):
    a, o, s, c, config = with_extra_ocean_cells(tiny_data)
    original = o.copy(deep=True)
    pre = Preprocessor(config.variables, ocean_support="sst")
    pre.fit(a.isel(time=slice(12)), o.isel(time=slice(12)), s, c.isel(time=slice(12)))
    wet = np.isfinite(o.sst.isel(time=slice(12))).any("time")
    manual = Preprocessor(config.variables)
    manual.fit(a.isel(time=slice(12)), o.isel(time=slice(12)).where(wet), s, c.isel(time=slice(12)))
    for spec in config.variables:
        xr.testing.assert_equal(pre.stats[spec.name]["center"], manual.stats[spec.name]["center"])
        assert pre.stats[spec.name]["std"] == manual.stats[spec.name]["std"]
    xr.testing.assert_equal(pre.encode(a,"atmosphere"),manual.encode(a,"atmosphere"))
    scaled = pre.encode(o,"ocean")
    assert all(np.isnan(scaled[v].isel(lon=0)).all() for v in scaled)
    assert np.isnan(scaled.ohc300.isel(lat=0,lon=1)).all()
    assert np.isfinite(scaled.sic.isel(lon=2)).all()  # real ice-free cells retained
    assert (pre.mask_ocean(o).sic.isel(lon=2) == 0).all()
    xr.testing.assert_equal(original,o)  # source dataset untouched
    ds = PhysicalStates(a,o,pre,co2_values(c,a.time))
    target = ds[0]["ocean"][None]
    assert torch.isnan(target[:,:,:,0]).all()
    prediction = torch.nan_to_num(target).clone().requires_grad_()
    with torch.no_grad():
        prediction[:,:,:,0] = 1.e12
    loss = reconstruction_mse(prediction,target,torch.tensor(s.lat.values)).sum()
    assert loss.item() == 0.
    loss.backward()
    assert prediction.grad[:,:,:,0].count_nonzero() == 0
    # Old pickled objects without the new attributes must keep legacy behavior.
    del manual.ocean_support, manual.ocean_support_mask
    assert np.isfinite(manual.encode(o,"ocean").ohc300.isel(lon=0)).all()


def test_new_run_only_changes_support_and_keeps_legacy_config_identity():
    root = Path(__file__).resolve().parents[1]
    old = ModelConfig.from_yaml(root/"examples/model-era5-oras5.yaml")
    new = ModelConfig.from_yaml(root/"examples/model-era5-oras5-sst-support.yaml")
    values = new.to_dict()
    assert values.pop("ocean_support") == "sst"
    assert values == old.to_dict()
    assert "ocean_support" not in old.to_dict()
    with pytest.raises(ValueError,match="ocean_support"):
        ModelConfig(ocean_support="unknown")
    launcher = (root/"scripts/train_era5_oras5_sst_support.sh").read_text()
    assert "runs/mdv1-era5-oras5-sst-support" in launcher
    assert "--device mps --threads 4 --load-in-memory" in launcher


def test_staged_training_forecast_and_reference_share_saved_mask(tiny_data,tmp_path):
    a,o,s,c,config = with_extra_ocean_cells(tiny_data)
    model = MonthlyDiffusionV1(statics=s,config=config,device="cpu")
    model.train(a.isel(time=slice(12)),o.isel(time=slice(12)),
        validation=(a.isel(time=slice(12,None)),o.isel(time=slice(12,None))),
        co2=c,directory=tmp_path/"run")
    restored = MonthlyDiffusionV1.load(tmp_path/"run/model.pt",device="cpu")
    xr.testing.assert_equal(model.preprocessor.ocean_support_mask,restored.preprocessor.ocean_support_mask)
    forecast_path = tmp_path/"forecast"
    decoded = restored.rollout(a.isel(time=slice(2)),o.isel(time=slice(2)),co2=c,
        months=2,members=2,seed=55,path=forecast_path).compute()
    for v in ("sst","ohc300","sic"):
        assert np.isnan(decoded[v].isel(lon=0)).all()
    assert np.isnan(decoded.ohc300.isel(lat=0,lon=1)).all()
    before = (forecast_path/"month-000001.zarr/.zmetadata").stat().st_mtime_ns
    restored.rollout(a.isel(time=slice(2)),o.isel(time=slice(2)),co2=c,
        months=2,members=2,seed=55,path=forecast_path,resume=True,open_result=False)
    assert (forecast_path/"month-000001.zarr/.zmetadata").stat().st_mtime_ns == before
    forecast = ClimateArchive([forecast_path],forecast=True)
    # The source retains the huge values that are outside model support.
    source = xr.merge([a,o],compat="no_conflicts")
    source.attrs["member_id"] = "test"
    source_path = tmp_path/"reference.zarr"
    source.to_zarr(source_path,consolidated=True)
    reference = ClimateArchive([source_path],ocean_support=forecast.ocean_support,
                               ocean_variables=forecast.ocean_variables)
    raw = ClimateArchive([source_path])
    assert np.isfinite(raw.read("ohc300",np.arange(2)).isel(lon=0)).all()
    field = reference.read("ohc300",np.arange(2))
    assert np.isnan(field.isel(lon=0)).all()
    expected = o.ohc300.isel(time=slice(2)).where(model.preprocessor.ocean_support_mask)
    xr.testing.assert_allclose(weighted_mean(field).isel(member=0,drop=True),weighted_mean(expected))
    xr.testing.assert_equal(reference.read("t2m",np.arange(2)),raw.read("t2m",np.arange(2)))
    regional = reference.read("ohc300",np.arange(2),lat=slice(2,4),lon=slice(0,3))
    assert np.isnan(regional.isel(lon=0)).all()
    # A corrupted archived mask must not silently change verification support.
    support = xr.load_dataset(forecast_path/"ocean_support.nc")
    support.ocean_support.values[0,0] = 1
    support.to_netcdf(forecast_path/"ocean_support.nc",mode="w")
    with pytest.raises(ValueError,match="checksum"):
        ClimateArchive([forecast_path],forecast=True)
