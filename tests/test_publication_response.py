"""The publication pass must change presentation, not response definitions."""
import json
import numpy as np
import pytest
import xarray as xr

from workflows.experiments import render_publication_forced_response as module


def make_products(root):
    for source, scale, members in (("reference", 1., 4), ("model", .5, 5)):
        (root/source).mkdir()
        for name, units, value in (("t2m", "K", 4.), ("mtpr", "kg m-2 s-1", 2/86400)):
            response = xr.DataArray(np.full((3, 4), value*scale), dims=("lat", "lon"),
                coords=dict(lat=[-60, 0, 60], lon=[45, 135, 225, 315]), attrs=dict(units=units))
            ds = xr.Dataset(dict(forced_response=response,
                climatology=xr.concat([xr.zeros_like(response), response], dim=xr.IndexVariable("period", ["early", "late"]))),
                coords=dict(member=np.arange(members)), attrs=dict(periods=json.dumps(dict(early=[1851, 1880], late=[2070, 2100]))))
            ds.to_netcdf(root/source/f"{name}.nc")


def test_retained_periods_units_and_difference(tmp_path):
    make_products(tmp_path)
    ref, model, metadata = module.load_responses(tmp_path, ["t2m", "mtpr"])
    np.testing.assert_allclose(ref.t2m, 4.)  # NOT 4 - 273.15
    np.testing.assert_allclose(ref.mtpr, 2.)
    np.testing.assert_allclose(model.t2m-ref.t2m, -2.)
    np.testing.assert_allclose(model.mtpr-ref.mtpr, -1.)
    assert metadata["periods"]["early"] == [1851, 1880]
    assert metadata["ensembles"]["reference"]["count"] == 4
    assert metadata["ensembles"]["model"]["count"] == 5


def test_retained_response_is_checked_against_climatologies(tmp_path):
    make_products(tmp_path)
    path = tmp_path/"model/t2m.nc"
    ds = xr.load_dataset(path)
    ds.forced_response.values += 1.
    ds.to_netcdf(path)
    with pytest.raises(AssertionError):
        module.load_responses(tmp_path, ["t2m", "mtpr"])


def test_six_lettered_panels_and_physical_colorbar_labels(tmp_path):
    make_products(tmp_path)
    ref, model, _ = module.load_responses(tmp_path, ["t2m", "mtpr"])
    display = module.plot_response_matrix(ref, model, variables=["t2m", "mtpr"],
        labels=dict(t2m="2-meter Temperature", mtpr="Precipitation"),
        units=dict(t2m="°C", mtpr="mm day⁻¹"), path=tmp_path/"figure", dpi=60)
    assert display["column_titles"] == ["E3SMv2", "MDv1-E3SMv2", "Difference"]
    assert display["title"] is None
    assert [p["panel"] for p in display["panels"]] == [f"{c})" for c in "abcdef"]
    assert [p["variable"] for p in display["panels"]] == ["t2m"]*3+["mtpr"]*3
    assert display["panels"][0]["colorbar_label"] == "2-meter Temperature (°C)"
    assert display["panels"][3]["colorbar_label"] == "Precipitation (mm day⁻¹)"
    assert display["panels"][0]["limit"] == display["panels"][1]["limit"]
    assert (tmp_path/"figure.png").exists() and (tmp_path/"figure.pdf").exists()
