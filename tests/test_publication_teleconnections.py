"""Per-degree Niño3.4 plots must not mistake +1 SD for +1 degree Celsius."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import xarray as xr

from monthly_diffusion.analysis.climate import internal_variability, teleconnection

path = Path(__file__).parents[1]/"scripts/render_publication_teleconnections.py"
spec = importlib.util.spec_from_file_location("publication_teleconnections", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_products(root):
    for source, scale, n in (("reference", 1., 4), ("model", 3., 5)):
        (root/source).mkdir()
        rng = np.random.default_rng(n)
        index = xr.DataArray(scale*rng.normal(size=(n, 24)), dims=("member", "time"),
            coords=dict(member=np.arange(n), time=np.arange("2000-01", "2002-01", dtype="datetime64[M]")),
            name="ENSO", attrs=dict(units="degC"))
        ds = index.to_dataset()
        ds.attrs["anomaly_definition"] = "sqrt(N/(N-1))*(field - exact same-month ensemble mean); no monthly re-centering"
        ds.to_netcdf(root/source/"indices.nc")
        slope = xr.DataArray([[2., -1.], [0., 3.]], dims=("lat", "lon"), coords=dict(lat=[-20, 20], lon=[0, 180]))
        for variable, factor in (("t2m", 1.), ("mtpr", 86400.)):
            # Spatially varying intercepts and missing field samples: still
            # divide by the GLOBAL index SD, not a per-cell subset SD.
            fields = (280.+7.*slope+index*slope)/factor
            fields.loc[dict(member=0, lat=20)] = np.nan
            regression = teleconnection(fields, index).expand_dims(mode=["ENSO"])
            regression.attrs["units"] = module.STYLE[variable][2]
            counts = np.isfinite(fields).sum(("member", "time")).expand_dims(mode=["ENSO"])
            xr.Dataset(dict(regression=regression, regression_count=counts),
                coords=dict(member=index.member, time=index.time),
                attrs=dict(regression="pooled OLS with intercept; native units per +1 SD")).to_netcdf(root/source/f"{variable}.nc")
    return slope


def test_each_ensemble_uses_own_sd_units_and_masked_intercept_fit(tmp_path):
    slope = make_products(tmp_path)
    ref, model, indices, counts, metadata = module.load_teleconnections(tmp_path, ["t2m", "mtpr"])
    for variable in ("t2m", "mtpr"):
        np.testing.assert_allclose(ref[variable], slope, atol=1e-12)
        np.testing.assert_allclose(model[variable], slope, atol=1e-12)
        np.testing.assert_allclose(model[variable]-ref[variable], 0., atol=1e-12)
    assert metadata["ensembles"]["model"]["index_std_degC"] != metadata["ensembles"]["reference"]["index_std_degC"]
    assert indices["reference"].sizes == {"member": 4, "time": 24}
    assert int(counts["model"].t2m.sel(lat=20).min()) == 4*24


def test_different_index_regression_dates_rejected(tmp_path):
    make_products(tmp_path)
    path = tmp_path/"model/t2m.nc"
    ds = xr.load_dataset(path)
    ds.attrs["regression_end"] = "2001-11-01"
    ds.to_netcdf(path)
    with pytest.raises(ValueError, match="time intervals differ"):
        module.load_teleconnections(tmp_path, ["t2m", "mtpr"])


def test_finite_ensemble_correction_cancels_in_per_degree_slope():
    rng = np.random.default_rng(12)
    index = xr.DataArray(rng.normal(size=(5, 36)), dims=("member", "time"))
    fields = 3.25*index + xr.DataArray(np.arange(36), dims="time")
    corrected_index, corrected_fields = internal_variability(index), internal_variability(fields)
    per_sd = teleconnection(corrected_fields, corrected_index)
    np.testing.assert_allclose(per_sd/corrected_index.std(), 3.25, atol=1e-12)


def test_requested_per_degree_colorbar_labels():
    assert module.STYLE["t2m"][1] == "°C / °C Niño3.4"
    assert module.STYLE["mtpr"][1] == "mm day⁻¹ / °C Niño3.4"
