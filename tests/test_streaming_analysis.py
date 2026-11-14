import numpy as np
import pytest
import xarray as xr
from workflows.analysis import internal_variability, teleconnection, temporal_spectrum
from workflows.analysis.long_rollout import RegressionMoments, save_dataset, variable_products


def test_streamed_ols_matches_package_regression_with_masks():
    rng = np.random.default_rng(92)
    time = xr.date_range("1966-01", periods=36, freq="MS")
    x = xr.DataArray(rng.normal(size=(4, 36)), dims=("member", "time"), coords=dict(member=range(4), time=time))
    x = internal_variability(x)
    y = (3*x+7).expand_dims(lat=[0, 30], lon=[1, 2]).transpose("member", "time", "lat", "lon").copy()
    y.values[:, :4, 0, 0] = np.nan
    y.values[:, :, 1, 1] = np.nan
    z = (x-x.mean())/x.std()
    moments = RegressionMoments(1, (2, 2))
    for start in range(0, 36, 7):
        moments.update(y.isel(time=slice(start, start+7)).values, z.isel(time=slice(start, start+7)).values[None])
    np.testing.assert_allclose(moments.result()[0], teleconnection(y, x), rtol=1.e-12, equal_nan=True)


def test_streamed_map_matches_direct_and_global_masks(tmp_path):
    rng = np.random.default_rng(42)
    dates = xr.date_range("2000-01", periods=36, freq="MS")
    y = xr.DataArray(rng.normal(size=(3,36,2,2)), dims=("member","time","lat","lon"),
        coords=dict(member=range(3),time=dates,lat=[0,60],lon=[0,180]), attrs=dict(units="K"))
    y.values[:,:,:,0] = np.nan
    indices = internal_variability(y.mean(("lat","lon"))).to_dataset(name="ENSO")
    class Reader:
        lat, lon = y.lat.values, y.lon.values
        def read(self, variable, block): return y.isel(time=block)
        def blocks(self, selection, size):
            for start in range(0,len(selection),size): yield selection[start:start+size]
    path = variable_products(Reader(),np.arange(36),indices,"sst",
        dict(early=(2000,2000),training=(2001,2001),late=(2002,2002)),tmp_path,7)
    ds = xr.load_dataset(path)
    np.testing.assert_allclose(ds.forced_response, y.sel(time="2002").mean(("time","member"))-y.sel(time="2000").mean(("time","member")),equal_nan=True)
    np.testing.assert_allclose(ds.regression.isel(mode=0),teleconnection(internal_variability(y),indices.ENSO),equal_nan=True)
    assert np.isfinite(ds.global_mean).all()


def test_spectrum_and_netcdf_bool(tmp_path):
    t = np.arange(600)
    index = xr.DataArray(np.sin(2*np.pi*t/36)[None], dims=("member","time"), coords=dict(member=[0],time=t))
    psd = temporal_spectrum(index, nperseg=360)
    assert float(psd.frequency[psd.argmax("frequency").item()]) == pytest.approx(1/36)
    psd.attrs["centered"] = True
    save_dataset(psd.to_dataset(name="psd"),tmp_path/"psd.nc")
    assert xr.load_dataset(tmp_path/"psd.nc").psd.attrs["centered"] == 1


def test_internal_variability_refuses_single_member():
    with pytest.raises(ValueError): internal_variability(xr.DataArray([1.,2.], dims="time"))
    with pytest.raises(ValueError): internal_variability(xr.DataArray([[1.,2.]], dims=("member","time")))


def test_archive_preserves_member_identity_missing_cells_and_derivations(tmp_path):
    from workflows.analysis.archives import ClimateArchive
    dates = xr.date_range("2000-01", periods=3, freq="MS")
    ds = xr.Dataset(coords=dict(time=dates,lat=[0.],lon=[0.,120.,240.]))
    for variable, values, unit in (("sic", [0.,.5,np.nan], "1"),
                                    ("sit", [np.nan,2.,np.nan], "m"),
                                    ("ohc300_700", [10.,20.,np.nan], "J m-2")):
        ds[variable] = (("time","lat","lon"),np.tile(values,(3,1,1)))
        ds[variable].attrs["units"] = unit
    paths = []
    for member in ("r2i1p1f1","r3i1p1f1"):
        ds.attrs["member_id"] = member
        path = tmp_path/f"{member}.zarr"
        ds.to_zarr(path,consolidated=True)
        paths.append(path)
    reader = ClimateArchive(paths)
    result = reader.read("sivol",np.arange(3))
    assert list(result.member.values) == ["r2i1p1f1","r3i1p1f1"]
    np.testing.assert_allclose(result.isel(member=0,time=0).values,[[0.,1.,np.nan]],equal_nan=True)
    xr.testing.assert_allclose(reader.read("ohc700_minus_ohc300",np.arange(3)).isel(member=0,drop=True),ds.ohc300_700)
    with pytest.raises(ValueError,match="Duplicate"):
        ClimateArchive([paths[0],paths[0]])
    with pytest.raises(ValueError,match="contiguous"):
        reader.read("sivol",[0,2])


def test_forecast_shard_wrong_date_is_rejected(tmp_path):
    import json
    from workflows.analysis.archives import ClimateArchive
    dates = xr.date_range("2000-01",periods=2,freq="MS")
    record = dict(months=2,valid_times=[str(t) for t in dates],checkpoint_sha256="abc")
    (tmp_path/"forecast.json").write_text(json.dumps(record))
    for i in range(2):
        # The second shard deliberately contains January a second time.
        ds = xr.Dataset(dict(sst=(("time","member","lat","lon"),np.ones((1,2,1,1)))),
            coords=dict(time=dates[:1],member=[0,1],lat=[0.],lon=[0.]),attrs=dict(checkpoint_sha256="abc"))
        ds.sst.attrs["units"] = "degC"
        ds.to_zarr(tmp_path/f"month-{i+1:06d}.zarr",consolidated=True)
    reader = ClimateArchive([tmp_path],forecast=True)
    assert reader.read("sst",[0]).shape == (2,1,1,1)
    with pytest.raises(ValueError,match="date differs"):
        reader.read("sst",[1])
