"""Check cycling changes only current history, preserves dates and all fields."""
from types import SimpleNamespace
import numpy as np
import torch
import xarray as xr
from workflows.forecast import Forecast
from test_gaussian_forecast import ToyModel

from workflows.experiments import pilot_cycled_initialization as pilot


class CycleModel(ToyModel):
    def __init__(self):
        super().__init__(nodes=4)
        self.config.latent_channels = 1
        for normalizer in self.latent_normalizers.values():
            normalizer.mean = torch.zeros(1)
        self.preprocessor = SimpleNamespace(ocean_support_mask=None,
            stats={v:dict(mask=np.ones((2, 2), bool)) for v in pilot.VARIABLES})
        self.encodes = 0

    def initialize(self, *, encoded_noise_start=None, gaussian_start=None, **kwargs):
        return Forecast(self, None, None,
            gaussian_start=encoded_noise_start or gaussian_start, **kwargs)

    def decode(self, a, o, *, co2):
        fields = []
        for z, name, offset in ((a, "t2m", 280.), (o, "sst", 10.)):
            values = z.values.reshape(1, z.sizes["member"], 2, 2)+offset
            ds = xr.Dataset({name:(("time", "member", "lat", "lon"), values)},
                coords=dict(time=z.time, member=z.member, lat=[-45.,45.], lon=[0.,180.]))
            ds[name].attrs["units"] = "K" if name == "t2m" else "degC"
            ds["other_"+name] = ds[name]+1
            if name == "t2m":
                ds["mtpr"] = xr.full_like(ds[name], 1.e-5)
                ds.mtpr.attrs["units"] = "kg m-2 s-1"
            fields.append(ds)
        return tuple(fields)

    def encode(self, a, o, *, co2):
        self.encodes += 1
        assert "other_t2m" in a and "other_sst" in o
        return tuple(xr.DataArray((ds[name].values-offset+.5).reshape(1,4,1),
                     dims=("time","node","channel"), coords=dict(time=ds.time))
                     for ds,name,offset in ((a,"t2m",280.),(o,"sst",10.)))


def test_cycle_only_current_slot_and_matched_control(tmp_path):
    model = CycleModel()
    ds, table = pilot.run_pilot(model, co2=284., first="1800-01", seed=32, steps=6, output=tmp_path)
    assert model.encodes == 6
    assert ds.sizes["time"] == 3  # initial + step 5 + final step 6
    np.testing.assert_array_equal(ds.step, [0,5,6])
    np.testing.assert_array_equal(ds.time.values.astype("datetime64[M]"),
                                  np.array(["1799-12","1800-05","1800-06"], dtype="datetime64[M]"))
    payload = torch.load(tmp_path/"latent_states.pt", weights_only=False)
    records = payload["records"]
    for step in range(1,7):
        cycled, control = records[2*step:2*step+2]
        for component in range(2):
            torch.testing.assert_close(cycled["fed_back"][component]-cycled["predicted"][component],
                                        torch.full((1,4,1), .5), atol=3.e-5, rtol=0)
            torch.testing.assert_close(control["fed_back"][component], control["predicted"][component])
    for component in range(2):
        torch.testing.assert_close(records[2]["predicted"][component], records[3]["predicted"][component])
        state = payload["states"]["cycled"]
        torch.testing.assert_close(state["history"][component][:,1], records[-4]["fed_back"][component])
    assert not torch.equal(records[4]["predicted"][0], records[5]["predicted"][0])
    assert (table.mtpr_min > 0).all()
