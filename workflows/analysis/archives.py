"""Bounded eager reads from canonical member stores and monthly forecast shards.

Only requested variables/months are decoded. No Dask graph spans a rollout.
The source adapter's OHC rename and SIC*SIT equivalent thickness are applied
after slicing, so unavailable ocean cells remain NaN.
"""
from functools import lru_cache
from pathlib import Path
import json
import hashlib
import numpy as np
import xarray as xr
import zarr


class ClimateArchive:
    def __init__(self, paths, *, forecast=False, ocean_support=None, ocean_variables=()):
        self.paths = [Path(p).resolve() for p in paths]
        self.forecast = forecast
        self.ocean_support = ocean_support
        self.ocean_variables = tuple(ocean_variables)
        self.stores = []
        if forecast:
            self.record = json.loads((self.paths[0]/"forecast.json").read_text())
            self.shards = sorted(self.paths[0].glob("month-*.zarr"))
            expected = [f"month-{i:06d}.zarr" for i in range(1, self.record["months"]+1)]
            if [p.name for p in self.shards] != expected:
                raise ValueError("Forecast is incomplete or has a missing monthly shard")
            first = xr.open_zarr(self.shards[0], chunks=None, consolidated=True)
            self.time = xr.DataArray(np.asarray(self.record["valid_times"], dtype="datetime64[ns]"), dims="time")
            self.members = first.member.values
        else:
            self.record = {}
            first = xr.open_zarr(self.paths[0], chunks=None, consolidated=True)
            self.time = first.time.load().copy()
            self.members = []
            for path in self.paths:
                with xr.open_zarr(path, chunks=None, consolidated=True) as ds:
                    if not ds.time.equals(first.time) or not ds.lat.equals(first.lat) or not ds.lon.equals(first.lon):
                        raise ValueError(f"Reference coordinates disagree: {path}")
                    self.members.append(ds.attrs["member_id"])
                self.stores.append(zarr.open_consolidated(str(path), mode="r"))
            if len(set(self.members)) != len(self.members):
                raise ValueError("Duplicate reference member IDs")
        self.lat, self.lon = first.lat.values.copy(), first.lon.values.copy()
        self.units = {name: da.attrs.get("units", "") for name, da in first.data_vars.items()}
        self.units.update(sivol="m", ohc700_minus_ohc300=self.units.get("ohc300_700", "J m-2"))
        first.close()
        if forecast and "ocean_support" in self.record:
            spec = self.record["ocean_support"]
            if spec["policy"] != "training_sst" or spec["file"] != "ocean_support.nc":
                raise ValueError("Unknown archived ocean-support policy")
            support = xr.load_dataset(self.paths[0]/spec["file"]).ocean_support
            digest = hashlib.sha256(np.asarray(support, dtype=np.uint8).tobytes()).hexdigest()
            if digest != spec["sha256"]:
                raise ValueError("Archived ocean-support checksum changed")
            self.ocean_support = support.astype(bool)
            self.ocean_variables = tuple(spec["variables"])
        if self.ocean_support is not None:
            if (self.ocean_support.dims != ("lat", "lon")
                    or not np.array_equal(self.ocean_support.lat, self.lat)
                    or not np.array_equal(self.ocean_support.lon, self.lon)):
                raise ValueError("Ocean support grid differs from archive")
            if not self.ocean_variables:
                raise ValueError("Specify variables governed by the ocean support")
        self.time = self.time.assign_coords(time=self.time.values)
        self.keys = self.time.dt.year.values*12 + self.time.dt.month.values
        if np.any(np.diff(self.keys) != 1):
            raise ValueError("Archive dates must be consecutive unique months")

    @lru_cache(maxsize=128)
    def shard(self, index):
        group = zarr.open_consolidated(str(self.shards[index]), mode="r")
        if group.attrs["checkpoint_sha256"] != self.record["checkpoint_sha256"]:
            raise ValueError("Mixed checkpoints in forecast archive")
        if (not np.array_equal(group["member"][:], self.members)
                or not np.array_equal(group["lat"][:], self.lat)
                or not np.array_equal(group["lon"][:], self.lon)):
            raise ValueError("Forecast member IDs or grid changed between monthly shards")
        encoded = group["time"]
        date = xr.coding.times.decode_cf_datetime(encoded[:], encoded.attrs["units"],
            calendar=encoded.attrs.get("calendar", "standard"))[0]
        if date != self.time.values[index]:
            raise ValueError(f"Forecast shard date differs from its record: {self.shards[index]}")
        return group

    def read(self, variable, indices, *, lat=slice(None), lon=slice(None)):
        indices = np.asarray(indices, dtype=int)
        if not len(indices) or np.any(np.diff(indices) != 1):
            raise ValueError("Read a nonempty contiguous block of months")
        if self.forecast:
            pieces = []
            for i in indices:
                group = self.shard(int(i))
                values = group[variable][0, :, lat, lon]
                pieces.append(values)
            values = np.stack(pieces, axis=1)
        else:
            source = "ohc300_700" if variable == "ohc700_minus_ohc300" else variable
            selection = (slice(indices[0], indices[-1]+1), lat, lon)
            pieces = []
            for group in self.stores:
                if variable == "sivol":
                    c, h = group["sic"][selection], group["sit"][selection]
                    value = np.where(c > 1.e-6, c*h, 0.)
                    value[~np.isfinite(c)] = np.nan
                else:
                    value = group[source][selection]
                pieces.append(value)
            values = np.stack(pieces)
        if self.ocean_support is not None and variable in self.ocean_variables:
            support = self.ocean_support.isel(lat=lat, lon=lon).values
            values = np.where(support[None, None], values, np.nan)
        return xr.DataArray(values.astype("float64"), dims=("member", "time", "lat", "lon"),
            coords=dict(member=self.members, time=self.time.values[indices], lat=self.lat[lat], lon=self.lon[lon]),
            name=variable, attrs=dict(units=self.units.get(variable, "")))

    def blocks(self, indices, size=12):
        for start in range(0, len(indices), size):
            yield np.asarray(indices[start:start+size])
