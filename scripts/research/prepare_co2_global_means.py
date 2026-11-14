"""Compute raw reference global means for the CO2-only regression experiment."""

import json
import sys
from pathlib import Path
from time import perf_counter

import dask
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from monthly_diffusion.preprocessing import mdv1_variables
from monthly_diffusion.rollout import co2_values, validate_months
from workflows.data.sources import open_state_dataset, open_co2

RUN_DIR = Path(__file__).resolve().parents[1] / "runs/test_gridpoint_no_co2"
OUTPUT = RUN_DIR / "analysis/co2_global_means"
START, END = "1850-01", "2100-12"

source = json.loads((RUN_DIR / "run.json").read_text())["source"]
specifications = mdv1_variables()
names = [variable.name for variable in specifications]
dataset = open_state_dataset(source["state_zarr"])[names].sel(time=slice(START, END))
validate_months(dataset.time)
assert len(dataset.time) == 3012
forcing = co2_values(open_co2(source["co2_netcdf"]), dataset.time)
weights = np.cos(np.deg2rad(dataset.lat.values))[None, :, None]
means = np.full((len(dataset.time), len(names)), np.nan, dtype=np.float64)
years = dataset.time.dt.year.values
started = perf_counter()

# Read the ORIGINAL fields, independently of AE reconstructions and fitted masks.
# Each variable uses its own finite cells, with latitude weights summing to one.
for year in np.unique(years):
    indices = np.flatnonzero(years == year)
    with dask.config.set(scheduler="threads", num_workers=2):
        fields = dataset.isel(time=indices).compute()
    for column, name in enumerate(names):
        values = fields[name].transpose("time", "lat", "lon").values.astype("float64")
        valid = np.isfinite(values)
        denominator = (weights * valid).sum(axis=(-2, -1))
        if np.any(denominator == 0):
            raise ValueError(f"No valid cells for {name} in {year}")
        means[indices, column] = (weights * np.where(valid, values, 0)).sum(
            axis=(-2, -1)
        ) / denominator
    if year % 25 == 0 or year == years[-1]:
        print(
            f"Global means through {year}: {perf_counter() - started:.1f} s", flush=True
        )

assert np.isfinite(means).all()
result = xr.Dataset(
    {name: ("time", means[:, i]) for i, name in enumerate(names)},
    coords={"time": dataset.time},
)
result["co2"] = forcing
for variable in specifications:
    result[variable.name].attrs.update(
        units=dataset[variable.name].attrs.get("units", variable.units),
        component=variable.component,
    )
result.attrs.update(
    source=source["state_zarr"],
    co2_source=source["co2_netcdf"],
    spatial_weights="cos(latitude), normalized over finite cells per variable and month",
    targets="Raw monthly physical global means; seasonal cycles retained; no AE masks",
)
OUTPUT.mkdir(parents=True, exist_ok=True)
result.to_netcdf(OUTPUT / "reference_global_means.nc")
result.to_dataframe().to_csv(OUTPUT / "reference_global_means.csv")
dataset.close()
print(OUTPUT / "reference_global_means.nc", flush=True)
