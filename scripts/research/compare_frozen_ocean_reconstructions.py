"""Compare saved frozen and current ocean AE reconstructions on common cells.

The frozen archive uses a float decoder mask, unlike its Boolean training mask.
These scores describe the archived inference behavior, not a clean comparison
of the trained networks. See archive_mask_audit.json beside the results.
"""

import csv
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import dask
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workflows.data.sources import open_state_dataset
from monthly_diffusion.rollout import validate_months

RUN = Path(__file__).resolve().parents[1] / "runs/test_gridpoint_no_co2"
FROZEN = Path("/Users/kyle/Dev/monthly-diffusion/output/umd-production")
CURRENT = RUN / "ae_reconstruction_ocean_fixed"
OUTPUT = RUN / "analysis/ocean_frozen_comparison"

frozen_info = json.loads((FROZEN / "trajectory_metadata.json").read_text())
current_info = json.loads((CURRENT / "reconstruction.json").read_text())
assert frozen_info["status"] == "complete"
assert frozen_info["sources"]["state_zarr"] == current_info["source"]["state_zarr"]
with Path(frozen_info["checkpoint"]).open("rb") as stream:
    assert (
        hashlib.file_digest(stream, "sha256").hexdigest()
        == frozen_info["checkpoint_sha256"]
    )
names = current_info["variables"]["ocean"]
assert names == frozen_info["variables"]["ocean"]
reference = open_state_dataset(current_info["source"]["state_zarr"])[names]
reference = reference.sel(time=slice("1850-01", "2100-12"))
validate_months(reference.time)
assert reference.sizes["time"] == frozen_info["months"] == 3012
np.testing.assert_array_equal(
    reference.time.values, np.asarray(current_info["times"], dtype="datetime64[ns]")
)
weights = np.cos(np.deg2rad(reference.lat.values.astype("float64")))[None, :, None]
means = np.full((3012, len(names), 3), np.nan)
rmse = np.full((3012, len(names), 2), np.nan)
bias = np.full_like(rmse, np.nan)
valid_cells = np.zeros((3012, len(names)), dtype=np.int64)
extra_exclusions = np.zeros(len(names), dtype=np.int64)
offset = 0
started = perf_counter()

# Frozen files contain up to two years. Read each file once and retain only
# this short block of physical fields; no model inference is necessary.
for path in sorted((FROZEN / "ocean/ae_trajectory").glob("reconstruction_*.nc")):
    with xr.open_dataset(path) as frozen:
        assert frozen.attrs["checkpoint_sha256"] == frozen_info["checkpoint_sha256"]
        assert (
            frozen.attrs["autoencoder_epoch"]
            == frozen_info["selected_autoencoder"]["epoch"]
        )
        count = frozen.sizes["time"]
        selection = slice(offset, offset + count)
        np.testing.assert_array_equal(
            frozen.time.values, reference.time.values[selection]
        )
        with dask.config.set(scheduler="threads", num_workers=2):
            observed = reference.isel(time=selection).load()
        blocks = []
        for year in np.unique(frozen.time.dt.year.values):
            with xr.open_zarr(
                CURRENT / f"year-{year:04d}.zarr", consolidated=True
            ) as shard:
                with dask.config.set(scheduler="threads", num_workers=2):
                    blocks.append(shard[names].load())
        current = xr.concat(blocks, dim="time").sel(time=frozen.time)
        observed, frozen, current = xr.align(observed, frozen, current, join="exact")
        for column, name in enumerate(names):
            values = [
                ds[name].transpose("time", "lat", "lon").values.astype("float64")
                for ds in (observed, frozen, current)
            ]
            valid = (
                np.isfinite(values[0]) & np.isfinite(values[1]) & np.isfinite(values[2])
            )
            extra_exclusions[column] += np.count_nonzero(
                np.isfinite(values[0]) & ~valid
            )
            valid_cells[selection, column] = valid.sum(axis=(-2, -1))
            area = (weights * valid).sum(axis=(-2, -1))
            assert np.all(area > 0)
            for index, field in enumerate(values):
                means[selection, column, index] = (
                    weights * np.where(valid, field, 0)
                ).sum(axis=(-2, -1)) / area
            for index, prediction in enumerate(values[1:]):
                error = np.where(valid, prediction - values[0], 0)
                bias[selection, column, index] = (weights * error).sum(
                    axis=(-2, -1)
                ) / area
                rmse[selection, column, index] = np.sqrt(
                    (weights * error**2).sum(axis=(-2, -1)) / area
                )
        offset += count
    if offset % 300 == 0 or offset == 3012:
        print(
            f"Compared {offset}/3012 months: {perf_counter() - started:.1f} s",
            flush=True,
        )

assert offset == 3012
assert all(np.isfinite(values).all() for values in (means, rmse, bias))
result = xr.Dataset(
    data_vars={
        "global_mean": (("time", "variable", "dataset"), means),
        "rmse": (("time", "variable", "model"), rmse),
        "bias": (("time", "variable", "model"), bias),
        "valid_cells": (("time", "variable"), valid_cells),
    },
    coords={
        "time": reference.time,
        "variable": names,
        "dataset": ["reference", "frozen", "current"],
        "model": ["frozen", "current"],
        "units": (
            "variable",
            [reference[name].attrs.get("units", "") for name in names],
        ),
    },
    attrs={
        "definition": "Raw physical reconstruction errors; cos(latitude) weights on common finite cells per variable/month",
        "frozen_checkpoint": frozen_info["checkpoint"],
        "frozen_checkpoint_sha256": frozen_info["checkpoint_sha256"],
        "frozen_epoch": frozen_info["selected_autoencoder"]["epoch"],
        "frozen_archive": str(FROZEN),
        "current_archive": str(CURRENT),
        "current_models": json.dumps(current_info["models"], sort_keys=True),
        "source": current_info["source"]["state_zarr"],
        "frozen_archive_warning": "Frozen public decode used a float32 ocean mask; training used Boolean. The archived reconstruction does not reproduce training behavior. See archive_mask_audit.json.",
    },
)
OUTPUT.mkdir(parents=True, exist_ok=True)
result.to_netcdf(OUTPUT / "monthly_metrics.nc")
summary = []
for period, start, end in [
    ("full", "1850", "2100"),
    ("validation", "1958", "1965"),
    ("late_century", "2070", "2100"),
]:
    selected = result.sel(time=slice(start, end))
    for name in names:
        item = selected.sel(variable=name)
        errors = np.sqrt((item.rmse**2).mean("time"))
        mean_errors = np.sqrt((item.bias**2).mean("time"))
        summary.append(
            dict(
                period=period,
                variable=name,
                units=str(item.units.values),
                frozen_rmse=float(errors.sel(model="frozen")),
                current_rmse=float(errors.sel(model="current")),
                rmse_change_percent=float(
                    100 * (errors.sel(model="current") / errors.sel(model="frozen") - 1)
                ),
                frozen_bias=float(item.bias.sel(model="frozen").mean()),
                current_bias=float(item.bias.sel(model="current").mean()),
                frozen_global_mean_rmse=float(mean_errors.sel(model="frozen")),
                current_global_mean_rmse=float(mean_errors.sel(model="current")),
            )
        )
with (OUTPUT / "summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)
response = result.global_mean.sel(time=slice("2070", "2100")).mean(
    "time"
) - result.global_mean.sel(time=slice("1850", "1880")).mean("time")
response.to_dataframe(name="forced_response").to_csv(OUTPUT / "forced_response.csv")
print(
    "Extra excluded reference cells over all months:",
    dict(zip(names, extra_exclusions.tolist())),
    flush=True,
)
for row in summary:
    print(row, flush=True)
print(OUTPUT, flush=True)
