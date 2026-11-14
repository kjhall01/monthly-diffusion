"""Rescore the ocean AE after excluding cells constant throughout training.

This is an evaluation of fixed predictions, not retraining. Fit the proposed
mask on 1966–2006 only, before the saved standard-deviation floor. Keep seasonal
zeros wherever a cell has any training variability. No model files are changed.
"""

import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import dask
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from monthly_diffusion.checkpoints import load_checkpoint, model_identity
from workflows.data.sources import open_state_dataset
from workflows.run_files import run_inputs

RUN = Path(__file__).resolve().parents[1] / "runs/test_gridpoint_no_transforms"
ARCHIVE = RUN / "ae_reconstruction"
OUTPUT = RUN / "analysis/zero_variance_masks"
PERIODS = {
    "training": (1966, 2006),
    "validation": (1958, 1965),
    "early": (1850, 1880),
    "late": (2070, 2100),
}

started = perf_counter()
setup = load_checkpoint(RUN / "ocean/setup.pt")
selected = load_checkpoint(RUN / "ocean/model.pt")
preprocessor = setup["preprocessor"]
assert preprocessor.centering == "gridpoint"
assert all(v.transform == "identity" for v in preprocessor.variables)
names = [v.name for v in preprocessor.variables]
metadata = json.loads((ARCHIVE / "reconstruction.json").read_text())
identity = model_identity(RUN)
assert metadata["models"] == identity
source, _, training_period, validation_period = run_inputs(RUN)
assert training_period == slice("1966-01", "2006-12")
assert validation_period == slice("1958-01", "1965-12")
assert metadata["source"]["state_zarr"] == str(source)
reference = open_state_dataset(source)[names]
shape = (len(names), reference.sizes["lat"], reference.sizes["lon"])
support = np.stack([preprocessor.stats[n]["mask"].values for n in names])
center = np.stack([preprocessor.stats[n]["center"].values for n in names])
scale = np.stack([preprocessor.stats[n]["std"].values for n in names])
area = np.broadcast_to(
    np.cos(np.deg2rad(reference.lat.values.astype("float64")))[:, None], shape[1:]
)
minimum = np.full(shape, np.inf)
maximum = np.full(shape, -np.inf)
counts = np.zeros(shape, dtype=np.int64)
squares = np.zeros(shape)
zero_counts = np.zeros(shape, dtype=np.int64)

# min == max is the exact constant-series test. Unlike a variance computed
# around a rounded mean, it cannot misclassify a constant nonzero time series.
for year in range(1966, 2007):
    with dask.config.set(scheduler="threads", num_workers=2):
        block = reference.sel(time=str(year)).load()
    for index, name in enumerate(names):
        values = block[name].transpose("time", "lat", "lon").values.astype("float64")
        valid = np.isfinite(values)
        minimum[index] = np.minimum(
            minimum[index], np.where(valid, values, np.inf).min(0)
        )
        maximum[index] = np.maximum(
            maximum[index], np.where(valid, values, -np.inf).max(0)
        )
        counts[index] += valid.sum(0)
        zero_counts[index] += (valid & (values == 0)).sum(0)
        squares[index] += (np.where(valid, values - center[index], 0) ** 2).sum(0)
    if year % 10 == 6 or year == 2006:
        print(
            f"Training variability through {year}: {perf_counter() - started:.1f} s",
            flush=True,
        )

assert np.array_equal(counts >= 2, support)
constant = support & (minimum == maximum)
varying = support & ~constant
raw_std = np.sqrt(squares / np.maximum(counts, 1))
raw_std[constant] = 0
floor = np.array([preprocessor.stats[n]["std_floor"] for n in names])[:, None, None]
np.testing.assert_allclose(
    scale[support], np.maximum(raw_std, floor)[support], rtol=1e-8, atol=1e-12
)
support_rows = []
for index, name in enumerate(names):
    total_area = (area * support[index]).sum()
    removed_area = (area * constant[index]).sum()
    support_rows.append(
        dict(
            variable=name,
            supported_cells=int(support[index].sum()),
            constant_cells=int(constant[index].sum()),
            constant_zero_cells=int((constant[index] & (minimum[index] == 0)).sum()),
            constant_area_percent=100 * removed_area / total_area,
            retained_error_weight_multiplier=total_area / (total_area - removed_area),
            raw_zero_gridpoint_month_percent=100
            * zero_counts[index][support[index]].sum()
            / counts[index][support[index]].sum(),
            raw_zero_area_month_percent=100
            * (area * zero_counts[index] * support[index]).sum()
            / (area * counts[index] * support[index]).sum(),
            std_floor=float(floor[index, 0, 0]),
        )
    )
    print(
        f"{name}: constant area {100 * removed_area / total_area:.2f}%; retained weights x{total_area / (total_area - removed_area):.3f}",
        flush=True,
    )

OUTPUT.mkdir(parents=True, exist_ok=True)
mask_dataset = xr.Dataset(
    {
        "original_support": (("variable", "lat", "lon"), support.astype("int8")),
        "constant_training": (("variable", "lat", "lon"), constant.astype("int8")),
        "varying_training": (("variable", "lat", "lon"), varying.astype("int8")),
        "unfloored_std": (
            ("variable", "lat", "lon"),
            np.where(support, raw_std, np.nan),
        ),
    },
    coords={"variable": names, "lat": reference.lat, "lon": reference.lon},
    attrs={
        "training_period": "1966-01 through 2006-12",
        "definition": "constant iff min == max over finite training values; at least two observations",
    },
)
mask_dataset.to_netcdf(OUTPUT / "masks.nc")
metrics = []
monthly = []
for period, (first, last) in PERIODS.items():
    changed_cells = np.zeros(shape, dtype=bool)
    for year in range(first, last + 1):
        with xr.open_zarr(
            ARCHIVE / f"year-{year:04d}.zarr", consolidated=True
        ) as shard:
            with dask.config.set(scheduler="threads", num_workers=2):
                observed = reference.sel(time=str(year)).load()
                predicted = shard[names].load()
        observed, predicted = xr.align(observed, predicted, join="exact")
        assert observed.sizes["time"] == 12
        for index, name in enumerate(names):
            target = (
                observed[name].transpose("time", "lat", "lon").values.astype("float64")
            )
            prediction = (
                predicted[name].transpose("time", "lat", "lon").values.astype("float64")
            )
            valid = np.isfinite(target) & support[index]
            assert np.isfinite(prediction[valid]).all()
            error = np.where(valid, prediction - target, 0)
            physical_error = error**2
            normalized_error = physical_error / scale[index] ** 2
            changed = valid & constant[index] & (target != minimum[index])
            changed_cells[index] |= changed.any(0)
            masks = {
                "all": valid,
                "varying": valid & varying[index],
                "constant": valid & constant[index],
            }
            for month, time in enumerate(observed.time.values):
                row = dict(period=period, time=str(time)[:10], variable=name)
                for label, mask in masks.items():
                    weights = area * mask[month]
                    denominator = weights.sum()
                    row[f"{label}_area"] = denominator
                    for space, errors in (
                        ("physical", physical_error),
                        ("normalized", normalized_error),
                    ):
                        numerator = (weights * errors[month]).sum()
                        row[f"{label}_{space}_mse"] = (
                            numerator / denominator if denominator else np.nan
                        )
                        # Contribution uses the original full support denominator.
                        row[f"{label}_{space}_contribution"] = (
                            numerator / (area * valid[month]).sum()
                        )
                row["constant_cells_changed_area_percent"] = (
                    100 * (area * changed[month]).sum() / row["all_area"]
                )
                monthly.append(row)
        if (year - first) % 10 == 0 or year == last:
            print(
                f"Rescored {period} through {year}: {perf_counter() - started:.1f} s",
                flush=True,
            )
    for index, name in enumerate(names):
        rows = [r for r in monthly if r["period"] == period and r["variable"] == name]
        mean = {
            key: float(np.mean([r[key] for r in rows]))
            for key in rows[0]
            if key not in ("period", "time", "variable")
        }
        result = dict(period=period, variable=name, months=len(rows))
        for space in ("physical", "normalized"):
            original = mean[f"all_{space}_mse"]
            masked = mean[f"varying_{space}_mse"]
            removed = mean[f"constant_{space}_contribution"]
            np.testing.assert_allclose(
                original, removed + mean[f"varying_{space}_contribution"], rtol=1e-12
            )
            result.update(
                {
                    f"original_{space}_rmse": np.sqrt(original),
                    f"masked_{space}_rmse": np.sqrt(masked),
                    f"original_{space}_mse": original,
                    f"masked_{space}_mse": masked,
                    f"constant_{space}_error_percent": 100 * removed / original,
                    f"{space}_mse_change_percent": 100 * (masked / original - 1),
                }
            )
        result["excluded_cells_ever_changed_area_percent"] = (
            100 * (area * changed_cells[index]).sum() / (area * support[index]).sum()
        )
        result["excluded_cells_changed_area_month_percent"] = mean[
            "constant_cells_changed_area_percent"
        ]
        metrics.append(result)

# The unmodified validation metric must reproduce the selected epoch's log.
with (RUN / "ocean/history.csv").open() as stream:
    history = next(
        row for row in csv.DictReader(stream) if int(row["epoch"]) == selected["epoch"]
    )
checks = []
for result in metrics:
    if result["period"] != "validation":
        continue
    name = result["variable"]
    for space, key in (
        ("physical", f"val_rmse_{name}"),
        ("normalized", f"val_normalized_rmse_{name}"),
    ):
        actual, expected = result[f"original_{space}_rmse"], float(history[key])
        np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=1e-7)
        checks.append(
            dict(variable=name, space=space, audit_rmse=actual, history_rmse=expected)
        )

for filename, rows in (
    ("support.csv", support_rows),
    ("metrics.csv", metrics),
    ("monthly_metrics.csv", monthly),
    ("history_checks.csv", checks),
):
    with (OUTPUT / filename).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
(OUTPUT / "provenance.json").write_text(
    json.dumps(
        dict(
            run=str(RUN),
            archive=str(ARCHIVE),
            selected_ocean_epoch=selected["epoch"],
            models=identity,
            training_period=[training_period.start, training_period.stop],
            periods=PERIODS,
            definition=mask_dataset.attrs["definition"],
            method="Rescore unchanged physical predictions, using saved identity-transform stds; per-month cosine-latitude weights normalized over each variable's retained finite cells.",
            limitation="No retraining; numerical loss changes do not establish an improvement in reconstruction skill.",
            original_validation_matches_history=True,
        ),
        indent=2,
    )
    + "\n"
)
reference.close()
print(f"Saved audit to {OUTPUT}; {perf_counter() - started:.1f} s", flush=True)
