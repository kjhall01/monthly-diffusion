"""Audit annual/calendar-month support and variability without retraining.

Use training data only to identify constant gridpoints and constant
calendar-month/gridpoint combinations. Rescore unchanged AE predictions.
For SSRD, also inspect polar summer and a training-defined near-dark mask.
"""

import csv
import json
import os
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
OLD = RUN.parent / "test_gridpoint"
OUTPUT = RUN / "analysis/atmosphere_masks"
PERIODS = {"training": (1966, 2006), "validation": (1958, 1965), "late": (2070, 2100)}

started = perf_counter()
pre = load_checkpoint(RUN / "atmosphere/setup.pt")["preprocessor"]
saved = load_checkpoint(RUN / "atmosphere/model.pt")
identity = model_identity(RUN)
assert (
    json.loads((RUN / "ae_reconstruction/reconstruction.json").read_text())["models"]
    == identity
)
assert pre.centering == "gridpoint" and all(
    v.transform == "identity" for v in pre.variables
)
source, _, train, validation = run_inputs(RUN)
reference = open_state_dataset(source)
names = [v.name for v in pre.variables]
area = np.broadcast_to(
    np.cos(np.deg2rad(reference.lat.values.astype(float)))[:, None],
    (len(reference.lat), len(reference.lon)),
)
assert train == slice("1966-01", "2006-12")
statistics, masks, monthly_centers = [], {}, {}
ssrd_stats = None

for name in names:
    with dask.config.set(scheduler="threads", num_workers=2):
        fields = reference[name].sel(time=train).transpose("time", "lat", "lon").load()
    np.testing.assert_array_equal(fields.time.dt.month, np.tile(np.arange(1, 13), 41))
    x = fields.values.astype(float).reshape(
        41, 12, len(reference.lat), len(reference.lon)
    )
    valid = np.isfinite(x)
    count = valid.sum(0)
    monthly_mean = np.where(valid, x, 0).sum(0) / np.maximum(count, 1)
    monthly_variance = np.where(valid, x - monthly_mean, 0) ** 2
    monthly_variance = monthly_variance.sum(0) / np.maximum(count, 1)
    minimum = np.where(valid, x, np.inf).min(0)
    maximum = np.where(valid, x, -np.inf).max(0)
    supported = pre.stats[name]["mask"].values
    seasonal_support = (count >= 2) & supported
    seasonal_constant = seasonal_support & (minimum == maximum)
    annual_constant = supported & (minimum.min(0) == maximum.max(0))
    annual_count = count.sum(0)
    center = pre.stats[name]["center"].values
    scale = pre.stats[name]["std"].values
    floor = pre.stats[name]["std_floor"]
    annual_variance = (np.where(valid, x - center, 0) ** 2).sum((0, 1)) / np.maximum(
        annual_count, 1
    )
    np.testing.assert_array_equal(annual_count >= 2, supported)
    np.testing.assert_allclose(
        scale[supported],
        np.maximum(np.sqrt(annual_variance), floor)[supported],
        rtol=1e-8,
        atol=1e-12,
    )
    all_area = (area * supported).sum()
    within_variance = (monthly_variance * count).sum(0) / np.maximum(annual_count, 1)
    z = np.where(valid & supported, (x - center) / scale, 0)
    square = z**2
    weighted_variance = (area * supported * annual_variance).sum()
    statistics.append(
        dict(
            variable=name,
            supported_cells=int(supported.sum()),
            annual_constant_cells=int(annual_constant.sum()),
            annual_constant_area_percent=100
            * (area * annual_constant).sum()
            / all_area,
            annual_floor_area_percent=100
            * (area * supported * (np.sqrt(annual_variance) <= floor)).sum()
            / all_area,
            calendar_constant_area_month_percent=100
            * (area * seasonal_constant).sum()
            / (12 * all_area),
            calendar_constant_zero_area_month_percent=100
            * (area * seasonal_constant * (monthly_mean == 0)).sum()
            / (12 * all_area),
            calendar_insufficient_training_area_month_percent=100
            * (area * supported * (count < 2)).sum()
            / (12 * all_area),
            calendar_mask_min_retained_area_percent=100
            * min((area * (seasonal_support & ~seasonal_constant)).sum((1, 2)))
            / all_area,
            training_zero_area_month_percent=100
            * (area * supported * (valid & (x == 0))).sum()
            / (area * supported * valid).sum(),
            seasonal_cycle_share_of_physical_variance_percent=100
            * (1 - (area * supported * within_variance).sum() / weighted_variance),
            fraction_normalized_variance_from_abs_z_gt_3_percent=100
            * (area * np.where(np.abs(z) > 3, square, 0)).sum()
            / (area * square).sum(),
            std_floor=float(floor),
        )
    )
    masks[name] = {
        "annual_constant": annual_constant,
        "calendar_constant": seasonal_constant,
        "calendar_supported": seasonal_support,
    }
    monthly_centers[name] = monthly_mean.astype("float32")
    if name == "ssrd":
        masks[name]["near_dark"] = seasonal_support & (maximum <= 0.1) & (minimum >= 0)
        ssrd_stats = xr.Dataset(
            dict(
                training_mean=(("month", "lat", "lon"), monthly_mean),
                training_std=(("month", "lat", "lon"), np.sqrt(monthly_variance)),
                training_max=(("month", "lat", "lon"), maximum),
                constant=(("month", "lat", "lon"), seasonal_constant.astype("int8")),
                near_dark=(
                    ("month", "lat", "lon"),
                    masks[name]["near_dark"].astype("int8"),
                ),
                annual_std=(("lat", "lon"), np.sqrt(annual_variance)),
            ),
            coords=dict(month=np.arange(1, 13), lat=reference.lat, lon=reference.lon),
            attrs=dict(
                fit_period="1966–2006 only",
                near_dark_definition="Every finite training value for this calendar month is between 0 and 0.1 W/m2",
            ),
        )
    print(
        name,
        {
            k: round(v, 4)
            for k, v in statistics[-1].items()
            if k.endswith("area_month_percent")
        },
        flush=True,
    )
    del x, valid, z, square, monthly_variance, fields

OUTPUT.mkdir(parents=True, exist_ok=True)
ssrd_stats.to_netcdf(OUTPUT / "ssrd_monthly_statistics.nc")
scores, ssrd_scores = [], []
for period, (first, last) in PERIODS.items():
    monthly_scores = {name: [] for name in names}
    for year in range(first, last + 1):
        with xr.open_zarr(
            RUN / f"ae_reconstruction/year-{year:04d}.zarr", consolidated=True
        ) as archive:
            with dask.config.set(scheduler="threads", num_workers=2):
                observed = reference[names].sel(time=str(year)).load()
                predicted = archive[names].load()
        observed, predicted = xr.align(observed, predicted, join="exact")
        for name in names:
            target = observed[name].transpose("time", "lat", "lon").values.astype(float)
            prediction = (
                predicted[name].transpose("time", "lat", "lon").values.astype(float)
            )
            valid = np.isfinite(target) & pre.stats[name]["mask"].values
            assert np.isfinite(prediction[valid]).all()
            error = np.where(valid, prediction - target, 0)
            physical = error**2
            normalized = physical / pre.stats[name]["std"].values ** 2
            regions = dict(
                all=valid,
                annual_retained=valid & ~masks[name]["annual_constant"],
                calendar_retained=valid
                & masks[name]["calendar_supported"]
                & ~masks[name]["calendar_constant"],
                calendar_constant=valid & masks[name]["calendar_constant"],
            )
            if name == "ssrd":
                regions["near_dark"] = valid & masks[name]["near_dark"]
                regions["near_dark_retained"] = valid & ~masks[name]["near_dark"]
            denominator = (area * valid).sum((1, 2))
            row = {}
            for group, included in regions.items():
                weights = area * included
                group_area = weights.sum((1, 2))
                row[group + "_area_percent"] = 100 * group_area / denominator
                for space, errors in (
                    ("physical", physical),
                    ("normalized", normalized),
                ):
                    numerator = (weights * errors).sum((1, 2))
                    row[f"{group}_{space}_contribution"] = numerator / denominator
                    row[f"{group}_{space}_mse"] = np.divide(
                        numerator,
                        group_area,
                        out=np.full(12, np.nan),
                        where=group_area > 0,
                    )
            changed = (
                valid
                & masks[name]["calendar_constant"]
                & (target != monthly_centers[name])
            )
            row["constant_changed_area_percent"] = (
                100 * (area * changed).sum((1, 2)) / denominator
            )
            monthly_scores[name].append(row)
            if name == "ssrd":
                regions = dict(
                    global_all=valid,
                    arctic_summer=valid
                    & (reference.lat.values[None, :, None] >= 60)
                    & np.isin(np.arange(1, 13), [6, 7, 8])[:, None, None],
                    antarctic_summer=valid
                    & (reference.lat.values[None, :, None] <= -60)
                    & np.isin(np.arange(1, 13), [12, 1, 2])[:, None, None],
                    polar=valid & (np.abs(reference.lat.values[None, :, None]) >= 60),
                    near_dark=valid & masks[name]["near_dark"],
                )
                for group, included in regions.items():
                    denominator_region = (area * included).sum()
                    ssrd_scores.append(
                        dict(
                            period=period,
                            year=year,
                            group=group,
                            reference_mean=(area * np.where(included, target, 0)).sum()
                            / denominator_region,
                            prediction_mean=(
                                area * np.where(included, prediction, 0)
                            ).sum()
                            / denominator_region,
                            bias=(area * np.where(included, error, 0)).sum()
                            / denominator_region,
                            physical_mse=(area * np.where(included, physical, 0)).sum()
                            / denominator_region,
                            area=(area * included).sum(),
                        )
                    )
        if (year - first) % 10 == 0 or year == last:
            print(
                f"Rescored {period} through {year}: {perf_counter()-started:.1f}s",
                flush=True,
            )
    for name, rows in monthly_scores.items():
        result = dict(period=period, variable=name)
        for key in rows[0]:
            values = np.concatenate([r[key] for r in rows])
            result[key] = float(np.mean(values))
        for space in ("physical", "normalized"):
            original = result[f"all_{space}_mse"]
            result[f"calendar_mask_{space}_mse_change_percent"] = 100 * (
                result[f"calendar_retained_{space}_mse"] / original - 1
            )
            result[f"calendar_constant_{space}_error_share_percent"] = (
                100 * result[f"calendar_constant_{space}_contribution"] / original
            )
            if name == "ssrd":
                result[f"near_dark_{space}_error_share_percent"] = (
                    100 * result[f"near_dark_{space}_contribution"] / original
                )
        scores.append(result)

with (RUN / "atmosphere/history.csv").open() as stream:
    current_history = {int(r["epoch"]): r for r in csv.DictReader(stream)}
with (OLD / "atmosphere/history.csv").open() as stream:
    old_history = {int(r["epoch"]): r for r in csv.DictReader(stream)}
checks = []
for result in scores:
    if result["period"] != "validation":
        continue
    name = result["variable"]
    for space, key in (
        ("physical", f"val_rmse_{name}"),
        ("normalized", f"val_normalized_rmse_{name}"),
    ):
        actual = result[f"all_{space}_mse"] ** 0.5
        logged = float(current_history[saved["epoch"]][key])
        np.testing.assert_allclose(actual, logged, rtol=5e-4, atol=1e-7)
        checks.append(
            dict(variable=name, space=space, audit_rmse=actual, history_rmse=logged)
        )
matched = min(saved["epoch"], load_checkpoint(OLD / "atmosphere/model.pt")["epoch"])
comparison = [
    dict(
        variable=n,
        epoch=matched,
        previous_transforms_rmse=float(old_history[matched]["val_rmse_" + n]),
        identity_rmse=float(current_history[matched]["val_rmse_" + n]),
        change_percent=100
        * (
            float(current_history[matched]["val_rmse_" + n])
            / float(old_history[matched]["val_rmse_" + n])
            - 1
        ),
    )
    for n in names
]
for filename, rows in (
    ("statistics.csv", statistics),
    ("metrics.csv", scores),
    ("ssrd_regions.csv", ssrd_scores),
    ("history_checks.csv", checks),
    ("matched_epoch_comparison.csv", comparison),
):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUTPUT / filename).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

# Check that proposed deterministic polar-night outputs remain zero outside
# the fitting window. This is verification only; the mask remains training-fit.
night_check = []
for year in range(1850, 2101):
    with dask.config.set(scheduler="threads", num_workers=2):
        values = (
            reference.ssrd.sel(time=str(year))
            .transpose("time", "lat", "lon")
            .load()
            .values
        )
    included = masks["ssrd"]["calendar_constant"] & np.isfinite(values)
    night_check.append(
        dict(
            year=year,
            inspected_cell_months=int(included.sum()),
            nonzero_cell_months=int(np.count_nonzero(values[included])),
            maximum_absolute_flux=float(np.max(np.abs(values[included]))),
        )
    )
with (OUTPUT / "ssrd_polar_night_check.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(night_check[0]))
    writer.writeheader()
    writer.writerows(night_check)

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/monthly-diffusion-atmosphere-audit")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), layout="constrained")
dark = ssrd_stats.constant.mean("lon").values.T
mesh = axes[0].pcolormesh(
    np.arange(0.5, 13.5),
    np.arange(-90, 91),
    dark,
    cmap="Blues",
    vmin=0,
    vmax=1,
    shading="flat",
)
axes[0].set(
    xticks=np.arange(1, 13),
    xticklabels=list("JFMAMJJASOND"),
    xlabel="Calendar month",
    ylabel="Latitude",
    title="Where SSRD is always zero in training",
)
fig.colorbar(mesh, ax=axes[0], label="Fraction of longitudes")
annual_rms = np.sqrt((ssrd_stats.annual_std**2).mean("lon"))
within_rms = np.sqrt((ssrd_stats.training_std**2).mean(("month", "lon")))
axes[1].plot(
    ssrd_stats.lat, annual_rms, label="Current: all-month std", color="#b85b1d", lw=2
)
axes[1].plot(
    ssrd_stats.lat,
    within_rms,
    label="After removing monthly climatology",
    color="#146b8c",
    lw=2,
)
axes[1].set(
    xlabel="Latitude",
    ylabel="RMS local training std (W/m²)",
    title="Seasonality dominates SSRD variability",
    xlim=(-90, 90),
    ylim=(0, None),
)
axes[1].grid(alpha=0.2)
axes[1].legend(fontsize=9)
fig.suptitle(
    "SSRD: calendar-month masks and normalization | fitted on 1966–2006", fontsize=12
)
for suffix in ("png", "pdf"):
    fig.savefig(OUTPUT / f"ssrd_seasonal_diagnostic.{suffix}", dpi=180)
plt.close(fig)
assert model_identity(RUN) == identity
(OUTPUT / "provenance.json").write_text(
    json.dumps(
        dict(
            run=str(RUN),
            models=identity,
            selected_atmosphere_epoch=saved["epoch"],
            periods=PERIODS,
            fit_period=[train.start, train.stop],
            calendar_mask="At least two finite values and nonconstant across training years for this calendar month/gridpoint; physical values unchanged.",
            near_dark="Training maximum <= 0.1 W/m2 for a calendar month/gridpoint, all values nonnegative; diagnostic only, not an astronomical mask.",
            method="Rescore unchanged archived predictions; normalize cosine-latitude weights independently per variable and sample.",
            limitation="No retraining; matched-epoch comparison is one run per preprocessing choice, not a replicated experiment.",
        ),
        indent=2,
    )
    + "\n"
)
reference.close()
print(f"Saved {OUTPUT}; {perf_counter()-started:.1f}s", flush=True)
