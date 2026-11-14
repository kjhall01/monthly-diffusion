"""Audit OHC statistics/errors against the separate bathymetry supplement.

Run evaluate_zero_variance_masks.py first. Bathymetry is diagnostic only: its
native depths/map are not proven identical to the published climate geometry.
Neither climate data, preprocessing nor model weights are changed.
"""

import csv
import json
import sys
from pathlib import Path

import dask
import numpy as np
import xarray as xr
from scipy.sparse import coo_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from monthly_diffusion.checkpoints import load_checkpoint, model_identity, file_digest
from workflows.data.sources import open_state_dataset
from workflows.run_files import run_inputs

RUN = Path(__file__).resolve().parents[1] / "runs/test_gridpoint_no_transforms"
BATHY = Path("/Users/kyle/Dev/monthly-diffusion-data/data/ocean-bathymetry-v1")
OUTPUT = RUN / "analysis/ohc_shallows"
NAMES = ["ohc300", "ohc700_minus_ohc300"]
PERIODS = {"training": (1966, 2006), "validation": (1958, 1965), "late": (2070, 2100)}

identity = model_identity(RUN)
audit = json.loads((RUN / "analysis/zero_variance_masks/provenance.json").read_text())
assert audit["models"] == identity
preprocessor = load_checkpoint(RUN / "ocean/setup.pt")["preprocessor"]
assert all(v.transform == "identity" for v in preprocessor.variables)
source, _, _, _ = run_inputs(RUN)
reference = open_state_dataset(source)
with xr.open_dataset(RUN / "analysis/zero_variance_masks/masks.nc") as ds:
    statistics = ds.load()
with xr.open_dataset(BATHY / "e3smv2/bathymetry_e3smv2_1deg.nc") as ds:
    bathy = ds.load()
report = json.loads((BATHY / "e3smv2/build-report.json").read_text())
assert (
    file_digest(BATHY / "e3smv2/bathymetry_e3smv2_1deg.nc") == report["product_sha256"]
)
reference, statistics, bathy = xr.align(reference, statistics, bathy, join="exact")
assert (
    bathy.attrs["reference_source_lock_sha256"] == reference.attrs["source_lock_sha256"]
)
wet = preprocessor.stats["sst"]["mask"].values
depth = bathy.deptho.values
assert np.isfinite(depth[wet]).all()
area = np.broadcast_to(
    np.cos(np.deg2rad(reference.lat.values.astype(float)))[:, None], wet.shape
)

# Use native depths to distinguish mixed shallow/deep target cells. Thresholding
# the remapped mean depth alone would discard some real deep-water contributions.
with (
    xr.open_dataset(BATHY / "cache/e3sm-bottomDepth-EC30to60E2r2.nc") as native,
    xr.open_dataset(BATHY / "cache/e3sm-map.nc") as mapping,
):
    native_depth = native.bottomDepth.values.astype(float)
    native_wet = native.maxLevelCell.values > 0
    matrix = coo_matrix(
        (
            mapping.S.values,
            (mapping.row.values.astype(int) - 1, mapping.col.values.astype(int) - 1),
        ),
        shape=(mapping.sizes["n_b"], mapping.sizes["n_a"]),
    ).tocsr()
    coverage = matrix @ native_wet.astype(float)
    projected = np.divide(
        matrix @ np.where(native_wet, native_depth, 0),
        coverage,
        out=np.full(len(coverage), np.nan),
        where=coverage > 0,
    ).reshape(wet.shape)
    np.testing.assert_allclose(projected[wet], depth[wet], rtol=1e-6)
    geometry = {}
    for name, lower, upper in ((NAMES[0], 0, 300), (NAMES[1], 300, 700)):
        thickness = np.clip(native_depth - lower, 0, upper - lower)
        geometry[name] = np.divide(
            matrix @ np.where(native_wet, thickness, 0),
            coverage,
            out=np.zeros(len(coverage)),
            where=coverage > 0,
        ).reshape(wet.shape)

groups = {}
summary = []
geometry_checks = []
for name in NAMES:
    support = preprocessor.stats[name]["mask"].values
    constant = statistics.constant_training.sel(variable=name).values.astype(bool)
    raw_std = statistics.unfloored_std.sel(variable=name).values
    floor = preprocessor.stats[name]["std_floor"]
    thickness = geometry[name]
    maximum = 300 if name == "ohc300" else 400
    groups[name] = {
        "all": support,
        "mean_depth_le_300m": support & (depth <= 300),
        "mean_depth_300_to_700m": support & (depth > 300) & (depth <= 700),
        "mean_depth_gt_700m": support & (depth > 700),
        "constant_training": support & constant,
        "variable_below_std_floor": support & ~constant & (raw_std < floor),
        "variable_above_std_floor": support & ~constant & (raw_std >= floor),
        "auxiliary_no_layer": support & (thickness == 0),
        "auxiliary_partial_layer": support
        & (thickness > 0)
        & (thickness < maximum - 1e-6),
        "auxiliary_full_layer": support & (thickness >= maximum - 1e-6),
    }
    for group, mask in groups[name].items():
        denominator = (area * mask).sum()
        if not mask.any():
            continue
        summary.append(
            dict(
                variable=name,
                group=group,
                cells=int(mask.sum()),
                area_percent=100 * denominator / (area * support).sum(),
                constant_area_percent_within_group=100
                * (area * mask * constant).sum()
                / denominator,
                std_floor_area_percent_within_group=100
                * (area * mask * (raw_std <= floor)).sum()
                / denominator,
                area_weighted_rms_training_std_GJ_m2=np.sqrt(
                    (area[mask] * raw_std[mask] ** 2).sum() / denominator
                )
                / 1e9,
                median_training_std_GJ_m2=float(np.median(raw_std[mask])) / 1e9,
                std_floor_GJ_m2=floor / 1e9,
                mean_effective_layer_thickness_m=(area[mask] * thickness[mask]).sum()
                / denominator,
            )
        )
    geometry_checks.append(
        dict(
            variable=name,
            constant_cells=int(constant.sum()),
            constant_cells_with_auxiliary_layer=int((constant & (thickness > 0)).sum()),
            varying_cells_without_auxiliary_layer=int(
                (support & ~constant & (thickness == 0)).sum()
            ),
            varying_cells_with_mean_depth_below_lower_bound=int(
                (
                    support & ~constant & (depth <= (0 if name == "ohc300" else 300))
                ).sum()
            ),
            varying_cells_with_training_std_under_1e4_J_m2=int(
                (support & ~constant & (raw_std < 1e4)).sum()
            ),
        )
    )

scores = []
for period, (first, last) in PERIODS.items():
    accumulated = {
        (name, group): []
        for name in NAMES
        for group, mask in groups[name].items()
        if mask.any()
    }
    for year in range(first, last + 1):
        with xr.open_zarr(
            RUN / f"ae_reconstruction/year-{year:04d}.zarr", consolidated=True
        ) as shard:
            with dask.config.set(scheduler="threads", num_workers=2):
                observed = reference[NAMES].sel(time=str(year)).load()
                predicted = shard[NAMES].load()
        observed, predicted = xr.align(observed, predicted, join="exact")
        for name in NAMES:
            target = observed[name].transpose("time", "lat", "lon").values.astype(float)
            prediction = (
                predicted[name].transpose("time", "lat", "lon").values.astype(float)
            )
            error = prediction - target
            std = preprocessor.stats[name]["std"].values
            valid = np.isfinite(target) & groups[name]["all"]
            assert np.isfinite(prediction[valid]).all()
            total_area = (area * valid).sum((1, 2))
            for group, mask in groups[name].items():
                if not mask.any():
                    continue
                included = valid & mask
                residual = np.where(included, error, 0)
                denominator = (area * included).sum((1, 2))
                physical = (area * residual**2).sum((1, 2))
                normalized = (area * (residual / std) ** 2).sum((1, 2))
                accumulated[name, group].append(
                    np.stack(
                        (
                            physical / denominator,
                            normalized / denominator,
                            physical / total_area,
                            normalized / total_area,
                        ),
                        axis=1,
                    )
                )
        if year == last:
            print(f"Scored {period}: {first}–{last}", flush=True)
    for name in NAMES:
        full = np.concatenate(accumulated[name, "all"]).mean(0)
        for group, mask in groups[name].items():
            if not mask.any():
                continue
            values = np.concatenate(accumulated[name, group]).mean(0)
            scores.append(
                dict(
                    variable=name,
                    period=period,
                    group=group,
                    physical_rmse_GJ_m2=np.sqrt(values[0]) / 1e9,
                    normalized_rmse=np.sqrt(values[1]),
                    physical_error_share_percent=100 * values[2] / full[2],
                    normalized_error_share_percent=100 * values[3] / full[3],
                )
            )

# Cross-check all-domain results against the independent preceding audit.
with (RUN / "analysis/zero_variance_masks/metrics.csv").open() as stream:
    previous = {(r["variable"], r["period"]): r for r in csv.DictReader(stream)}
for row in scores:
    if row["group"] == "all":
        old = previous[row["variable"], row["period"]]
        np.testing.assert_allclose(
            row["physical_rmse_GJ_m2"],
            float(old["original_physical_rmse"]) / 1e9,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            row["normalized_rmse"],
            float(old["original_normalized_mse"]) ** 0.5,
            rtol=1e-12,
        )
assert model_identity(RUN) == identity
OUTPUT.mkdir(parents=True, exist_ok=True)
for filename, rows in (
    ("statistics.csv", summary),
    ("errors.csv", scores),
    ("geometry_checks.csv", geometry_checks),
):
    with (OUTPUT / filename).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
(OUTPUT / "provenance.json").write_text(
    json.dumps(
        dict(
            run=str(RUN),
            models=identity,
            bathymetry=str(BATHY),
            bathymetry_sha256=report["product_sha256"],
            qualification=bathy.attrs["source_compatibility"],
            method="Unchanged predictions and training statistics; cosine-latitude weighted metrics; depth/layer groups diagnostic only.",
            source_profile_cache_available=False,
            limitation="Native auxiliary bathymetry and mapping are not verified against the historical restart/published volume field. An exact wet-layer validity mask requires the original volcello/thetao geometry.",
            periods=PERIODS,
        ),
        indent=2,
    )
    + "\n"
)
print(f"Saved {OUTPUT}", flush=True)
