"""Compare candidate transforms' training distributions, without fitting AEs.

Tail fractions describe normalized target variance, not reconstruction skill.
The log1p scales are explicit experiment candidates, not fitted optima.
"""

import csv
import sys
from pathlib import Path

import dask
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workflows.data.sources import open_state_dataset
from workflows.run_files import run_inputs
from monthly_diffusion.checkpoints import load_checkpoint

RUN = Path(__file__).resolve().parents[1] / "runs/test_gridpoint_no_transforms"
OUTPUT = RUN / "analysis/atmosphere_masks/transform_tail_diagnostics.csv"
CANDIDATES = [
    ("mtpr", "atmosphere", 1 / 86400),  # 1 mm/day, in the stored kg/m2/s units
    ("mld", "ocean", 50.0),  # m
    ("sivol", "ocean", 0.5),  # m of equivalent thickness
    ("d20", "ocean", 50.0),  # m
]

source, _, training_period, _ = run_inputs(RUN)
dataset = open_state_dataset(source)
area = np.cos(np.deg2rad(dataset.lat.values.astype(float)))[None, :, None]
rows = []
for name, component, characteristic_scale in CANDIDATES:
    preprocessor = load_checkpoint(RUN / component / "setup.pt")["preprocessor"]
    with dask.config.set(scheduler="threads", num_workers=2):
        x = (
            dataset[name]
            .sel(time=training_period)
            .transpose("time", "lat", "lon")
            .load()
            .values.astype(float)
        )
    valid = np.isfinite(x) & preprocessor.stats[name]["mask"].values
    assert (x[valid] >= 0).all()
    count = valid.sum(0)
    for transform in ("identity", "log1p"):
        values = x if transform == "identity" else np.log1p(x / characteristic_scale)
        center = np.where(valid, values, 0).sum(0) / np.maximum(count, 1)
        variance = (np.where(valid, values - center, 0) ** 2).sum(0) / np.maximum(
            count, 1
        )
        typical_std = np.sqrt(
            (area[0] * np.where(count >= 2, variance, 0)).sum()
            / (area[0] * (count >= 2)).sum()
        )
        floor = max(preprocessor.gridpoint_std_floor * typical_std, 1e-6)
        z = np.where(valid, (values - center) / np.maximum(np.sqrt(variance), floor), 0)
        square = z**2
        tail = (np.abs(z) > 3) & valid
        row = dict(
            variable=name,
            transform=transform,
            scale=characteristic_scale if transform == "log1p" else 1.0,
            valid_area_months_above_3_sigma_percent=100
            * (area * tail).sum()
            / (area * valid).sum(),
            normalized_variance_from_above_3_sigma_percent=100
            * (area * np.where(tail, square, 0)).sum()
            / (area * square).sum(),
        )
        rows.append(row)
        print(row, flush=True)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT.open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
dataset.close()
