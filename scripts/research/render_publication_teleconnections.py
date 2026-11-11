#!/usr/bin/env python
"""Render retained monthly Niño3.4 teleconnections in field units per °C.

Run from the repository root with the publication-nino34-t2m-precip YAML.
The saved long-rollout diagnostics subtract each ensemble's exact monthly
ensemble mean and multiply both fields and SST by sqrt(N/(N-1)). Niño3.4 is
the cosine-latitude-weighted SST anomaly over 5S--5N, 170W--120W. There is no
second monthly-climatology removal, detrending, smoothing, or seasonal subset.

Saved OLS maps regress internal fields on independently standardized indices,
pooling members and months with an intercept. For I in °C and z=(I-mean)/sd,
beta_I = beta_z / sd(I), EXACTLY, including under spatially varying missing
field data. This script uses each ensemble's own saved index standard deviation
(ddof=0) on the same sample interval; it does not use the reference's SD for MD.
Finite-ensemble rescaling cancels in this per-degree slope because the same
factor was applied to predictor and response. T2M differences need no Kelvin
offset; precipitation converts from kg m-2 s-1 to mm day-1 with factor 86400.

These are contemporaneous monthly associations, not lagged responses or causal
effects. Pooling historical and SSP3-7.0 months can hide changes in teleconnection
strength with forcing. Per-degree maps assess coupling, not ENSO amplitude.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import yaml

from monthly_diffusion.analysis.long_rollout import save_dataset
from monthly_diffusion.analysis.plotting import plot_response_matrix


from monthly_diffusion.experiments.render_publication_teleconnections import (
    STYLE,
    load_teleconnections,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("config", type=Path)
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logging.getLogger("fontTools").setLevel(logging.WARNING)
config = yaml.safe_load(args.config.read_text())
root = Path(config["output"])
root.mkdir(parents=True, exist_ok=True)
marker = root / "configuration.json"
if marker.exists() and json.loads(marker.read_text()) != config:
    raise ValueError("Configuration differs; select a new output folder")
reference, model, indices, counts, metadata = load_teleconnections(
    config["diagnostics"], config["variables"]
)
marker.write_text(json.dumps(config, indent=2) + "\n")
difference = model - reference
for v in difference:
    difference[v].attrs = dict(
        units=STYLE[v][1], long_name="MD minus E3SMv2 regression slope"
    )
for source, fields in (
    ("reference", reference),
    ("model", model),
    ("difference", difference),
):
    fields.attrs.update(
        regression=metadata["regression"], period=json.dumps(metadata["period"])
    )
    save_dataset(fields, root / f"{source}_teleconnections.nc")
for source in ("reference", "model"):
    save_dataset(indices[source], root / f"{source}_nino34.nc")
    save_dataset(counts[source], root / f"{source}_regression_count.nc")
metadata["display"] = plot_response_matrix(
    reference,
    model,
    variables=config["variables"],
    labels={v: STYLE[v][0] for v in config["variables"]},
    units={v: STYLE[v][1] for v in config["variables"]},
    path=root / "nino34_teleconnections_robinson",
    column_titles=tuple(config["column_titles"]),
    percentile=config.get("display_percentile", 99.0),
    dpi=config.get("dpi", 450),
)
scientific_config = Path(config["diagnostics"]) / "configuration.json"
record = json.loads(scientific_config.read_text())
metadata["checkpoint_sha256"] = record["forecast"]["checkpoint_sha256"]
metadata["diagnostic_configuration_sha256"] = hashlib.sha256(
    scientific_config.read_bytes()
).hexdigest()
metadata["code_sha256"] = {
    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (
        Path(__file__),
        Path(__file__).parents[2] / "monthly_diffusion/analysis/plotting.py",
    )
}
(root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
logging.info("Saved %s", root.resolve())
