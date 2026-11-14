#!/usr/bin/env python
"""Render retained frozen-package response maps without changing diagnostics."""

# Make the repository workflow helpers available when running this file directly.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
import logging

import matplotlib

matplotlib.use("Agg")
import yaml

from workflows.analysis.long_rollout import save_dataset
from workflows.analysis.plotting import plot_response_matrix


from workflows.experiments.render_publication_forced_response import STYLE, load_responses

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
marker.write_text(json.dumps(config, indent=2) + "\n")
reference, model, metadata = load_responses(config["diagnostics"], config["variables"])
for name, fields in (
    ("reference", reference),
    ("model", model),
    ("difference", model - reference),
):
    save_dataset(fields, root / f"{name}_response.nc")
metadata["display"] = plot_response_matrix(
    reference,
    model,
    variables=config["variables"],
    labels={v: STYLE[v][0] for v in config["variables"]},
    units={v: STYLE[v][1] for v in config["variables"]},
    path=root / "forced_response_robinson",
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
        Path(__file__).parents[2] / "workflows/analysis/plotting.py",
    )
}
(root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
logging.info("Saved %s", root.resolve())
