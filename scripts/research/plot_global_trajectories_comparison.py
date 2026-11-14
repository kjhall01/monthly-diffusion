#!/usr/bin/env python
"""Raw annual global means: E3SM/reanalysis and their real-state-start emulators.

Streams only the selected physical variables; never runs inference. Spatial means use
cos(latitude) over valid cells, with fixed training-SST support for SST. Annual
means give equal weight to 12 monthly means, following long-rollout diagnostics.
No climatology subtraction, trend removal, rebaselining, or bias correction.
"""

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


from workflows.experiments.plot_global_trajectories_comparison import calculate, plot

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("config", type=Path)
parser.add_argument("--plots-only", action="store_true")
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logging.getLogger("fontTools").setLevel(logging.WARNING)
config = yaml.safe_load(args.config.read_text())
root = Path(config["output"])
root.mkdir(parents=True, exist_ok=True)
marker = root / "configuration.json"
if marker.exists() and json.loads(marker.read_text()) != config:
    raise ValueError("Configuration changed: choose a new output directory")
marker.write_text(json.dumps(config, indent=2) + "\n")
code = [
    Path(__file__),
    Path(__file__).parents[2] / "workflows/analysis/archives.py",
    Path(__file__).parents[2] / "workflows/analysis/climate.py",
    Path(__file__).parents[2] / "workflows/analysis/spatial.py",
]
(root / "code_provenance.json").write_text(
    json.dumps(
        {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code}, indent=2
    )
    + "\n"
)
if "data_directory" in config:
    data_root = Path(config["data_directory"])
    source_files = [data_root / f"{name}_annual.nc" for name in config["sources"]]
    source_files += [
        data_root / "data_provenance.json",
        data_root / "configuration.json",
    ]
    (root / "input_series_sha256.json").write_text(
        json.dumps(
            {
                str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in source_files
            },
            indent=2,
        )
        + "\n"
    )
elif not args.plots_only:
    calculate(config, root)
plot(config, root)
logging.info("Saved %s", root.resolve())
