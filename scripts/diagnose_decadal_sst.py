#!/usr/bin/env python
"""Stream saved SST archives, compare decadal variability, and render figures.

Run from the repository root with PYTHONPATH=frozen. Annual 5-degree caches
make subsequent EOF/plot iterations cheap; no forecasts are regenerated.
"""

import argparse
import hashlib
import json
import logging
import importlib.metadata
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import yaml


from monthly_diffusion.experiments.diagnose_decadal_sst import run

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
    raise ValueError("Configuration differs; use a new output directory")
marker.write_text(json.dumps(config, indent=2) + "\n")
code = [
    Path(__file__),
    *[
        Path(__file__).parents[1] / "monthly_diffusion/analysis" / p
        for p in (
            "decadal.py",
            "decadal_plots.py",
            "archives.py",
            "eofs.py",
            "climate.py",
        )
    ],
]
provenance = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code}
(root / ("plot_code.json" if args.plots_only else "code_provenance.json")).write_text(
    json.dumps(provenance, indent=2) + "\n"
)
environment = dict(
    python=platform.python_version(),
    platform=platform.platform(),
    packages={
        p: importlib.metadata.version(p)
        for p in ("numpy", "scipy", "xarray", "zarr", "matplotlib", "cartopy", "pandas")
    },
)
(
    root / ("plot_environment.json" if args.plots_only else "environment.json")
).write_text(json.dumps(environment, indent=2) + "\n")
if not args.plots_only:
    run(config, root)
from monthly_diffusion.analysis.decadal_plots import plot_decadal

plot_decadal(root, config)
logging.info("Finished decadal SST analysis: %s", root.resolve())
