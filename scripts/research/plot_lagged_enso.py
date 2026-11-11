#!/usr/bin/env python
"""Saved-run lagged ENSO regressions and publication maps; no new inference."""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import yaml

from monthly_diffusion.analysis.archives import ClimateArchive
from monthly_diffusion.analysis.lagged_enso import (
    lagged_enso_products,
    plot_lagged_enso,
    SEASONS,
)
from monthly_diffusion.analysis.long_rollout import save_dataset
from monthly_diffusion.analysis.spatial import (
    weighted_rmse,
    weighted_pattern_correlation,
)
from monthly_diffusion.data.configs import Variable
from monthly_diffusion.data.dataloader import validate_units


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("config", type=Path)
parser.add_argument("--start-year", type=int)
parser.add_argument("--end-year", type=int)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logging.getLogger("fontTools").setLevel(logging.WARNING)
config = yaml.safe_load(args.config.read_text())
if args.start_year is not None:
    config["years"][0] = args.start_year
if args.end_year is not None:
    config["years"][1] = args.end_year
if args.output is not None:
    config["output"] = str(args.output)
root = Path(config["output"])
root.mkdir(parents=True, exist_ok=True)
model = ClimateArchive([config["forecast"]], forecast=True)
reference = ClimateArchive(
    config["references"],
    ocean_support=model.ocean_support,
    ocean_variables=model.ocean_variables,
)
if config["training_member"] in reference.members or len(reference.members) < 2:
    raise ValueError("Use held-out reference members, excluding the training member")
if not np.array_equal(model.lat, reference.lat) or not np.array_equal(
    model.lon, reference.lon
):
    raise ValueError("Model and reference grids differ")
variables = config["variables"]
for archive in (model, reference):
    validate_units(
        xr.Dataset(
            {
                v: xr.DataArray(attrs={"units": archive.units[v]})
                for v in ["sst", *variables]
            }
        ),
        [
            Variable("sst", "ocean", "degC"),
            Variable("z500", "atmosphere", "m2 s-2"),
            Variable("mtpr", "atmosphere", "kg m-2 s-1"),
        ],
    )
source_files = [
    Path(__file__),
    Path(__file__).parents[2] / "monthly_diffusion/analysis/lagged_enso.py",
]
identity = dict(
    config=config,
    forecast_checkpoint=model.record["checkpoint_sha256"],
    forecast_record_sha256=hashlib.sha256(
        (model.paths[0] / "forecast.json").read_bytes()
    ).hexdigest(),
    reference_members=list(reference.members),
    forecast_members=list(map(int, model.members)),
    scripts={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
)
marker = root / "configuration.json"
if marker.exists() and json.loads(marker.read_text()) != identity:
    raise ValueError("Analysis identity changed; choose a new output directory")
marker.write_text(json.dumps(identity, indent=2) + "\n")
products = {}
for name, archive in (("model", model), ("reference", reference)):
    path = root / name / "regressions.nc"
    if path.exists():
        products[name] = xr.load_dataset(path)
    else:
        products[name] = lagged_enso_products(
            archive,
            np.arange(config["years"][0], config["years"][1] + 1),
            variables,
            root / name,
            calendar=config["calendar"],
        )
differences = xr.Dataset(
    {v: products["model"][v] - products["reference"][v] for v in variables}
)
for v in variables:
    differences[v].attrs.update(products["model"][v].attrs)
differences.attrs["definition"] = "Monthly Diffusion v1.0 minus held-out E3SMv2"
save_dataset(differences, root / "differences.nc")
display, rows = {}, []
for variable in variables:
    mod, ref = products["model"][variable], products["reference"][variable]
    # Dataset-level coverage metadata are supplied to plotting explicitly.
    mod.attrs.update(
        {k: products["model"].attrs[k] for k in ("winter_start", "winter_end")}
    )
    display[variable] = plot_lagged_enso(
        ref, mod, variable, root / f"enso_lagged_{variable}"
    )
    for season in SEASONS:
        m, r = mod.sel(season=season), ref.sel(season=season)
        rows.append(
            dict(
                variable=variable,
                season=season,
                centered_pattern_correlation=float(weighted_pattern_correlation(m, r)),
                rmse_display=float(weighted_rmse(m, r))
                * display[variable]["display_factor"],
                display_units=display[variable]["display_units"],
            )
        )
(root / "display.json").write_text(json.dumps(display, indent=2) + "\n")
pd.DataFrame(rows).to_csv(root / "map_metrics.csv", index=False)
logging.info("Finished: %s", root.resolve())
