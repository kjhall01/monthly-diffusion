#!/usr/bin/env python
"""Render retained frozen-package response maps without changing diagnostics."""
import argparse
import hashlib
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import xarray as xr
import yaml

from monthly_diffusion.analysis.long_rollout import save_dataset
from monthly_diffusion.analysis.plotting import plot_response_matrix

STYLE = {
    "t2m": ("2-meter Temperature", "°C", "K", 1.),
    "sst": ("Sea Surface Temperature", "°C", "degC", 1.),
    "mtpr": ("Precipitation", "mm day⁻¹", "kg m-2 s-1", 86400.),
}


def load_responses(root, variables):
    """Reuse exact saved fields/periods/members; temperature DIFFERENCES need no offset."""
    root = Path(root)
    responses, inputs, periods, counts = {}, {}, [], {}
    for source in ("reference", "model"):
        maps = {}
        for variable in variables:
            path = root/source/f"{variable}.nc"
            with xr.open_dataset(path) as ds:
                field = ds.forced_response.load()
                if field.attrs.get("units") != STYLE[variable][2]:
                    raise ValueError(f"Unexpected {variable} source units: {field.attrs}")
                period = json.loads(ds.attrs["periods"])
                periods.append({name: period[name] for name in ("early", "late")})
                # Verify the retained response against its saved climatologies.
                xr.testing.assert_allclose(field, ds.climatology.sel(period="late", drop=True)-
                                            ds.climatology.sel(period="early", drop=True))
                maps[variable] = field*STYLE[variable][3]
                maps[variable].attrs["units"] = STYLE[variable][1]
                counts[source] = dict(members=list(map(str, ds.member.values)), count=ds.sizes["member"])
            inputs[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        responses[source] = xr.Dataset(maps)
    if any(period != periods[0] for period in periods):
        raise ValueError("Response maps use different periods")
    reference, model = xr.align(responses["reference"], responses["model"], join="exact")
    return reference, model, dict(periods=periods[0], ensembles=counts, input_sha256=inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    config = yaml.safe_load(args.config.read_text())
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=True)
    marker = root/"configuration.json"
    if marker.exists() and json.loads(marker.read_text()) != config:
        raise ValueError("Configuration differs; select a new output folder")
    marker.write_text(json.dumps(config, indent=2)+"\n")
    reference, model, metadata = load_responses(config["diagnostics"], config["variables"])
    for name, fields in (("reference", reference), ("model", model), ("difference", model-reference)):
        save_dataset(fields, root/f"{name}_response.nc")
    metadata["display"] = plot_response_matrix(reference, model, variables=config["variables"],
        labels={v: STYLE[v][0] for v in config["variables"]},
        units={v: STYLE[v][1] for v in config["variables"]},
        path=root/"forced_response_robinson", column_titles=tuple(config["column_titles"]),
        percentile=config.get("display_percentile", 99.), dpi=config.get("dpi", 450))
    scientific_config = Path(config["diagnostics"])/"configuration.json"
    record = json.loads(scientific_config.read_text())
    metadata["checkpoint_sha256"] = record["forecast"]["checkpoint_sha256"]
    metadata["diagnostic_configuration_sha256"] = hashlib.sha256(scientific_config.read_bytes()).hexdigest()
    metadata["code_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), Path(__file__).parents[1]/"monthly_diffusion/analysis/plotting.py")}
    (root/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    logging.info("Saved %s", root.resolve())


if __name__ == "__main__":
    main()
