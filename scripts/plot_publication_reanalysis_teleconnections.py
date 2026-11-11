#!/usr/bin/env python
"""Matched ERA5/ORAS5 and MDv1 Niño3.4 monthly teleconnections, per degree C.

Reuse the observed-initialized rollout and matched-period index analysis from
the four-source publication figure. SST/Niño3.4 reference is ORAS5; atmospheric
targets are ERA5. Both trajectories lose their own monthly climatology and
linear trend over the common 1958-03--2025-12 interval. This is descriptive
within-period regression, not independent forecast verification. Stream only
the requested physical variables; no inference, no change to the E3SM figure.
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import xarray as xr
import yaml

from monthly_diffusion.analysis.archives import ClimateArchive
from monthly_diffusion.analysis.long_rollout import save_dataset
from monthly_diffusion.analysis.monthly_regression import regress_on_monthly_residual
from monthly_diffusion.analysis.plotting import plot_response_matrix

STYLE = {
    "t2m": ("2-meter Temperature", "°C / °C Niño3.4", "K", 1.),
    "mtpr": ("Precipitation", "mm day⁻¹ / °C Niño3.4", "kg m-2 s-1", 86400.),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    config = yaml.safe_load(args.config.read_text())
    root, source_root = Path(config["output"]), Path(config["index_analysis"])
    root.mkdir(parents=True, exist_ok=True)
    source_config = json.loads((source_root/"configuration.json").read_text())
    identity = dict(config=config, index_configuration=source_config)
    marker = root/"configuration.json"
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise ValueError("Configuration changed; choose a new directory")
    marker.write_text(json.dumps(identity, indent=2)+"\n")
    maps, metadata, intervals = {}, dict(sources={}), []
    for role, source in config["sources"].items():
        spec = source_config["sources"][source]
        archive = ClimateArchive(spec["paths"], forecast=spec.get("forecast", False))
        recorded = json.loads((source_root/source/"source_identity.json").read_text())
        if list(map(str, archive.members)) != recorded["members"] or list(map(str, archive.paths)) != recorded["paths"]:
            raise ValueError("Archive members/paths differ from the saved index analysis")
        if archive.forecast:
            digest = hashlib.sha256((archive.paths[0]/"forecast.json").read_bytes()).hexdigest()
            if digest != recorded["forecast_record_sha256"]:
                raise ValueError("Forecast identity changed since index extraction")
        else:
            digests = {str(p): hashlib.sha256((p/".zmetadata").read_bytes()).hexdigest() for p in archive.paths}
            if digests != recorded["source_metadata_sha256"]:
                raise ValueError("Reference identity changed since index extraction")
        index_path = source_root/source/"indices.nc"
        indices = xr.load_dataset(index_path)
        index = indices.nino34
        if index.attrs.get("units") != "degC":
            raise ValueError("Niño3.4 must retain degrees Celsius")
        interval = [str(index.time.values[0])[:10], str(index.time.values[-1])[:10]]
        intervals.append(interval)
        metadata["sources"][role] = dict(identity=recorded, period=interval,
            index_std_degC=float(index.std()), index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
            members=index.sizes["member"], months_per_member=index.sizes["time"], total_samples=index.size)
        save_dataset(index.rename("ENSO").to_dataset(), root/f"{role}_nino34.nc")
        fields = {}
        for variable in config["variables"]:
            if archive.units[variable] != STYLE[variable][2]:
                raise ValueError(f"Unexpected {variable} units")
            product = regress_on_monthly_residual(archive, index, variable, config.get("block_months", 12))
            save_dataset(product, root/role/f"{variable}_native_regression.nc")
            fields[variable] = product.slope*STYLE[variable][3]
            fields[variable].attrs.update(units=STYLE[variable][1], predictor="Niño3.4 SST anomaly (°C)")
        maps[role] = xr.Dataset(fields)
        save_dataset(maps[role], root/f"{role}_teleconnections.nc")
    if intervals[0] != intervals[1]:
        raise ValueError("Reference and model analysis periods differ")
    reference, model = xr.align(maps["reference"], maps["model"], join="exact")
    difference = model-reference
    for v in difference:
        difference[v].attrs["units"] = STYLE[v][1]
    save_dataset(difference, root/"difference_teleconnections.nc")
    metadata.update(period=intervals[0], lag_months=0, season="all months",
        anomaly_definition="per-trajectory monthly means + linear trend removed over common analysis interval; no ensemble subtraction or variance standardization",
        reference="ERA5 atmospheric fields regressed on ORAS5 Niño3.4; MD fields regressed on MD Niño3.4",
        difference="MDv1-ERA5/ORAS5 minus ERA5/ORAS5",
        caveat="Different anomaly definition and shorter period than the ensemble-residual 1851–2100 E3SM figure; no significance claim")
    metadata["display"] = plot_response_matrix(reference, model, variables=config["variables"],
        labels={v: STYLE[v][0] for v in config["variables"]}, units={v: STYLE[v][1] for v in config["variables"]},
        path=root/"nino34_teleconnections_robinson", column_titles=tuple(config["column_titles"]),
        percentile=config.get("display_percentile", 99.), dpi=config.get("dpi", 450))
    (root/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    code = [Path(__file__), *[Path(__file__).parents[1]/"monthly_diffusion/analysis"/f"{name}.py"
        for name in ("monthly_regression", "long_rollout", "archives", "plotting")]]
    (root/"code_provenance.json").write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code}, indent=2)+"\n")
    logging.info("Saved %s", root.resolve())


if __name__ == "__main__":
    main()
