#!/usr/bin/env python
"""Three-source ENSO dynamical diagnostics from existing monthly archives."""
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
from monthly_diffusion.analysis.enso_dynamics import (fixed_sst_support, extract_source,
    lagged_products, OCEAN_VARIABLES)
from monthly_diffusion.analysis.enso_dynamics_plots import plot_dynamics, STYLE
from monthly_diffusion.analysis.long_rollout import save_dataset
from monthly_diffusion.analysis.spatial import weighted_rmse, weighted_pattern_correlation
from monthly_diffusion.data.configs import Variable
from monthly_diffusion.data.dataloader import validate_units


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--stage", choices=("all", "calculate", "plots"), default="all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    config = yaml.safe_load(args.config.read_text())
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=True)
    marker = root/"configuration.json"
    if marker.exists() and json.loads(marker.read_text()) != config:
        raise ValueError("Configuration changed; choose a new output directory")
    marker.write_text(json.dumps(config, indent=2)+"\n")
    code_paths = [Path(__file__), *list((Path(__file__).parents[1]/"monthly_diffusion/analysis").glob("enso_dynamics*.py")),
        Path(__file__).parents[1]/"monthly_diffusion/analysis/lagged_enso.py",
        Path(__file__).parents[1]/"monthly_diffusion/analysis/trajectory_anomalies.py"]
    provenance = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths}
    (root/f"code_provenance_{args.stage}.json").write_text(json.dumps(provenance, indent=2)+"\n")
    summaries, source_info = [], {}
    if args.stage != "plots":
        for name, spec in config["sources"].items():
            out = root/name
            out.mkdir(exist_ok=True)
            maskpath = out/"sst_support.nc"
            if maskpath.exists():
                support = xr.load_dataset(maskpath).sst_support.astype(bool)
            else:
                logging.info("%s: deriving fixed SST support from canonical training data", name)
                support = fixed_sst_support(ClimateArchive(spec.get("support_paths", spec["paths"])))
                save_dataset(support.rename("sst_support").to_dataset(), maskpath)
            archive = ClimateArchive(spec["paths"], forecast=spec.get("forecast", False),
                                     ocean_support=support, ocean_variables=OCEAN_VARIABLES)
            if name == "e3sm" and config["training_member"] in archive.members:
                raise ValueError("E3SM references must exclude the training member")
            units = dict(sst="degC", ohc300="J m-2", d20="m", tau_x="N m-2", ssh="m")
            validate_units(xr.Dataset({v: xr.DataArray(attrs={"units": archive.units[v]}) for v in units}),
                           [Variable(v, "", u) for v, u in units.items()])
            source_info[name] = dict(paths=list(map(str, archive.paths)), members=list(map(str, archive.members)),
                calendar=spec["calendar"], units={v: archive.units[v] for v in units},
                sst_support_cells=int(support.sum()),
                support_sha256=hashlib.sha256(np.asarray(support, dtype=np.uint8).tobytes()).hexdigest(),
                checkpoint=archive.record.get("checkpoint_sha256"))
            if archive.forecast:
                source_info[name]["forecast_record_sha256"] = hashlib.sha256((archive.paths[0]/"forecast.json").read_bytes()).hexdigest()
            else:
                source_info[name]["dataset_attrs"] = []
                for p in archive.paths:
                    with xr.open_zarr(p, chunks=None) as ds:
                        source_info[name]["dataset_attrs"].append(ds.attrs)
            extract_source(archive, config, spec, out)
            for variable in config["map_variables"]:
                map_path = out/f"seasonal_{variable}.nc"
                maps = xr.load_dataset(map_path)
                nominal = len(archive.members)*(config["winter_years"][1]-config["winter_years"][0]+1)
                maps["sufficient_support"] = maps["count"] >= np.ceil(.9*nominal)
                maps["supported_slope"] = maps.slope.where(maps.sufficient_support)
                maps.supported_slope.attrs.update(maps.slope.attrs)
                maps.attrs.update(minimum_case_fraction=.9, nominal_cases=nominal,
                    support_description="Display/comparison requires >=90% of nominal member-winter cases; unfiltered slope retained")
                save_dataset(maps, map_path)
            lagged_products(out, config, spec)
            summaries.append(pd.read_csv(out/"event_summary.csv").assign(source=name))
        pd.concat(summaries, ignore_index=True).to_csv(root/"event_summary.csv", index=False)
        (root/"data_provenance.json").write_text(json.dumps(source_info, indent=2)+"\n")
        rows = []
        for variable in config["map_variables"]:
            mod = xr.load_dataset(root/"model"/f"seasonal_{variable}.nc").supported_slope
            ref = xr.load_dataset(root/"e3sm"/f"seasonal_{variable}.nc").supported_slope
            era = xr.load_dataset(root/"reanalysis"/f"seasonal_{variable}.nc").supported_slope
            difference = mod-ref
            save_dataset(difference.rename("difference").to_dataset(), root/f"difference_{variable}.nc")
            for method in mod.anomaly_method.values:
                for season in mod.season.values:
                    m = mod.sel(anomaly_method=method, season=season, lat=slice(-30, 30), lon=slice(120, 280))
                    r = ref.sel(anomaly_method=method, season=season, lat=slice(-30, 30), lon=slice(120, 280))
                    rows.append(dict(variable=variable, method=method, season=season, comparison="MD vs E3SMv2",
                        pattern_r=float(weighted_pattern_correlation(m, r)),
                        rmse=float(weighted_rmse(m, r))*STYLE[variable][0], units=STYLE[variable][2]+" / °C"))
            # ERA vs E3SM is a model/reanalysis climate comparison, not emulator error.
            for season in mod.season.values:
                r = ref.sel(anomaly_method="trajectory", season=season, lat=slice(-30, 30), lon=slice(120, 280))
                e = era.sel(anomaly_method="trajectory", season=season, lat=slice(-30, 30), lon=slice(120, 280))
                rows.append(dict(variable=variable, method="trajectory", season=season, comparison="E3SMv2 vs ERA5/ORAS5",
                    pattern_r=float(weighted_pattern_correlation(r, e)), rmse=float(weighted_rmse(r, e))*STYLE[variable][0],
                    units=STYLE[variable][2]+" / °C"))
        pd.DataFrame(rows).to_csv(root/"map_metrics.csv", index=False)
    if args.stage != "calculate":
        plot_dynamics(root, config)
    logging.info("Finished %s: %s", args.stage, root.resolve())


if __name__ == "__main__":
    main()
