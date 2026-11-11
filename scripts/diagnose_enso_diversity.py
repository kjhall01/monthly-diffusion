#!/usr/bin/env python
"""Extend E/C event distributions without regenerating lagged dynamics plots."""
import argparse
import hashlib
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import yaml

from monthly_diffusion.analysis.archives import ClimateArchive
from monthly_diffusion.analysis.enso_dynamics import detect_events
from monthly_diffusion.analysis.enso_diversity import extract_boxes, regional_trajectory_anomalies, event_summary
from monthly_diffusion.analysis.enso_dynamics_plots import plot_ec
from monthly_diffusion.analysis.long_rollout import save_dataset
from monthly_diffusion.data.configs import Variable
from monthly_diffusion.data.dataloader import validate_units


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--plots-only", action="store_true")
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
    code_paths = [Path(__file__), *list((Path(__file__).parents[1]/"monthly_diffusion/analysis").glob("enso_*.py")),
                  Path(__file__).parents[1]/"monthly_diffusion/analysis/archives.py"]
    (root/("code_provenance_plots.json" if args.plots_only else "code_provenance.json")).write_text(
        json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths}, indent=2)+"\n")
    if not args.plots_only:
        summaries, provenance = [], {}
        for name, spec in config["sources"].items():
            out = root/name
            out.mkdir(exist_ok=True)
            archive = ClimateArchive(spec["paths"], forecast=spec.get("forecast", False))
            validate_units(xr.Dataset({"sst": xr.DataArray(attrs={"units": archive.units["sst"]})}),
                           [Variable("sst", "", "degC")])
            logging.info("%s: using ALL %d members: %s", name, len(archive.members), archive.members)
            source = dict(paths=list(map(str, archive.paths)), members=list(map(str, archive.members)),
                calendar=spec["calendar"], units=archive.units["sst"],
                checkpoint=archive.record.get("checkpoint_sha256"),
                support_paths=list(map(str, map(Path, spec.get("support_paths", spec["paths"])))),
                training_member_included=str(config["training_member"]) in list(map(str, archive.members)))
            if archive.forecast:
                source["forecast_record_sha256"] = hashlib.sha256((archive.paths[0]/"forecast.json").read_bytes()).hexdigest()
            else:
                source["dataset_attrs"] = []
                for path in archive.paths:
                    with xr.open_zarr(path, chunks=None) as ds:
                        source["dataset_attrs"].append(ds.attrs)
            identity_path = out/"source_identity.json"
            if identity_path.exists() and json.loads(identity_path.read_text()) != source:
                raise ValueError("Source identity changed; choose a new output directory")
            identity_path.write_text(json.dumps(source, indent=2)+"\n")
            raw_path = out/"raw_box_indices.nc"
            if raw_path.exists():
                raw = xr.load_dataset(raw_path)
            else:
                support_archive = ClimateArchive(spec.get("support_paths", spec["paths"]))
                raw, support = extract_boxes(archive, support_archive, config["analysis_years"])
                save_dataset(support.to_dataset(), out/"sst_support_roi.nc")
                save_dataset(raw, raw_path)
            indices, fits = regional_trajectory_anomalies(raw)
            save_dataset(indices, out/"indices.nc")
            save_dataset(fits, out/"anomaly_fits.nc")
            events, smoothed = detect_events(indices, threshold=config["threshold_degC"],
                peak_years=config["analysis_years"], max_lag=0)
            save_dataset(smoothed, out/"smoothed_indices.nc")
            events.to_csv(out/"events.csv", index=False)
            table = event_summary(events, indices).assign(source=name)
            table.to_csv(out/"event_summary.csv", index=False)
            summaries.append(table)
            used = events.loc[events.used.astype(bool)]
            used.groupby(["member", "phase"], observed=True).agg(events=("E", "size"),
                mean_E=("E", "mean"), mean_C=("C", "mean"), max_E=("E", "max")).to_csv(out/"events_by_member.csv")
            requested = pd.date_range(f"{config['analysis_years'][0]}-01-01",
                                      f"{config['analysis_years'][1]}-12-01", freq="MS")
            source.update(first_month=str(raw.time.values[0])[:10], last_month=str(raw.time.values[-1])[:10],
                months=raw.sizes["time"], missing_requested_months=[str(t.date()) for t in requested.difference(raw.time.values)],
                used_events=int(events.used.sum()), excluded_events=int((~events.used.astype(bool)).sum()),
                excluded_boundary_censored=int(events.boundary_censored.sum()))
            provenance[name] = source
            logging.info("%s: %d used events; %d censored", name, source["used_events"], source["excluded_events"])
        pd.concat(summaries, ignore_index=True).to_csv(root/"event_summary.csv", index=False)
        (root/"data_provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42}):
        plot_ec(root, config)
    logging.info("Finished: %s", root.resolve())


if __name__ == "__main__":
    main()
