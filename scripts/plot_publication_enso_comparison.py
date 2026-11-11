#!/usr/bin/env python
"""Four-source, matched-period ENSO statistics from physical SST archives.

No new inference. Read tropical SST only in small blocks, cache raw box means,
fit each member's monthly means plus linear trend over the COMMON interval,
then reuse package Welch/ACF definitions and fixed-bandwidth density estimates.
No ensemble subtraction or variance standardization. The common anomaly fit
is deliberately different from ensemble-residual teleconnection maps: there
is only one reanalysis trajectory. This is a descriptive historical comparison,
not out-of-sample validation. Complete source identity and all curves are saved.
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import xarray as xr
import yaml

from monthly_diffusion.analysis.archives import ClimateArchive
from monthly_diffusion.analysis.enso_diversity import extract_boxes, regional_trajectory_anomalies
from monthly_diffusion.analysis.enso_comparison import enso_statistics, plot_enso_comparison
from monthly_diffusion.analysis.long_rollout import save_dataset


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
        raise ValueError("Configuration changed; choose a new directory")
    marker.write_text(json.dumps(config, indent=2)+"\n")
    raw, provenance = {}, {}
    for source, spec in config["sources"].items():
        out = root/source
        out.mkdir(exist_ok=True)
        archive = ClimateArchive(spec["paths"], forecast=spec.get("forecast", False))
        if archive.units["sst"] not in ("degC", "°C", "degree_Celsius"):
            raise ValueError("Require SST in Celsius")
        identity = dict(paths=list(map(str, archive.paths)), members=list(map(str, archive.members)),
            support_paths=spec["support_paths"], units=archive.units["sst"],
            checkpoint_sha256=archive.record.get("checkpoint_sha256"))
        if archive.forecast:
            identity["forecast_record_sha256"] = hashlib.sha256((archive.paths[0]/"forecast.json").read_bytes()).hexdigest()
        else:
            identity["source_metadata_sha256"] = {str(p): hashlib.sha256((p/".zmetadata").read_bytes()).hexdigest() for p in archive.paths}
        identity_path = out/"source_identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("Archive identity changed; choose a new directory")
        identity_path.write_text(json.dumps(identity, indent=2)+"\n")
        raw_path = out/"raw_box_indices.nc"
        if raw_path.exists():
            raw[source] = xr.load_dataset(raw_path)
        else:
            support = ClimateArchive(spec["support_paths"])
            logging.info("Extracting %s: %d members, tropical SST only", source, len(archive.members))
            raw[source], mask = extract_boxes(archive, support, config["years"])
            save_dataset(raw[source], raw_path)
            save_dataset(mask.to_dataset(), out/"sst_support_roi.nc")
        provenance[source] = identity
        provenance[source]["raw_indices_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    # All curves use exactly the same months, including any partial edge year.
    common = np.asarray(next(iter(raw.values())).time)
    for ds in raw.values():
        common = np.intersect1d(common, ds.time.values)
    if not len(common):
        raise ValueError("No shared analysis months")
    coverage = [str(common[0])[:10], str(common[-1])[:10]]
    anomalies = {}
    for source, ds in raw.items():
        indices, fits = regional_trajectory_anomalies(ds.sel(time=common))
        anomalies[source] = indices.nino34.assign_attrs(units="degC", anomaly_definition=indices.attrs["anomaly_definition"])
        save_dataset(indices, root/source/"indices.nc")
        save_dataset(fits, root/source/"anomaly_fits.nc")
    bound = np.ceil(2*(max(float(np.abs(x).max()) for x in anomalies.values())+3*config["kde_bandwidth_degC"]))/2
    density_grid = np.linspace(-bound, bound, 601)
    statistics, rows = {}, []
    for source, index in anomalies.items():
        ds = enso_statistics(index, density_grid=density_grid,
            bandwidth_degC=config["kde_bandwidth_degC"], segment_months=config["segment_months"],
            max_lag=config["max_lag_months"])
        statistics[source] = ds
        save_dataset(ds, root/source/"statistics.nc")
        ds.psd.mean("member").to_dataframe(name="psd_degC2_year").to_csv(root/source/"psd.csv")
        ds.acf.mean("member").to_dataframe(name="acf").to_csv(root/source/"acf.csv")
        ds.density.mean("member").to_dataframe(name="density_per_degC").to_csv(root/source/"density.csv")
        ds[["pooled_seasonal_std", "pooled_seasonal_amplification"]].to_dataframe().to_csv(root/source/"seasonal_amplitude.csv")
        psd = ds.psd.mean("member").isel(frequency=slice(1, None))
        acf = ds.acf.mean("member")
        month = int(ds.pooled_seasonal_amplification.idxmax())
        rows.append(dict(source=source, members=index.sizes["member"], months_per_member=index.sizes["time"],
            std_degC=float(index.std()), peak_psd_period_years=1/float(psd.idxmax()),
            minimum_acf=float(acf.min()), minimum_acf_lag_months=int(acf.idxmin()),
            peak_variance_month=month, seasonal_max_over_min=float(ds.pooled_seasonal_std.max()/ds.pooled_seasonal_std.min())))
    pd.DataFrame(rows).to_csv(root/"summary.csv", index=False)
    plot_enso_comparison(statistics, config, root/"enso_comparison", coverage)
    provenance.update(coverage=coverage, matched_months=len(common),
        anomaly_definition="per-trajectory joint OLS: 12 monthly intercepts + linear trend over common comparison period; no ensemble subtraction, no variance normalization",
        frequency_estimator=f"Welch Hann, {config['segment_months']} months, 50% overlap; memberwise then mean, PSD in degC^2 year",
        member_bands="10th–90th percentile across member statistics, NOT confidence intervals",
        seasonal_amplification="pooled calendar-month standard deviation divided by pooled overall standard deviation",
        limitation="Descriptive comparison including training years; detrending cannot perfectly separate nonlinear forced change from internal variability. Reanalysis has only one realization.")
    (root/"metadata.json").write_text(json.dumps(provenance, indent=2)+"\n")
    code_paths = [Path(__file__), *[Path(__file__).parents[1]/"monthly_diffusion/analysis"/f"{x}.py"
        for x in ("enso_comparison", "enso_diversity", "indices", "spectra", "archives")]]
    (root/"code_provenance.json").write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths}, indent=2)+"\n")
    logging.info("Saved matched %s through %s comparison: %s", *coverage, root.resolve())


if __name__ == "__main__":
    main()
