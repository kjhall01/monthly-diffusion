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
import numpy as np
import pandas as pd
import xarray as xr
import yaml

from monthly_diffusion.analysis.archives import ClimateArchive
from monthly_diffusion.analysis.decadal import (
    extract_annual_sst, lanczos_weights, variability_fields, region_weights,
    fit_covariance_eofs, match_eofs, pc_diagnostics,
)
from monthly_diffusion.analysis.enso_dynamics import fixed_sst_support
from monthly_diffusion.analysis.long_rollout import save_dataset


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
        raise ValueError("Configuration differs; use a new output directory")
    marker.write_text(json.dumps(config, indent=2)+"\n")
    code = [Path(__file__), *[Path(__file__).parents[1]/"monthly_diffusion/analysis"/p
        for p in ("decadal.py", "decadal_plots.py", "archives.py", "eofs.py", "climate.py")]]
    provenance = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code}
    (root/("plot_code.json" if args.plots_only else "code_provenance.json")).write_text(json.dumps(provenance, indent=2)+"\n")
    environment = dict(python=platform.python_version(), platform=platform.platform(),
        packages={p: importlib.metadata.version(p) for p in
            ("numpy", "scipy", "xarray", "zarr", "matplotlib", "cartopy", "pandas")})
    (root/("plot_environment.json" if args.plots_only else "environment.json")).write_text(json.dumps(environment, indent=2)+"\n")
    if not args.plots_only:
        run(config, root)
    from monthly_diffusion.analysis.decadal_plots import plot_decadal
    plot_decadal(root, config)
    logging.info("Finished decadal SST analysis: %s", root.resolve())


def run(config, root):
    annual, identities, supports = {}, {}, []
    for name, spec in config["sources"].items():
        archive = ClimateArchive(spec["paths"], forecast=spec.get("forecast", False))
        identities[name] = dict(paths=list(map(str, archive.paths)), members=list(map(str, archive.members)),
            first_month=str(archive.time.values[0]), last_month=str(archive.time.values[-1]),
            months=len(archive.time), units=archive.units["sst"],
            checkpoint=archive.record.get("checkpoint_sha256"),
            metadata_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for base in archive.paths for p in ([base/"forecast.json"] if archive.forecast else [base/".zmetadata"])})
        if archive.forecast and archive.ocean_support is not None:
            supports.append(archive.ocean_support)
    # Training-only masks from the two reference families, intersected with
    # any explicitly archived decoder support. Identity checks precede caching.
    identity_path = root/"source_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identities:
        raise ValueError("Source archive identity changed; use a new output directory")
    identity_path.write_text(json.dumps(identities, indent=2)+"\n")
    for name, spec in config["supports"].items():
        target = root/f"support_{name}.nc"
        if target.exists():
            mask = xr.load_dataset(target).support.astype(bool)
        else:
            logging.info("Computing training-only %s SST support", name)
            mask = fixed_sst_support(ClimateArchive(spec), config["training"])
            save_dataset(mask.rename("support").to_dataset(), target)
        supports.append(mask)
    supports = xr.align(*supports, join="exact")
    support = xr.concat(supports, dim="source").all("source")
    save_dataset(support.rename("support").to_dataset(), root/"common_fine_ocean_support.nc")
    support_sha = hashlib.sha256(np.asarray(support, dtype=np.uint8).tobytes()).hexdigest()
    for name, spec in config["sources"].items():
        path = root/f"annual_{name}.nc"
        if path.exists():
            ds = xr.load_dataset(path)
            if ds.attrs["support_sha256"] != support_sha:
                raise ValueError("Annual cache has different ocean support")
        else:
            logging.info("Reducing %s SST to annual, %d-degree fields", name, config["coarsen_factor"])
            archive = ClimateArchive(spec["paths"], forecast=spec.get("forecast", False))
            ds = extract_annual_sst(archive, support, config["coarsen_factor"])
            ds.attrs["support_sha256"] = support_sha
            save_dataset(ds, path)
        annual[name] = ds.sst
        if name == config["basis"]["source"]:
            area = ds.wet_area
    # Fixed basis support includes only cells complete in EVERY annual source.
    # Retain exclusions so missing coast/polar cells are auditable.
    valid = xr.concat([np.isfinite(x).all(("member", "year")) for x in annual.values()], dim="source").all("source")
    support_summary = dict(common_fine_cells=int(support.sum()), total_fine_cells=support.size,
        wet_coarse_cells_before_completeness=int((area > 0).sum()),
        wet_coarse_cells_retained=int(((area > 0) & valid).sum()),
        retained_wet_area_fraction=float(area.where(valid, 0.).sum()/area.sum()),
        support_sha256=support_sha)
    (root/"support_summary.json").write_text(json.dumps(support_summary, indent=2)+"\n")
    area = area.where(valid, 0.)
    save_dataset(xr.Dataset(dict(wet_area=area, complete_support=valid)), root/"analysis_support.nc")
    kernel = lanczos_weights(config["cutoff_years"], config["filter_half_width"])
    omega = np.linspace(0., .5, 501)
    lags = np.arange(-config["filter_half_width"], config["filter_half_width"]+1)
    response = np.exp(-2j*np.pi*omega[:, None]*lags)@kernel
    save_dataset(xr.Dataset(dict(coefficients=("lag_years", kernel),
        amplitude_response=("frequency", abs(response))),
        coords=dict(lag_years=lags, frequency=omega)), root/"filter.nc")
    basis_fields = variability_fields(annual[config["basis"]["source"]], config["basis"]["years"], kernel)
    save_dataset(basis_fields, root/"reference_basis_fields.nc")
    bases = {}
    for region, spec in config["regions"].items():
        logging.info("Fitting historical reference covariance EOFs: %s", region)
        bases[region] = fit_covariance_eofs(basis_fields.lowpass.rename(filtered_year="year"),
            region_weights(area, spec["bounds"]), config["modes"])
        save_dataset(bases[region], root/f"basis_{region}.nc")
    summary, matching, coverage, members = [], [], {}, []
    for comparison, spec in config["comparisons"].items():
        logging.info("Comparing %s (%s)", comparison, spec["method"])
        folder = root/comparison
        folder.mkdir(exist_ok=True)
        coverage[comparison] = {}
        for name in spec["sources"]:
            fields = variability_fields(annual[name].where(area > 0), spec["years"], kernel, spec["method"])
            save_dataset(fields, folder/f"{name}_fields.nc")
            coverage[comparison][name] = dict(fields.attrs)
            for region, basis in bases.items():
                product = pc_diagnostics(fields, basis, config["spectrum_segment_years"])
                save_dataset(product, folder/f"{name}_{region}_pcs.nc")
                own = fit_covariance_eofs(fields.lowpass.rename(filtered_year="year"), basis.weight, config["modes"])
                order, signs, similarity = match_eofs(basis, own)
                matched = own.isel(mode=order).assign_coords(mode=basis.mode)
                signs = xr.DataArray(signs, dims="mode", coords=dict(mode=basis.mode))
                matched["pattern"] = matched.pattern*signs
                matched["loading"] = matched.loading*signs
                matched["original_rank"] = ("mode", order+1)
                matched["reference_pattern_cosine"] = ("mode", similarity)
                save_dataset(matched, folder/f"{name}_{region}_independent_eofs.nc")
                w = basis.weight/basis.weight.sum()
                total_var = float((fields.annual_variance*w).sum())
                low_var = float((fields.lowpass_variance*w).sum())
                summary.append(dict(comparison=comparison, source=name, region=region,
                    method=spec["method"], first_year=int(fields.filtered_year[0]),
                    last_year=int(fields.filtered_year[-1]), members=fields.sizes["member"],
                    annual_rms_std_degC=np.sqrt(total_var), lowpass_rms_std_degC=np.sqrt(low_var),
                    lowpass_variance_fraction=low_var/total_var))
                for k, mode in enumerate(basis.mode.values):
                    matching.append(dict(comparison=comparison, source=name, region=region, mode=int(mode),
                        independent_rank=int(order[k]+1), weighted_pattern_cosine=float(similarity[k]),
                        independent_explained_fraction=float(matched.explained_fraction.sel(mode=mode)),
                        common_pc_lowpass_std_degC=float(np.sqrt(product.lowpass_variance.sel(mode=mode).mean("member"))),
                        common_pc_annual_std_degC=float(np.sqrt(product.annual_variance.sel(mode=mode).mean("member"))),
                        common_pc_lowpass_variance_fraction=float(product.lowpass_variance.sel(mode=mode).mean("member")/
                            product.annual_variance.sel(mode=mode).mean("member"))))
                    for member in product.member.values:
                        selected = product.sel(member=member, mode=mode)
                        members.append(dict(comparison=comparison, source=name, region=region, mode=int(mode),
                            member=str(member), lowpass_std_degC=float(np.sqrt(selected.lowpass_variance)),
                            annual_std_degC=float(np.sqrt(selected.annual_variance)),
                            lowpass_fraction=float(selected.lowpass_fraction)))
    pd.DataFrame(summary).to_csv(root/"variance_summary.csv", index=False)
    pd.DataFrame(matching).to_csv(root/"mode_summary.csv", index=False)
    pd.DataFrame(members).to_csv(root/"member_pc_statistics.csv", index=False)
    (root/"coverage.json").write_text(json.dumps(coverage, indent=2)+"\n")
    logging.info("Saved variance/EOF/PC products; rendering figures")


if __name__ == "__main__":
    main()
