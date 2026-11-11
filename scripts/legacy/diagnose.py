"""Small reusable-analysis example, not a second monolithic diagnostics pipeline.

Usage: python scripts/legacy/diagnose.py FORECAST_DIRECTORY REFERENCE... OUTPUT_DIRECTORY
One or more reference files may be supplied.  Multiple single-member E3SM
stores are concatenated along ``member`` before diagnostics.
"""

import argparse
from pathlib import Path
import numpy as np
import xarray as xr
from monthly_diffusion import open_rollout
from monthly_diffusion.analysis import (
    climate_diagnostics,
    internal_variability,
    pdo_loading,
)
from monthly_diffusion.analysis.plotting import plot_map_comparison


from monthly_diffusion.analysis.forced_maps import run_forced_only


p = argparse.ArgumentParser()
p.add_argument("forecast", type=Path)
p.add_argument("reference", type=Path, nargs="+")
p.add_argument("output", type=Path)
p.add_argument("--early", type=int, nargs=2, default=(1851, 1880))
p.add_argument("--late", type=int, nargs=2, default=(2070, 2100))
p.add_argument("--pdo-basis", type=Path, default=None)
p.add_argument(
    "--forced-only",
    action="store_true",
    help="Use the low-memory variable-wise forced-response path",
)
args = p.parse_args()
generated = open_rollout(args.forecast)
references = []
for path in args.reference:
    item = xr.open_dataset(path) if path.is_file() else xr.open_zarr(path)
    if "member" not in item.dims:
        item = item.expand_dims(member=[path.stem])
    references.append(item)
if len(references) == 1:
    reference = references[0]
else:
    reference = xr.concat(
        references,
        dim="member",
        data_vars="minimal",
        coords="minimal",
        compat="equals",
        join="exact",
    )
if not generated.lat.equals(reference.lat) or not generated.lon.equals(reference.lon):
    raise ValueError(
        "Reference and forecast grids differ; explicit regridding is required"
    )
args.output.mkdir(parents=True, exist_ok=True)
if args.forced_only:
    run_forced_only(
        generated, args.reference, args.output, tuple(args.early), tuple(args.late)
    )
    raise SystemExit(0)
reference_training = reference.sel(time=slice("1966", "2006"))
if "member" in reference_training.dims and reference_training.sizes["member"] >= 2:
    reference_training = internal_variability(reference_training)
elif "member" not in reference_training.dims:
    # e3sm-local.yaml points to the r1 training member, not a multi-member
    # ensemble.  A deterministic reference cannot support finite-ensemble
    # subtraction, so use its physical fields directly for the fixed PDO
    # basis and make the limitation explicit in the output metadata.
    reference_training.attrs["internal_variability_note"] = (
        "single reference member; no ensemble subtraction"
    )
else:
    raise ValueError("Reference has a member dimension but fewer than two members")
basis = (
    xr.open_dataarray(args.pdo_basis)
    if args.pdo_basis
    else pdo_loading(reference_training, remove_monthly=False)
)
# NetCDF classic attributes cannot encode boolean scalars produced by the
# climatology/index helpers.
basis.attrs = {
    key: int(value) if isinstance(value, (bool, np.bool_)) else value
    for key, value in basis.attrs.items()
}
basis.to_netcdf(args.output / "reference_pdo_loading.nc")
diagnostics = []
for name, fields in (("reference", reference), ("model", generated)):
    result = climate_diagnostics(
        fields, pdo_basis=basis, early=tuple(args.early), late=tuple(args.late)
    )
    for product, ds in result.items():
        ds.to_netcdf(args.output / f"{name}_{product}.nc")
    diagnostics.append(result)
for variable in ("t2m", "sst", "mtpr"):
    plot_map_comparison(
        diagnostics[0]["forced_response"][variable],
        diagnostics[1]["forced_response"][variable],
        path=args.output / f"forced_{variable}",
        title=variable,
    )
