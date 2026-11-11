"""Step 7: compare free-running climate statistics with the reference data."""

from pathlib import Path
import argparse
import logging
import dask
import xarray as xr
import yaml
from monthly_diffusion.data.sources import open_canonical_state
from monthly_diffusion.inference.archive import open_rollout
from monthly_diffusion.analysis.trajectory import compare_trajectories
from monthly_diffusion.analysis.long_rollout import save_dataset

RUN_DIR = Path("runs/production")
BASELINE = (1966, 2006)
EARLY, LATE = (1850, 1880), (2070, 2100)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, default=RUN_DIR)
parser.add_argument("--data", type=Path, help="Defaults to run/data_sources.yaml")
parser.add_argument("--input", type=Path, help="Defaults to run/ar_rollout")
parser.add_argument("--output", type=Path, help="Defaults to run/analysis/ar")
parser.add_argument(
    "--variables", nargs="+", help="Defaults to all reconstructed physical variables"
)
parser.add_argument(
    "--baseline", type=int, nargs=2, default=BASELINE, metavar=("FIRST", "LAST")
)
parser.add_argument(
    "--early", type=int, nargs=2, default=EARLY, metavar=("FIRST", "LAST")
)
parser.add_argument(
    "--late", type=int, nargs=2, default=LATE, metavar=("FIRST", "LAST")
)
parser.add_argument(
    "--skip-forced-response",
    action="store_true",
    help="For sources without both response periods",
)
parser.add_argument(
    "--no-plots", action="store_true", help="Save numerical results only"
)
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
source = yaml.safe_load((args.data or args.run / "data_sources.yaml").read_text())
atmosphere, ocean, _ = open_canonical_state(
    source["state_zarr"], expected_attrs=source.get("expected_attrs")
)
reference = xr.merge([atmosphere, ocean], join="exact")
archive = args.input or args.run / "ar_rollout"
generated = open_rollout(archive)
variables = args.variables or [
    name for name in reference.data_vars if name in generated.data_vars
]
if not variables or any(
    name not in reference or name not in generated for name in variables
):
    raise ValueError(
        "Choose physical variables present in both reference and generated data"
    )
output = args.output or args.run / "analysis/ar"
output.mkdir(parents=True, exist_ok=True)
if not args.no_plots:
    from monthly_diffusion.analysis.trajectory_plots import plot_series, plot_maps

# Process one variable in monthly blocks. Only reduced time series and maps are
# materialized, keeping the 1850--2100 physical fields out of memory.
for variable in variables:
    series, maps = compare_trajectories(
        reference[variable].chunk({"time": 12}),
        generated[variable].chunk({"time": 12}),
        reconstruction=False,
        baseline=args.baseline,
        early=None if args.skip_forced_response else args.early,
        late=args.late,
    )
    with dask.config.set(scheduler="threads", num_workers=4):
        results = xr.merge([series, maps]).compute()
    series, maps = results[list(series.data_vars)], results[list(maps.data_vars)]
    for result in (series, maps):
        result.attrs.update(
            source=str(source["state_zarr"]), generated_archive=str(archive)
        )
    save_dataset(series, output / f"{variable}_series.nc")
    save_dataset(maps, output / f"{variable}_maps.nc")
    if not args.no_plots:
        plot_series(series, output / f"{variable}_series.png")
        plot_maps(maps, output / f"{variable}_maps.png")
    logging.info("Saved %s diagnostics to %s", variable, output)
generated.close()
reference.close()
