"""Step 6: compare AE reconstructions with the reference data."""

from pathlib import Path
from monthly_diffusion.run_files import check_script_arguments
import dask
import xarray as xr
from monthly_diffusion.run_files import read_run
from monthly_diffusion.data.sources import open_canonical_state
from monthly_diffusion.inference.reconstruction import open_reconstruction
from monthly_diffusion.analysis.trajectory import compare_trajectories
from monthly_diffusion.analysis.long_rollout import save_dataset

check_script_arguments()

RUN_DIR = Path("runs/production")
BASELINE = (1966, 2006)
EARLY, LATE = (1850, 1880), (2070, 2100)

INPUT = RUN_DIR / "ae_reconstruction"
OUTPUT = RUN_DIR / "analysis/ae"
VARIABLES = None  # e.g. ["t2m", "sst"]; None means all physical fields
SKIP_FORCED_RESPONSE = False
MAKE_PLOTS = True

source = read_run(RUN_DIR)["source"]
atmosphere, ocean, _ = open_canonical_state(
    source["state_zarr"], expected_attrs=source.get("expected_attrs")
)
reference = xr.merge([atmosphere, ocean], join="exact")
archive = INPUT
generated = open_reconstruction(archive)
# Exact date selection raises if any reconstructed month is absent in the source.
reference = reference.sel(time=generated.time)
variables = VARIABLES or [
    name for name in reference.data_vars if name in generated.data_vars
]
if not variables or any(
    name not in reference or name not in generated for name in variables
):
    raise ValueError(
        "Choose physical variables present in both reference and generated data"
    )
output = OUTPUT
output.mkdir(parents=True, exist_ok=True)
if MAKE_PLOTS:
    from monthly_diffusion.analysis.trajectory_plots import plot_series, plot_maps

# Process one variable in monthly blocks. Only reduced time series and maps are
# materialized, keeping the 1850--2100 physical fields out of memory.
for variable in variables:
    series, maps = compare_trajectories(
        reference[variable].chunk({"time": 12}),
        generated[variable].chunk({"time": 12}),
        reconstruction=True,
        baseline=BASELINE,
        early=None if SKIP_FORCED_RESPONSE else EARLY,
        late=LATE,
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
    if MAKE_PLOTS:
        plot_series(series, output / f"{variable}_series.png")
        plot_maps(maps, output / f"{variable}_maps.png")
    print(f"Saved {variable} diagnostics to {output}")
generated.close()
reference.close()
