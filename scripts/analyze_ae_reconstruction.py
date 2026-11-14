"""Step 6: compare AE reconstructions with the reference data."""

# Make the repository workflow helpers available when running this file directly.
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workflows.run_files import check_script_arguments
import dask
import xarray as xr
from workflows.data.run_data import load_reference_data, reference_path
from workflows.inference.reconstruction import open_reconstruction
from workflows.analysis.trajectory import compare_trajectories
from workflows.analysis.long_rollout import save_dataset

check_script_arguments()

RUN_DIR = Path(__file__).resolve().parent / "runs/test_gridpoint_log1p_mtpr_masked"
BASELINE = (1966, 2006)
EARLY, LATE = (1850, 1880), (2070, 2100)

INPUT = RUN_DIR / "ae_reconstruction"
OUTPUT = RUN_DIR / "analysis/ae"
VARIABLES = None  # e.g. ["t2m", "sst"]; None means all physical fields
SKIP_FORCED_RESPONSE = False
MAKE_PLOTS = True

atmosphere, ocean, _ = load_reference_data(RUN_DIR)
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
    from workflows.analysis.trajectory_plots import plot_series, plot_maps

# Fit baseline means, then accumulate diagnostics one year at a time.
# Only the small final time series and maps stay in memory between variables.
for index, variable in enumerate(variables, start=1):
    started = perf_counter()
    print(f"Analyzing {variable} ({index}/{len(variables)})...", flush=True)
    with dask.config.set(scheduler="threads", num_workers=4):
        series, maps = compare_trajectories(
            reference[variable],
            generated[variable],
            reconstruction=True,
            baseline=BASELINE,
            early=None if SKIP_FORCED_RESPONSE else EARLY,
            late=LATE,
        )
    for result in (series, maps):
        result.attrs.update(
            source=str(reference_path(RUN_DIR)), generated_archive=str(archive)
        )
    save_dataset(series, output / f"{variable}_series.nc")
    save_dataset(maps, output / f"{variable}_maps.nc")
    if MAKE_PLOTS:
        plot_series(series, output / f"{variable}_series.png")
        plot_maps(maps, output / f"{variable}_maps.png")
    print(
        f"Saved {variable} diagnostics to {output} ({perf_counter() - started:.1f} s)",
        flush=True,
    )
generated.close()
reference.close()
