"""Reconstruct the reference chronology through the selected frozen AEs."""

# Make the repository workflow helpers available when running this file directly.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workflows.run_files import check_script_arguments
import numpy as np
import torch
from monthly_diffusion.loading import load_models
from workflows.data.run_data import load_run_data
from workflows.data.dataloader import validate_months, co2_values
from workflows.inference.reconstruction import (
    reconstruction_metadata,
    prepare_reconstruction_archive,
    completed_year,
    write_reconstruction_year,
)

check_script_arguments()

RUN_DIR = Path(__file__).resolve().parent / "runs/test_gridpoint_log1p_mtpr_masked"
OUTPUT = RUN_DIR / "ae_reconstruction"
START, END = "1850-01", "2100-12"
RESUME = False
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
THREADS = 4
BATCH_SIZE = 24

torch.set_num_threads(THREADS)
model = load_models(RUN_DIR, device=DEVICE, diffusion=False, batch_size=BATCH_SIZE)
atmosphere, ocean, _, co2 = load_run_data(RUN_DIR)
atmosphere = atmosphere.sel(time=slice(START, END))
ocean = ocean.sel(time=slice(START, END))
model._validate_pair(atmosphere, ocean)
validate_months(atmosphere.time)
if (
    str(atmosphere.time.values[0])[:7] != START
    or str(atmosphere.time.values[-1])[:7] != END
):
    raise ValueError("Source does not cover START through END")
metadata = reconstruction_metadata(RUN_DIR, model, atmosphere, co2)
output = prepare_reconstruction_archive(OUTPUT, metadata, resume=RESUME)

# Observed states are encoded and decoded with their own month and CO2.
# One year is written atomically; internal inference uses BATCH_SIZE months.
for year in np.unique(atmosphere.time.dt.year.values):
    atmos_year, ocean_year = atmosphere.sel(time=str(year)), ocean.sel(time=str(year))
    destination = output / f"year-{year:04d}.zarr"
    if completed_year(destination, atmos_year.time):
        print(f"Keeping completed reconstruction {year}")
        continue
    latents = model.encode(atmos_year, ocean_year, co2=co2)
    fields = model.decode(*latents, co2=co2)
    write_reconstruction_year(
        fields, latents, destination, co2=co2_values(co2, atmos_year.time)
    )
    print(f"Saved reconstruction and latents for {year}")
