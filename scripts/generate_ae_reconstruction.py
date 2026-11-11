"""Reconstruct the reference chronology through the selected frozen AEs."""

from pathlib import Path
from monthly_diffusion.run_files import check_script_arguments
import json
import numpy as np
import torch
from monthly_diffusion.model_loading import load_models
from monthly_diffusion.run_files import read_run, model_identity
from monthly_diffusion.data.sources import open_canonical_state, open_co2
from monthly_diffusion.data.dataloader import validate_months, co2_values
from monthly_diffusion.inference.reconstruction import (
    prepare_reconstruction_archive,
    completed_year,
    write_reconstruction_year,
)

check_script_arguments()

RUN_DIR = Path("runs/production")
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
BATCH_SIZE = 4

torch.set_num_threads(THREADS)
model = load_models(RUN_DIR, device=DEVICE, diffusion=False, batch_size=BATCH_SIZE)
source = read_run(RUN_DIR)["source"]
atmosphere, ocean, _ = open_canonical_state(
    source["state_zarr"], expected_attrs=source.get("expected_attrs")
)
co2 = open_co2(source["co2_netcdf"], expected_sha256=source.get("co2_sha256"))
atmosphere = atmosphere.sel(time=slice(START, END))
ocean = ocean.sel(time=slice(START, END))
model._validate_pair(atmosphere, ocean)
validate_months(atmosphere.time)
if (
    str(atmosphere.time.values[0])[:7] != START
    or str(atmosphere.time.values[-1])[:7] != END
):
    raise ValueError("Source does not cover START through END")
forcing = co2_values(co2, atmosphere.time)
metadata = dict(
    format_version=2,
    kind="autoencoder_reconstruction",
    models=model_identity(RUN_DIR),
    source=source,
    source_attrs=dict(atmosphere.attrs),
    times=[str(t) for t in atmosphere.time.values],
    co2_ppm=forcing.values.tolist(),
    variables=model.names,
    latent_coordinates="fixed_transition_normalized",
)
metadata = json.loads(json.dumps(metadata, default=str))
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
