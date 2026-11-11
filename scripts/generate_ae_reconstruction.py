"""Step 4: reconstruct the reference chronology through the selected frozen AEs."""

from pathlib import Path
import argparse
import json
import logging
import numpy as np
import torch
import yaml
from monthly_diffusion import MonthlyDiffusionV1
from monthly_diffusion.data.sources import open_canonical_state, open_co2
from monthly_diffusion.data.dataloader import validate_months, co2_values
from monthly_diffusion.inference.reconstruction import (
    checkpoint_digest,
    prepare_reconstruction_archive,
    completed_year,
    write_reconstruction_year,
)

RUN_DIR = Path("runs/production")
START, END = "1850-01", "2100-12"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, default=RUN_DIR)
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--data", type=Path)
parser.add_argument("--output", type=Path)
parser.add_argument("--start", default=START)
parser.add_argument("--end", default=END)
parser.add_argument("--device", default=None)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
torch.set_num_threads(args.threads)
checkpoint = args.checkpoint or args.run / "model.pt"
model = MonthlyDiffusionV1.load(checkpoint, device=args.device)
if "autoencoder" not in model.stages or not model.latent_normalizers:
    raise ValueError(
        "Finish both AE stages and their latent cache before reconstruction"
    )
source = yaml.safe_load((args.data or args.run / "data_sources.yaml").read_text())
atmosphere, ocean, _ = open_canonical_state(
    source["state_zarr"], expected_attrs=source.get("expected_attrs")
)
co2 = open_co2(source["co2_netcdf"], expected_sha256=source.get("co2_sha256"))
atmosphere = atmosphere.sel(time=slice(args.start, args.end))
ocean = ocean.sel(time=slice(args.start, args.end))
model._validate_pair(atmosphere, ocean)
validate_months(atmosphere.time)
if not atmosphere.sizes["time"]:
    raise ValueError("The requested reconstruction period is empty")
# Refuse silently truncated requests. Supply the actual coverage for shorter sources.
if (
    str(atmosphere.time.values[0])[:7] != args.start
    or str(atmosphere.time.values[-1])[:7] != args.end
):
    raise ValueError("Source does not cover --start through --end")
forcing = co2_values(co2, atmosphere.time)
metadata = dict(
    format_version=1,
    kind="autoencoder_reconstruction",
    checkpoint_sha256=checkpoint_digest(checkpoint),
    source=source,
    source_attrs=dict(atmosphere.attrs),
    times=[str(t) for t in atmosphere.time.values],
    co2_ppm=forcing.values.tolist(),
    variables={name: list(fields) for name, fields in model.names.items()},
    latent_coordinates="fixed_transition_normalized",
)
metadata = json.loads(json.dumps(metadata, default=str))
output = prepare_reconstruction_archive(
    args.output or args.run / "ae_reconstruction", metadata, resume=args.resume
)

# Encode and decode observed states, with their own calendar and CO2. No AR step.
# One year is materialized at a time; encode/decode use smaller internal batches.
for year in np.unique(atmosphere.time.dt.year.values):
    atmos_year = atmosphere.sel(time=str(year))
    ocean_year = ocean.sel(time=str(year))
    destination = output / f"year-{year:04d}.zarr"
    if completed_year(destination, atmos_year.time):
        logging.info("Keeping completed reconstruction %s", year)
        continue
    latents = model.encode(atmos_year, ocean_year, co2=co2)
    fields = model.decode(*latents, co2=co2)
    write_reconstruction_year(
        fields, latents, destination, co2=co2_values(co2, atmos_year.time)
    )
    logging.info("Saved reconstruction and latents for %s", year)
