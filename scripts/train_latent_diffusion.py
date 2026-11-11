"""Step 3: freeze the selected AEs and train coupled diffusion on cached latents."""

from pathlib import Path
import argparse
from dataclasses import asdict, replace
import logging
import torch
import yaml
from monthly_diffusion import MonthlyDiffusionV1
from monthly_diffusion.data.sources import open_canonical_state, open_co2
from monthly_diffusion.data.dataloader import (
    PhysicalStates,
    LatentSequences,
    co2_values,
)
from monthly_diffusion.training.data import validate_training_split
from monthly_diffusion.training.latents import cache_latents
from monthly_diffusion.training.coupled_training import train_coupled_diffusion

RUN_DIR = Path("runs/production")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, default=RUN_DIR)
parser.add_argument(
    "--data", type=Path, help="Defaults to the source record saved by AE training"
)
parser.add_argument("--device", default=None)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--epochs", type=int)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
torch.set_num_threads(args.threads)
if args.epochs is not None and args.epochs < 1:
    parser.error("--epochs must be positive")
checkpoint = args.run / "model.pt"
model = MonthlyDiffusionV1.load(checkpoint, device=args.device)
if "autoencoder" not in model.stages:
    raise ValueError(
        "Finish train_atmosphere_autoencoder.py and train_ocean_autoencoder.py first"
    )
if "diffusion" in model.stages:
    if not args.resume:
        raise FileExistsError("Diffusion is already selected; use --resume to keep it")
    logging.info("Diffusion already complete; keeping the selected checkpoint")
    raise SystemExit(0)

source = yaml.safe_load((args.data or args.run / "data_sources.yaml").read_text())
if source != yaml.safe_load((args.run / "data_sources.yaml").read_text()):
    raise ValueError("Source changed; choose a new run directory")
atmosphere, ocean, _ = open_canonical_state(
    source["state_zarr"], expected_attrs=source.get("expected_attrs")
)
co2 = open_co2(source["co2_netcdf"], expected_sha256=source.get("co2_sha256"))
train_atmos = atmosphere.sel(time=model.preprocessor.training_times)
train_ocean = ocean.sel(time=model.preprocessor.training_times)
# Old paired runs used this validation period and may not have a split record.
split_record = args.run / "splits.yaml"
splits = (
    yaml.safe_load(split_record.read_text())
    if split_record.exists()
    else {"validation": ["1958-01", "1965-12"]}
)
validation_period = slice(*splits["validation"])
val_atmos, val_ocean = (
    atmosphere.sel(time=validation_period),
    ocean.sel(time=validation_period),
)
validate_training_split(model, train_atmos, train_ocean, (val_atmos, val_ocean), co2)

# Legacy combined AE checkpoints also work here. Validate or create their cache.
model.autoencoder.eval().requires_grad_(False)
training_data = PhysicalStates(
    train_atmos, train_ocean, model.preprocessor, co2_values(co2, train_atmos.time)
)
validation_data = PhysicalStates(
    val_atmos, val_ocean, model.preprocessor, co2_values(co2, val_atmos.time)
)
cache = cache_latents(model, training_data, validation_data, args.run)
model.save(checkpoint)

# These are ordinary training parameters; edit them here for a new experiment.
training = replace(model.config.diffusion_training)
if args.epochs is not None:
    training.max_epochs = args.epochs
model.config.save(args.run / "config.yaml")
training_sequences = LatentSequences(
    cache["training"]["atmosphere"],
    cache["training"]["ocean"],
    cache["training_time"],
    training.rollout_steps,
)
validation_sequences = LatentSequences(
    cache["validation"]["atmosphere"],
    cache["validation"]["ocean"],
    cache["validation_time"],
    training.rollout_steps,
)
selected = train_coupled_diffusion(
    model.transition,
    training_sequences,
    validation_sequences,
    config=training,
    directory=args.run / "diffusion",
    resume=args.resume,
)
model.stages["diffusion"] = {
    key: value for key, value in selected.items() if key != "history"
}
model.stages["diffusion"]["training_config"] = asdict(training)
model.save(checkpoint)
logging.info("Saved the complete production model to %s", checkpoint)
