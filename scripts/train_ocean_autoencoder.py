"""Step 2: train the ocean autoencoder and save its selected weights."""

from pathlib import Path
import argparse
from dataclasses import asdict, replace
import logging
import torch
import yaml
from monthly_diffusion import MonthlyDiffusionV1
from monthly_diffusion.data.sources import open_canonical_state, open_co2
from monthly_diffusion.data.dataloader import (
    ComponentStates,
    PhysicalStates,
    co2_values,
)
from monthly_diffusion.training.autoencoder import train_autoencoder
from monthly_diffusion.training.data import validate_training_split
from monthly_diffusion.training.latents import cache_latents

# Paths and time periods are ordinary Python values; CLI flags can override paths.
RUN_DIR = Path("runs/production")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, default=RUN_DIR)
parser.add_argument(
    "--data", type=Path, help="Source YAML; later stages reuse run/data_sources.yaml"
)
parser.add_argument("--device", default=None)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument(
    "--epochs", type=int, help="Override the epoch limit for this component"
)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--load-in-memory", action="store_true")
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
torch.set_num_threads(args.threads)
if args.device == "mps" and not torch.backends.mps.is_available():
    raise RuntimeError("MPS requested but unavailable")
if args.epochs is not None and args.epochs < 1:
    parser.error("--epochs must be positive")
checkpoint = args.run / "model.pt"
if not checkpoint.exists():
    raise FileNotFoundError("Run train_atmosphere_autoencoder.py first")
model = MonthlyDiffusionV1.load(checkpoint, device=args.device)
if "atmosphere_autoencoder" not in model.stages and "autoencoder" not in model.stages:
    raise ValueError("Finish atmosphere training before training the ocean")
if "autoencoder" in model.stages:
    if not args.resume:
        raise FileExistsError(
            "The ocean AE is already selected; use --resume to keep it"
        )
    logging.info("Ocean AE already complete; keeping its selected weights")
    raise SystemExit(0)
source_path = args.data or args.run / "data_sources.yaml"
source = yaml.safe_load(source_path.read_text())
record = args.run / "data_sources.yaml"
if record.exists() and yaml.safe_load(record.read_text()) != source:
    raise ValueError("Source changed; choose a new run directory")
atmosphere, ocean, statics = open_canonical_state(
    source["state_zarr"], expected_attrs=source.get("expected_attrs")
)
co2 = open_co2(source["co2_netcdf"], expected_sha256=source.get("co2_sha256"))
train_atmos, train_ocean = (
    atmosphere.sel(time=model.preprocessor.training_times),
    ocean.sel(time=model.preprocessor.training_times),
)
splits = yaml.safe_load((args.run / "splits.yaml").read_text())
validation_period = slice(*splits["validation"])
val_atmos, val_ocean = (
    atmosphere.sel(time=validation_period),
    ocean.sel(time=validation_period),
)
validate_training_split(model, train_atmos, train_ocean, (val_atmos, val_ocean), co2)
if not model.initialization_moments:
    model.fit_stochastic_initialization(train_atmos, train_ocean)
    model.save(checkpoint)

# Edit these stage parameters here, or pass --epochs for a short experiment.
training = replace(model.config.autoencoder_training)
if args.epochs is not None:
    training.max_epochs = args.epochs
model.config.save(args.run / "config.yaml")
training_fields = train_ocean
validation_fields = val_ocean
if args.load_in_memory:
    training_fields.load()
    validation_fields.load()
training_data = ComponentStates(training_fields, "ocean", model.preprocessor, co2)
validation_data = ComponentStates(validation_fields, "ocean", model.preprocessor, co2)
static_fields = (
    model.preprocessor.static_tensor(model.statics).to(model.device).unsqueeze(0)
)
weights = torch.tensor(
    [v.weight for v in model.config.variables if v.component == "ocean"]
)

# Resume may have already selected ocean weights before cache creation finished.
if "ocean_autoencoder" in model.stages:
    if not args.resume:
        raise FileExistsError(
            "Ocean weights are already selected; use --resume to finish the cache"
        )
else:
    # Only this component enters the optimizer; the other AE and diffusion stay fixed.
    model.autoencoder.requires_grad_(False)
    network = model.autoencoder.ocean.requires_grad_(True)
    selected = train_autoencoder(
        network,
        training_data,
        validation_data,
        statics=static_fields,
        latitudes=torch.tensor(model.statics.lat.values),
        weights=weights,
        config=training,
        directory=args.run / "ocean",
        resume=args.resume,
    )
    model.stages["ocean_autoencoder"] = {
        key: value for key, value in selected.items() if key != "history"
    }
    model.stages["ocean_autoencoder"]["training_config"] = asdict(training)
    model.save(checkpoint)

# Both selected AEs are now frozen. Encode the splits once and fit fixed latent
# scaling on training data, so reconstruction is usable before diffusion trains.
model.autoencoder.eval().requires_grad_(False)
physical_train = PhysicalStates(
    train_atmos, train_ocean, model.preprocessor, co2_values(co2, train_atmos.time)
)
physical_validation = PhysicalStates(
    val_atmos, val_ocean, model.preprocessor, co2_values(co2, val_atmos.time)
)
cache_latents(model, physical_train, physical_validation, args.run)
model.stages["autoencoder"] = {
    "selection": "independent component validation",
    "atmosphere": model.stages["atmosphere_autoencoder"],
    "ocean": model.stages["ocean_autoencoder"],
}
model.save(checkpoint)
logging.info("Saved selected ocean AE in %s", checkpoint)
