"""Step 1: train the atmosphere autoencoder and save its selected weights."""

from pathlib import Path
import argparse
from dataclasses import asdict, replace
import logging
import torch
import yaml
from monthly_diffusion import MonthlyDiffusionV1, ModelConfig, TrainingConfig
from monthly_diffusion.data.sources import open_canonical_state, open_co2
from monthly_diffusion.data.dataloader import ComponentStates
from monthly_diffusion.training.autoencoder import train_autoencoder
from monthly_diffusion.training.data import validate_training_split
from monthly_diffusion.training.resume import restore_training_state

# Paths and time periods are ordinary Python values; CLI flags can override paths.
RUN_DIR = Path("runs/production")
TRAIN_PERIOD = slice("1966-01", "2006-12")
VALIDATION_PERIOD = slice("1958-01", "1965-12")

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
parser.add_argument("--config", type=Path, help="Optional existing model YAML recipe")
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
torch.set_num_threads(args.threads)
if args.device == "mps" and not torch.backends.mps.is_available():
    raise RuntimeError("MPS requested but unavailable")
if args.epochs is not None and args.epochs < 1:
    parser.error("--epochs must be positive")
checkpoint = args.run / "model.pt"
# The atmosphere step creates the shared checkpoint and fits physical statistics
# on the training years only. The two networks still have independent weights.
settings = ModelConfig(
    ae_width=102,
    latent_channels=4,
    mesh_levels=(4, 3, 2),
    refiners=(6, 6, 4),
    rotation=(17, 23, 11),
    centering="annual_zonal",
    transition_width=132,
    transition_depth=4,
    transition_heads=4,
    transition_ffn_ratio=1.55,
    time_embedding_channels=64,
    coupling="jacobi",
    cross_gradients=False,
    atmosphere_tau=0.5,
    ocean_tau=6.0,
    autoencoder_training=TrainingConfig(batch_size=4, max_epochs=1000, patience=5),
    diffusion_training=TrainingConfig(
        batch_size=4, max_epochs=1000, patience=5, rollout_steps=3, loss_weight=0.01
    ),
)
if args.config is not None:
    settings = ModelConfig.from_yaml(args.config)
if checkpoint.exists():
    model = MonthlyDiffusionV1.load(checkpoint, device=args.device)
    if args.config is not None:
        model.config = settings
        restore_training_state(model, checkpoint)
else:
    model = None
if model is not None and (
    "atmosphere_autoencoder" in model.stages or "autoencoder" in model.stages
):
    if not args.resume:
        raise FileExistsError(
            "The atmosphere AE is already selected; use --resume to keep it"
        )
    logging.info("Atmosphere AE already complete; keeping its selected weights")
    raise SystemExit(0)
if (args.run / "autoencoder/latest.pt").exists():
    raise ValueError(
        "This is an unfinished paired-AE run; resume with scripts/legacy/train_model.py"
    )
if args.data is None and not (args.run / "data_sources.yaml").exists():
    parser.error("The first stage requires --data, for example examples/e3sm.yaml")
source_path = args.data or args.run / "data_sources.yaml"
source = yaml.safe_load(source_path.read_text())
record = args.run / "data_sources.yaml"
if record.exists() and yaml.safe_load(record.read_text()) != source:
    raise ValueError("Source changed; choose a new run directory")
splits = {
    "training": [TRAIN_PERIOD.start, TRAIN_PERIOD.stop],
    "validation": [VALIDATION_PERIOD.start, VALIDATION_PERIOD.stop],
}
split_record = args.run / "splits.yaml"
if split_record.exists() and yaml.safe_load(split_record.read_text()) != splits:
    raise ValueError("Time periods changed; choose a new run directory")
atmosphere, ocean, statics = open_canonical_state(
    source["state_zarr"], expected_attrs=source.get("expected_attrs")
)
co2 = open_co2(source["co2_netcdf"], expected_sha256=source.get("co2_sha256"))
train_atmos, train_ocean = (
    atmosphere.sel(time=TRAIN_PERIOD),
    ocean.sel(time=TRAIN_PERIOD),
)
val_atmos, val_ocean = (
    atmosphere.sel(time=VALIDATION_PERIOD),
    ocean.sel(time=VALIDATION_PERIOD),
)
args.run.mkdir(parents=True, exist_ok=True)
if model is None:
    model = MonthlyDiffusionV1(statics=statics, config=settings, device=args.device)
    model.prepare(train_atmos, train_ocean, co2=co2)
    model.save(checkpoint)
record.write_text(yaml.safe_dump(source, sort_keys=False))
validate_training_split(model, train_atmos, train_ocean, (val_atmos, val_ocean), co2)
split_record.write_text(yaml.safe_dump(splits))
if not model.initialization_moments:
    model.fit_stochastic_initialization(train_atmos, train_ocean)
    model.save(checkpoint)

# Edit these stage parameters here, or pass --epochs for a short experiment.
training = replace(model.config.autoencoder_training)
if args.epochs is not None:
    training.max_epochs = args.epochs
model.config.save(args.run / "config.yaml")
training_fields = train_atmos
validation_fields = val_atmos
if args.load_in_memory:
    training_fields.load()
    validation_fields.load()
training_data = ComponentStates(training_fields, "atmosphere", model.preprocessor, co2)
validation_data = ComponentStates(
    validation_fields, "atmosphere", model.preprocessor, co2
)
static_fields = (
    model.preprocessor.static_tensor(model.statics).to(model.device).unsqueeze(0)
)
weights = torch.tensor(
    [v.weight for v in model.config.variables if v.component == "atmosphere"]
)

# Only this component enters the optimizer; the other AE and diffusion stay fixed.
model.autoencoder.requires_grad_(False)
network = model.autoencoder.atmosphere.requires_grad_(True)
selected = train_autoencoder(
    network,
    training_data,
    validation_data,
    statics=static_fields,
    latitudes=torch.tensor(model.statics.lat.values),
    weights=weights,
    config=training,
    directory=args.run / "atmosphere",
    resume=args.resume,
)
model.stages["atmosphere_autoencoder"] = {
    key: value for key, value in selected.items() if key != "history"
}
model.stages["atmosphere_autoencoder"]["training_config"] = asdict(training)
model.save(checkpoint)
logging.info("Saved selected atmosphere AE in %s", checkpoint)
