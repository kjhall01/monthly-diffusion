"""Explicit E3SM/ERA5 example: same split, staged architecture, no hidden downloads.

After `pip install -e .`, run:
    python examples/train_model.py examples/e3sm.yaml --output runs/e3sm
This is intentionally a readable script, not a framework CLI.
"""

import argparse
import logging
from pathlib import Path
import yaml
import torch
from monthly_diffusion import MonthlyDiffusionV1, ModelConfig, TrainingConfig
from monthly_diffusion.data.sources import open_canonical_state, open_co2
from monthly_diffusion.data.dataloader import (
    PhysicalStates,
    LatentSequences,
    co2_values,
)
from monthly_diffusion.training.data import validate_training_split
from monthly_diffusion.training.resume import restore_training_state
from monthly_diffusion.training.latents import cache_latents
from monthly_diffusion.training.autoencoder import train_autoencoder
from monthly_diffusion.training.coupled_training import train_coupled_diffusion


parser = argparse.ArgumentParser()
parser.add_argument("data_yaml", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--config",
    type=Path,
    default=None,
    help="Optional YAML parameter overrides; defaults are written below",
)
parser.add_argument("--device", default=None)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument(
    "--load-in-memory",
    action="store_true",
    help="Keep only train/validation physical fields in CPU RAM (about 3.6 GiB at 1 degree)",
)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
torch.set_num_threads(args.threads)
if args.device == "mps" and not torch.backends.mps.is_available():
    raise RuntimeError(
        "MPS was requested but is unavailable; refusing a silent CPU run"
    )
logging.info("Opening pinned source from %s", args.data_yaml)
source = yaml.safe_load(args.data_yaml.read_text())
atmosphere, ocean, statics = open_canonical_state(
    source["state_zarr"], expected_attrs=source.get("expected_attrs")
)
co2 = open_co2(source["co2_netcdf"], expected_sha256=source.get("co2_sha256"))
args.output.mkdir(parents=True, exist_ok=True)
# Persist the source record beside model/config/history, not in the package.
(args.output / "data_sources.yaml").write_text(yaml.safe_dump(source, sort_keys=False))
# These are ordinary parameter values. Edit them here for a new experiment.
# --config still accepts the existing YAML files when reproducing an old run.
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
model = MonthlyDiffusionV1(statics=statics, config=settings, device=args.device)
train = slice("1966-01", "2006-12")
validation = slice("1958-01", "1965-12")
logging.info(
    "Device %s | training 1966-2006 (%d months) | validation 1958-1965 (%d months)",
    model.device,
    atmosphere.sel(time=train).sizes["time"],
    atmosphere.sel(time=validation).sizes["time"],
)
logging.info(
    "Preparing geometry and training-only normalization; output %s", args.output
)
train_atmos, train_ocean = atmosphere.sel(time=train), ocean.sel(time=train)
val_atmos, val_ocean = atmosphere.sel(time=validation), ocean.sel(time=validation)
if args.load_in_memory:
    logging.info(
        "Loading training and validation fields into CPU RAM; test/future states remain lazy"
    )
    for dataset in (train_atmos, train_ocean, val_atmos, val_ocean):
        dataset.load()
# Test 2007--2014 is deliberately not supplied to either trainer.
checkpoint = args.output / "model.pt"
if checkpoint.exists():
    if not args.resume:
        raise FileExistsError("Run already exists; use --resume or another directory")
    restore_training_state(model, checkpoint)

# Fit physical normalization on training data only. Resume uses the saved fit.
if model.autoencoder is None:
    model.prepare(train_atmos, train_ocean, co2=co2)
    model.save(checkpoint)
validate_training_split(model, train_atmos, train_ocean, (val_atmos, val_ocean), co2)
if not model.initialization_moments:
    model.fit_stochastic_initialization(train_atmos, train_ocean)
    model.save(checkpoint)
model.config.save(args.output / "config.yaml")

training_data = PhysicalStates(
    train_atmos, train_ocean, model.preprocessor, co2_values(co2, train_atmos.time)
)
validation_data = PhysicalStates(
    val_atmos, val_ocean, model.preprocessor, co2_values(co2, val_atmos.time)
)
static_fields = (
    model.preprocessor.static_tensor(model.statics).to(model.device).unsqueeze(0)
)

# 1. Train reconstruction only; the trainer selects the best validation EMA.
if "autoencoder" not in model.stages:
    model.autoencoder.requires_grad_(True)
    weights = {
        component: torch.tensor(
            [
                variable.weight
                for variable in model.config.variables
                if variable.component == component
            ]
        )
        for component in model.names
    }
    selected = train_autoencoder(
        model.autoencoder,
        training_data,
        validation_data,
        statics=static_fields,
        latitudes=torch.tensor(model.statics.lat.values),
        weights=weights,
        config=model.config.autoencoder_training,
        directory=args.output / "autoencoder",
        resume=args.resume,
    )
    model.stages["autoencoder"] = {
        key: value for key, value in selected.items() if key != "history"
    }
    model.save(checkpoint)

# 2. Freeze the selected AE, encode once, and fit fixed train-only latent scaling.
model.autoencoder.eval().requires_grad_(False)
cache = cache_latents(model, training_data, validation_data, args.output)
model.save(checkpoint)

# 3. Train monthly transitions on cached latent sequences. The AE stays frozen.
if "diffusion" not in model.stages:
    training_sequences = LatentSequences(
        cache["training"]["atmosphere"],
        cache["training"]["ocean"],
        cache["training_time"],
        model.config.diffusion_training.rollout_steps,
    )
    validation_sequences = LatentSequences(
        cache["validation"]["atmosphere"],
        cache["validation"]["ocean"],
        cache["validation_time"],
        model.config.diffusion_training.rollout_steps,
    )
    selected = train_coupled_diffusion(
        model.transition,
        training_sequences,
        validation_sequences,
        config=model.config.diffusion_training,
        directory=args.output / "diffusion",
        resume=args.resume,
    )
    model.stages["diffusion"] = {
        key: value for key, value in selected.items() if key != "history"
    }
    model.save(checkpoint)
