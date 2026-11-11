"""Explicit E3SM/ERA5 example: same split, staged architecture, no hidden downloads.

After `pip install -e .`, run:
    python scripts/legacy/train_model.py examples/e3sm.yaml --output runs/e3sm
This is intentionally a readable script, not a framework CLI.
"""

from dataclasses import asdict, replace
import csv
import random
import time
import numpy as np
from torch.utils.data import DataLoader
from monthly_diffusion.training.ema import EMA
from monthly_diffusion.checkpoints import atomic_save
from monthly_diffusion.random_state import rng_state, restore_rng, fixed_rng
from monthly_diffusion.modules.autoencoder import reconstruction_mse

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

# Historical paired selection is kept here for the existing experiment recipes.
# New experiments use the two component scripts at the top of scripts/.
log = logging.getLogger(__name__)

for stage in ("autoencoder", "diffusion"):
    if stage == "autoencoder":
        network = model.autoencoder.requires_grad_(True)
        training = replace(
            model.config.autoencoder_training,
            validation_seeds=(model.config.autoencoder_training.validation_seeds[0],),
        )
        weights = {
            c: torch.tensor(
                [v.weight for v in model.config.variables if v.component == c]
            )
            for c in model.names
        }
        latitudes = torch.tensor(model.statics.lat.values)
    else:
        model.autoencoder.eval().requires_grad_(False)
        cache = cache_latents(model, training_data, validation_data, args.output)
        model.save(checkpoint)
        training = model.config.diffusion_training
        training_data, validation_data = [
            LatentSequences(
                cache[split]["atmosphere"],
                cache[split]["ocean"],
                cache[split + "_time"],
                training.rollout_steps,
            )
            for split in ("training", "validation")
        ]
        network = model.transition
    if stage in model.stages:
        continue
    training_directory = args.output / stage
    resume = args.resume
    training_directory = Path(training_directory)
    training_directory.mkdir(parents=True, exist_ok=True)
    device = next(network.parameters()).device
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
        eps=training.optimizer_eps,
    )
    ema = EMA(network, training.ema_decay)
    history, step, first_epoch, stale = ([], 0, 1, 0)
    best, patience_best = (float("inf"), float("inf"))
    latest_path = training_directory / "latest.pt"
    if latest_path.exists() and (not resume):
        raise FileExistsError(f"Use resume=True or a new run directory: {latest_path}")
    if resume and latest_path.exists():
        saved = torch.load(latest_path, map_location=device, weights_only=False)
        prior = saved["training_config"]
        for name in (
            "batch_size",
            "learning_rate",
            "weight_decay",
            "optimizer_eps",
            "ema_decay",
            "seed",
            "rollout_steps",
            "loss_weight",
        ):
            if prior[name] != asdict(training)[name]:
                raise ValueError(
                    f"Resume changes training setting {name}; use a new run"
                )
        network.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        ema.state = saved["ema"]
        history, step = (saved["history"], saved["step"])
        if history and "val_transition" in history[-1]:
            best = min((float(row["val_transition"]) for row in history))
            patience_best = best
            stale = 0
        else:
            stale = saved["stale"]
            best, patience_best = (saved["best"], saved["patience_best"])
        first_epoch = saved["epoch"] + 1
        restore_rng(saved["rng"])
    else:
        torch.manual_seed(training.seed)
        np.random.seed(training.seed)
        random.seed(training.seed)
    validation_loader = DataLoader(
        validation_data, batch_size=training.batch_size, shuffle=False
    )
    for epoch in range(first_epoch, training.max_epochs + 1):
        started = time.monotonic()
        network.train()
        generator = torch.Generator().manual_seed(training.seed + epoch)
        loader = DataLoader(
            training_data,
            batch_size=training.batch_size,
            shuffle=True,
            generator=generator,
        )
        totals, count = ({}, 0)
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            if stage == "autoencoder":
                terms, loss = ({}, 0.0)
                static_batch = static_fields.expand(batch["month"].shape[0], -1, -1, -1)
                for component in ("atmosphere", "ocean"):
                    component_model = getattr(network, component)
                    node_mask = (
                        torch.isfinite(batch[component][:, 0])
                        if getattr(component_model, "mask_aware", False)
                        else None
                    )
                    prediction, _ = component_model(
                        batch[component],
                        static_batch,
                        batch["month"],
                        batch["co2"],
                        node_mask=node_mask,
                    )
                    per_field = reconstruction_mse(
                        prediction, batch[component], latitudes
                    )
                    terms[component + "_reconstruction"] = per_field.mean()
                    for i, value in enumerate(per_field):
                        terms[f"{component}_field{i}"] = value
                    loss = loss + (per_field * weights[component].to(per_field)).sum()
                terms["loss"] = loss
            else:
                terms = network.rollout_loss(batch, loss_weight=training.loss_weight)
            if not torch.isfinite(terms["loss"]):
                raise FloatingPointError("Nonfinite training loss")
            terms["loss"].backward()
            optimizer.step()
            step += 1
            ema.update(network, step)
            n = len(next(iter(batch.values())))
            count += n
            for name, value in terms.items():
                totals[name] = totals.get(name, 0.0) + n * float(value.detach())
        row = {
            "epoch": epoch,
            "optimizer_steps": step,
            **{f"train_{k}": v / count for k, v in totals.items()},
        }
        totals, count = ({}, 0)
        network.eval()
        with ema.applied(network), torch.no_grad():
            for seed in training.validation_seeds:
                with fixed_rng(seed, device):
                    for batch in validation_loader:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        if stage == "autoencoder":
                            terms, loss = ({}, 0.0)
                            static_batch = static_fields.expand(
                                batch["month"].shape[0], -1, -1, -1
                            )
                            for component in ("atmosphere", "ocean"):
                                component_model = getattr(network, component)
                                node_mask = (
                                    torch.isfinite(batch[component][:, 0])
                                    if getattr(component_model, "mask_aware", False)
                                    else None
                                )
                                prediction, _ = component_model(
                                    batch[component],
                                    static_batch,
                                    batch["month"],
                                    batch["co2"],
                                    node_mask=node_mask,
                                )
                                per_field = reconstruction_mse(
                                    prediction, batch[component], latitudes
                                )
                                terms[component + "_reconstruction"] = per_field.mean()
                                for i, value in enumerate(per_field):
                                    terms[f"{component}_field{i}"] = value
                                loss = (
                                    loss
                                    + (
                                        per_field * weights[component].to(per_field)
                                    ).sum()
                                )
                            terms["loss"] = loss
                        else:
                            terms = network.rollout_loss(
                                batch, loss_weight=training.loss_weight
                            )
                        n = len(next(iter(batch.values())))
                        count += n
                        for name, value in terms.items():
                            totals[name] = totals.get(name, 0.0) + n * float(value)
        row.update({f"val_{k}": v / count for k, v in totals.items()})
        row["seconds"] = time.monotonic() - started
        score = row.get("val_transition", row["val_loss"])
        if not np.isfinite(score):
            raise FloatingPointError("Nonfinite validation loss")
        improved = score < best
        if improved:
            best = score
            atomic_save(
                dict(model=ema.state, epoch=epoch, score=score),
                training_directory / "best.pt",
            )
        if score < patience_best - training.min_delta:
            stale, patience_best = (0, score)
        else:
            stale += 1
        row["stale_epochs"] = stale
        history.append(row)
        atomic_save(
            dict(
                model=network.state_dict(),
                optimizer=optimizer.state_dict(),
                ema=ema.state,
                history=history,
                epoch=epoch,
                step=step,
                best=best,
                patience_best=patience_best,
                stale=stale,
                rng=rng_state(device),
                training_config=asdict(training),
            ),
            latest_path,
        )
        with (training_directory / "history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        log.info(
            "%s epoch %d | train %.6f val %.6f | %.1fs | stale %d/%d",
            training_directory.name,
            epoch,
            row["train_loss"],
            score,
            row["seconds"],
            stale,
            training.patience,
        )
        if stale >= training.patience:
            break
    chosen = torch.load(
        training_directory / "best.pt", map_location=device, weights_only=False
    )
    network.load_state_dict(chosen["model"])
    network.eval()
    selected = dict(epoch=chosen["epoch"], score=chosen["score"], history=history)
    model.stages[stage] = {
        key: value for key, value in selected.items() if key != "history"
    }
    model.save(checkpoint)
logging.info("Saved selected paired model to %s", checkpoint)
