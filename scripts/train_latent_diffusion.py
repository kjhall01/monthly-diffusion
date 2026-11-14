"""Train coupled diffusion on the selected AE latents. Edit the values below."""

import sys
from pathlib import Path
from time import perf_counter
import random
import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from monthly_diffusion.diffusion import CoupledLatentDiffusion
from monthly_diffusion.rollout import fixed_rng, co2_values
from workflows.run_files import check_script_arguments
from workflows.data.run_data import load_training_latents, load_run_co2
from workflows.data.dataloader import LatentSequences
from workflows.training.ema import EMA
from workflows.training.time_limit import TrainingTimeLimit, check_time_limit
from workflows.training.losses import rollout_loss, DiffusionLoss
from workflows.training.files import (
    save_diffusion_setup,
    load_training_progress,
    save_best_weights,
    save_training_progress,
    add_diffusion_history,
    select_best_weights,
)

check_script_arguments()

# Resolve beside this script, regardless of the shell's working directory.
RUN_DIR = Path(__file__).resolve().parent / "runs/test_gridpoint_co2_cfg"
TIME_LIMIT_HOURS = 4.0
if not 0 < TIME_LIMIT_HOURS < float("inf"):
    raise ValueError("TIME_LIMIT_HOURS must be positive and finite")
# Count setup and training; allow the final model/cache export to finish.
deadline = perf_counter() + TIME_LIMIT_HOURS * 3600
RESUME = False
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
THREADS = 4

SEED = 20260716
ROLLOUT_STEPS = 3

# Load only tensors from the two selected AEs. No AE networks are constructed.
train_atmosphere, train_ocean, val_atmosphere, val_ocean, train_dates, val_dates = (
    load_training_latents(RUN_DIR)
)
forcing = load_run_co2(RUN_DIR)
training_co2 = co2_values(
    forcing, xr.DataArray(train_dates, dims="time", coords={"time": train_dates})
).values
validation_co2 = co2_values(
    forcing, xr.DataArray(val_dates, dims="time", coords={"time": val_dates})
).values
training_data = LatentSequences(
    train_atmosphere, train_ocean, train_dates, ROLLOUT_STEPS, co2=training_co2
)
validation_data = LatentSequences(
    val_atmosphere, val_ocean, val_dates, ROLLOUT_STEPS, co2=validation_co2
)
torch.set_num_threads(THREADS)
device = torch.device(DEVICE)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Build the transition. Edit these values to change the architecture.
WIDTH = 132
DEPTH = 4
HEADS = 4
FFN_RATIO = 1.55
TIME_EMBEDDING_CHANNELS = 64
ATMOSPHERE_TAU = 0.5
OCEAN_TAU = 6.0
COUPLING = "jacobi"
CROSS_GRADIENTS = False
CONDITION_ON_CO2 = True
CO2_DROPOUT = 0.10  # drop CO2 for 10% of training sequences, retaining month/history
# Fit only on the training months. The model saves these values and accepts ppm.
CO2_MEAN = float(np.mean(training_co2))
CO2_STD = max(float(np.std(training_co2)), 1e-6)

diffusion = CoupledLatentDiffusion(
    atmosphere_latent_channels=train_atmosphere.shape[-1],
    ocean_latent_channels=train_ocean.shape[-1],
    n_latent_tokens=train_atmosphere.shape[1],
    width=WIDTH,
    depth=DEPTH,
    n_heads=HEADS,
    ffn_ratio=FFN_RATIO,
    time_embedding_channels=TIME_EMBEDDING_CHANNELS,
    atmosphere_tau=ATMOSPHERE_TAU,
    ocean_tau=OCEAN_TAU,
    coupling=COUPLING,
    cross_gradients=CROSS_GRADIENTS,
    condition_on_co2=CONDITION_ON_CO2,
    co2_dropout=CO2_DROPOUT if CONDITION_ON_CO2 else 0.0,
    co2_mean=CO2_MEAN,
    co2_std=CO2_STD,
).to(device)
print(f"Diffusion: {sum(p.numel() for p in diffusion.parameters()):,} parameters")
directory = RUN_DIR / "diffusion"
directory.mkdir(parents=True, exist_ok=True)
save_diffusion_setup(
    directory, diffusion, training_co2=training_co2, validation_co2=validation_co2
)
print(
    f"CO2 FiLM: {CONDITION_ON_CO2}; dropout {diffusion.co2_dropout:.0%}; normalization {CO2_MEAN:.3f} +/- {CO2_STD:.3f} ppm"
)
if (directory / "model.pt").exists():
    if not RESUME:
        raise FileExistsError("Selected diffusion exists; use a new run or RESUME=True")
    print("Keeping the selected diffusion model.")
    raise SystemExit(0)

# Training controls are used directly by the optimizer and loop.
BATCH_SIZE = 4
EPOCHS = 1000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
OPTIMIZER_EPS = 1e-6
EMA_DECAY = 0.995
LOSS_WEIGHT = 1.0
PATIENCE = 5
MIN_DELTA = 1e-4
VALIDATION_SEEDS = (271828, 314159)


if min(BATCH_SIZE, EPOCHS, PATIENCE, THREADS) < 1 or not VALIDATION_SEEDS:
    raise ValueError("Require positive training counts and validation seeds")
validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False)

optimizer = torch.optim.AdamW(
    diffusion.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    eps=OPTIMIZER_EPS,
)
ema = EMA(diffusion, EMA_DECAY)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
progress = load_training_progress(
    directory,
    diffusion,
    optimizer,
    ema,
    resume=RESUME,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    patience=PATIENCE,
    min_delta=MIN_DELTA,
    seed=SEED,
    validation_seeds=VALIDATION_SEEDS,
    rollout_steps=ROLLOUT_STEPS,
    loss_weight=LOSS_WEIGHT,
)

try:
    for epoch in range(progress.epoch + 1, EPOCHS + 1):
        check_time_limit(deadline)
        if progress.stale_epochs >= PATIENCE:
            break
        started = perf_counter()
        diffusion.train()
        generator = torch.Generator().manual_seed(SEED + epoch)
        training_loader = DataLoader(
            training_data, batch_size=BATCH_SIZE, shuffle=True, generator=generator
        )
        atmosphere_sum = torch.zeros(ROLLOUT_STEPS, device=device)
        ocean_sum = torch.zeros(ROLLOUT_STEPS, device=device)
        count = 0
        for atmosphere, ocean, months, co2 in training_loader:
            check_time_limit(deadline)
            atmosphere = atmosphere.to(device)
            ocean = ocean.to(device)
            months = months.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = rollout_loss(
                diffusion,
                atmosphere,
                ocean,
                months,
                co2=co2.to(device),
                loss_weight=LOSS_WEIGHT,
            )
            if not torch.isfinite(losses.total):
                raise FloatingPointError("Nonfinite diffusion loss")
            losses.total.backward()
            optimizer.step()
            progress.optimizer_steps += 1
            ema.update(diffusion, progress.optimizer_steps)
            atmosphere_sum += len(months) * losses.atmosphere.detach()
            ocean_sum += len(months) * losses.ocean.detach()
            count += len(months)
        training_losses = DiffusionLoss(
            atmosphere_sum / count, ocean_sum / count, LOSS_WEIGHT
        )

        diffusion.eval()
        validation_sum = 0.0
        atmosphere_sum = torch.zeros(ROLLOUT_STEPS, device=device)
        ocean_sum = torch.zeros(ROLLOUT_STEPS, device=device)
        count = 0
        with ema.applied(diffusion), torch.no_grad():
            for seed in VALIDATION_SEEDS:
                # Replay the same validation draws without changing the training RNG.
                with fixed_rng(seed, device):
                    for atmosphere, ocean, months, co2 in validation_loader:
                        check_time_limit(deadline)
                        losses = rollout_loss(
                            diffusion,
                            atmosphere.to(device),
                            ocean.to(device),
                            months.to(device),
                            co2=co2.to(device),
                            loss_weight=LOSS_WEIGHT,
                        )
                        validation_sum += len(months) * float(losses.transition)
                        atmosphere_sum += len(months) * losses.atmosphere
                        ocean_sum += len(months) * losses.ocean
                        count += len(months)
        validation_losses = DiffusionLoss(
            atmosphere_sum / count, ocean_sum / count, LOSS_WEIGHT
        )
        score = validation_sum / count  # average the unscaled batch losses
        if not np.isfinite(score):
            raise FloatingPointError("Nonfinite validation loss")
        if score < progress.best_validation:
            progress.best_validation = score
            save_best_weights(directory, ema, epoch, score)
        if score < progress.patience_best - MIN_DELTA:
            progress.patience_best = score
            progress.stale_epochs = 0
        else:
            progress.stale_epochs += 1
        progress.epoch = epoch
        epoch_seconds = perf_counter() - started  # training and validation
        add_diffusion_history(
            progress, training_losses, validation_losses, epoch_seconds
        )
        save_training_progress(directory, diffusion, optimizer, ema, progress)
        print(
            f"Epoch {epoch:04d} | train {float(training_losses.transition):.6f}"
            f" | validation {score:.6f} | {epoch_seconds:.1f} s/epoch"
            f" | stale {progress.stale_epochs}/{PATIENCE}"
        )

except TrainingTimeLimit as error:
    if not (directory / "best.pt").exists():
        raise RuntimeError(
            "Time limit reached before any epoch finished validation; "
            "increase TIME_LIMIT_HOURS before restarting."
        ) from error
    print("Training time limit reached; exporting the best completed validation epoch.")
    # An unfinished epoch is discarded. latest.pt remains the last complete
    # epoch, and best.pt supplies the weights used by every following stage.

selected_epoch = select_best_weights(directory, diffusion)
print(f"Selected epoch {selected_epoch}; saved diffusion in {directory}")
