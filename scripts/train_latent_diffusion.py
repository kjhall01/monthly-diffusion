"""Train coupled diffusion on the selected AE caches. Edit these values and run."""

from pathlib import Path
from monthly_diffusion.run_files import check_script_arguments
from time import perf_counter
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from monthly_diffusion.components.coupled_latent_diffusion import CoupledLatentDiffusion
from monthly_diffusion.data.dataloader import LatentSequences
from monthly_diffusion.run_files import (
    read_run,
    load_latents,
    file_digest,
    save_selection,
    save_history,
)
from monthly_diffusion.checkpoints import atomic_save
from monthly_diffusion.training.ema import EMA
from monthly_diffusion.training.checks import check_resume_settings
from monthly_diffusion.random_state import rng_state, restore_rng, fixed_rng

check_script_arguments()

RUN_DIR = Path("runs/production")
RESUME = False
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
THREADS = 4

WIDTH = 132
DEPTH = 4
HEADS = 4
FFN_RATIO = 1.55
TIME_EMBEDDING_CHANNELS = 64
ATMOSPHERE_TAU = 0.5
OCEAN_TAU = 6.0
COUPLING = "jacobi"
CROSS_GRADIENTS = False

BATCH_SIZE = 4
EPOCHS = 1000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
OPTIMIZER_EPS = 1e-6
EMA_DECAY = 0.995
ROLLOUT_STEPS = 3
LOSS_WEIGHT = 0.01
PATIENCE = 5
MIN_DELTA = 1e-4
SEED = 20260716
VALIDATION_SEEDS = (271828, 314159)

# The AEs are not instantiated for training diffusion: all needed states are cached.
read_run(RUN_DIR)
atmosphere = load_latents(RUN_DIR, "atmosphere")
ocean = load_latents(RUN_DIR, "ocean")
for split in ("training", "validation"):
    if not np.array_equal(atmosphere[split + "_time"], ocean[split + "_time"]):
        raise ValueError("Atmosphere/ocean latent dates differ")
    if atmosphere[split].shape[1] != ocean[split].shape[1]:
        raise ValueError("Coupled diffusion requires the same latent mesh")
# Equal node counts alone are insufficient: rotation and latent refinement must match.
ae_parameters = [
    torch.load(RUN_DIR / c / "model.pt", map_location="cpu", weights_only=False)[
        "parameters"
    ]
    for c in ("atmosphere", "ocean")
]
if (ae_parameters[0]["mesh_levels"][-1], tuple(ae_parameters[0]["rotation"])) != (
    ae_parameters[1]["mesh_levels"][-1],
    tuple(ae_parameters[1]["rotation"]),
):
    raise ValueError("Both AE caches must use the same latent mesh coordinates")
training_data = LatentSequences(
    atmosphere["training"],
    ocean["training"],
    atmosphere["training_time"],
    ROLLOUT_STEPS,
)
validation_data = LatentSequences(
    atmosphere["validation"],
    ocean["validation"],
    atmosphere["validation_time"],
    ROLLOUT_STEPS,
)
validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False)

parameters = dict(
    atmosphere_latent_channels=atmosphere["training"].shape[-1],
    ocean_latent_channels=ocean["training"].shape[-1],
    n_latent_tokens=atmosphere["training"].shape[1],
    width=WIDTH,
    depth=DEPTH,
    n_heads=HEADS,
    ffn_ratio=FFN_RATIO,
    time_embedding_channels=TIME_EMBEDDING_CHANNELS,
    atmosphere_tau=ATMOSPHERE_TAU,
    ocean_tau=OCEAN_TAU,
    coupling=COUPLING,
    cross_gradients=CROSS_GRADIENTS,
)
settings = dict(
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    optimizer_eps=OPTIMIZER_EPS,
    ema_decay=EMA_DECAY,
    rollout_steps=ROLLOUT_STEPS,
    loss_weight=LOSS_WEIGHT,
    patience=PATIENCE,
    min_delta=MIN_DELTA,
    seed=SEED,
    validation_seeds=VALIDATION_SEEDS,
)
if min(BATCH_SIZE, EPOCHS, PATIENCE, THREADS) < 1 or not VALIDATION_SEEDS:
    raise ValueError(
        "Require positive training counts and at least one validation seed"
    )
torch.set_num_threads(THREADS)
device = torch.device(DEVICE)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
diffusion = CoupledLatentDiffusion(**parameters).to(device)
print(f"Diffusion: {sum(p.numel() for p in diffusion.parameters()):,} parameters")

# Bind this transition to the exact selected AEs, preprocessing and cache contents.
directory = RUN_DIR / "diffusion"
directory.mkdir(parents=True, exist_ok=True)
setup_path, model_path = directory / "setup.pt", directory / "model.pt"
setup = dict(
    parameters=parameters,
    ae_digests={
        c: {
            name: file_digest(RUN_DIR / c / name)
            for name in ("model.pt", "setup.pt", "latents.pt")
        }
        for c in ("atmosphere", "ocean")
    },
)
if setup_path.exists():
    if torch.load(setup_path, map_location="cpu", weights_only=False) != setup:
        raise ValueError("Diffusion parameters or AE caches changed; choose a new run")
else:
    atomic_save(setup, setup_path)
setup_digest = file_digest(setup_path)
if model_path.exists():
    if not RESUME:
        raise FileExistsError(
            "Selected diffusion exists; set RESUME=True or choose a new RUN_DIR"
        )
    print("Keeping the selected diffusion model.")
    raise SystemExit(0)

optimizer = torch.optim.AdamW(
    diffusion.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    eps=OPTIMIZER_EPS,
)
ema = EMA(diffusion, EMA_DECAY)
# Separate weight initialization from the training RNG, as in the original trainer.
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
history, step, first_epoch, stale = [], 0, 1, 0
best, patience_best = float("inf"), float("inf")
latest_path = directory / "latest.pt"
if latest_path.exists():
    if not RESUME:
        raise FileExistsError("Interrupted diffusion training exists; set RESUME=True")
    saved = torch.load(latest_path, map_location=device, weights_only=False)
    check_resume_settings(saved["settings"], settings)
    if saved["setup_digest"] != setup_digest:
        raise ValueError("AE inputs changed since the interrupted training")
    diffusion.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    ema.state = saved["ema"]
    history, step, first_epoch, stale = (
        saved["history"],
        saved["step"],
        saved["epoch"] + 1,
        saved["stale"],
    )
    best, patience_best = saved["best"], saved["patience_best"]
    restore_rng(saved["rng"])

for epoch in range(first_epoch, EPOCHS + 1):
    if stale >= PATIENCE:
        break
    started = perf_counter()
    diffusion.train()
    generator = torch.Generator().manual_seed(SEED + epoch)
    training_loader = DataLoader(
        training_data, batch_size=BATCH_SIZE, shuffle=True, generator=generator
    )
    totals, count = {}, 0
    for batch in training_loader:
        batch = {name: value.to(device) for name, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        # The network implements the coupled multi-step diffusion objective.
        # Its own generated history carries gradients; cross-component paths obey
        # CROSS_GRADIENTS. No AE is trained and no physical data enter this loop.
        terms = diffusion.rollout_loss(batch, loss_weight=LOSS_WEIGHT)
        if not torch.isfinite(terms["loss"]):
            raise FloatingPointError("Nonfinite diffusion loss")
        terms["loss"].backward()
        optimizer.step()
        step += 1
        ema.update(diffusion, step)
        n = len(batch["month"])
        count += n
        for name, value in terms.items():
            totals[name] = totals.get(name, 0.0) + n * float(value.detach())
    row = dict(
        epoch=epoch,
        optimizer_steps=step,
        **{f"train_{k}": v / count for k, v in totals.items()},
    )

    diffusion.eval()
    totals, count = {}, 0
    with ema.applied(diffusion), torch.no_grad():
        for seed in VALIDATION_SEEDS:
            # Replayed validation noise does not consume training's RNG stream.
            with fixed_rng(seed, device):
                for batch in validation_loader:
                    batch = {name: value.to(device) for name, value in batch.items()}
                    terms = diffusion.rollout_loss(batch, loss_weight=LOSS_WEIGHT)
                    n = len(batch["month"])
                    count += n
                    for name, value in terms.items():
                        totals[name] = totals.get(name, 0.0) + n * float(value)
    row.update({f"val_{k}": v / count for k, v in totals.items()})
    score = row["val_transition"]  # Selection uses the unscaled scientific objective.
    if not np.isfinite(score):
        raise FloatingPointError("Nonfinite validation loss")
    if score < best:
        best = score
        atomic_save(
            dict(model=ema.state, epoch=epoch, score=score), directory / "best.pt"
        )
    if score < patience_best - MIN_DELTA:
        patience_best, stale = score, 0
    else:
        stale += 1
    row.update(seconds=perf_counter() - started, stale_epochs=stale)
    history.append(row)
    atomic_save(
        dict(
            model=diffusion.state_dict(),
            optimizer=optimizer.state_dict(),
            ema=ema.state,
            epoch=epoch,
            step=step,
            history=history,
            best=best,
            patience_best=patience_best,
            stale=stale,
            settings=settings,
            setup_digest=setup_digest,
            rng=rng_state(device),
        ),
        latest_path,
    )
    save_history(history, directory / "history.csv")
    print(
        f"Epoch {epoch:04d} | train {row['train_transition']:.6f} | validation {score:.6f} | stale {stale}/{PATIENCE}"
    )

selected = save_selection(
    directory,
    parameters=parameters,
    setup_digest=setup_digest,
    training_settings=settings,
)
print(f"Selected epoch {selected['epoch']}; saved diffusion in {directory}")
