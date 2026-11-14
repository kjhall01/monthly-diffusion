"""Compare old and residual-only FiLM on synthetic atmosphere batches.

Edit the values below and run this file on the training machine. No data files,
checkpoints, optimizer, or physical metrics are involved: this isolates model
forward/backward time. Each architecture starts from the same random weights.
"""

import sys
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from monthly_diffusion.autoencoder import Autoencoder
from monthly_diffusion.mesh import build_mesh_hierarchy
from workflows.devices import synchronize

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
THREADS = 4
BATCH_SIZE = 4
GRID_SHAPE = (180, 360)
CHANNELS = 16
WIDTH = 128
LATENT_CHANNELS = 32
MESH_LEVELS = (4, 3, 2)
REFINERS = (2, 4, 8)
WARMUP = 2
REPEATS = 10

torch.set_num_threads(THREADS)
device = torch.device(DEVICE)
torch.manual_seed(31)
geometry = build_mesh_hierarchy(
    nlat=GRID_SHAPE[0], nlon=GRID_SHAPE[1], mesh_levels=MESH_LEVELS
)
statics = torch.randn(2, *GRID_SHAPE)
state = torch.randn(BATCH_SIZE, CHANNELS, *GRID_SHAPE, device=device)
month = torch.randn(BATCH_SIZE, 2, device=device)
co2 = torch.randn(BATCH_SIZE, 1, device=device)
print(f"Torch {torch.__version__}; {device}; {THREADS} CPU threads; batch {BATCH_SIZE}")

medians = []
for residual_only in (False, True):
    torch.manual_seed(41)
    model = Autoencoder(
        CHANNELS,
        geometry,
        GRID_SHAPE,
        statics=statics,
        width=WIDTH,
        latent_channels=LATENT_CHANNELS,
        refiners=REFINERS,
        residual_film_only=residual_only,
    ).to(device)
    times = []
    for repeat in range(WARMUP + REPEATS):
        synchronize(device)
        start = perf_counter()
        model.zero_grad(set_to_none=True)
        prediction, latent = model(state, month, co2)
        loss = (prediction - state).square().mean()
        loss.backward()
        synchronize(device)
        elapsed = perf_counter() - start
        if repeat >= WARMUP:
            times.append(elapsed)
        del prediction, latent, loss
    medians.append(median(times))
    label = "residual-only FiLM" if residual_only else "old FiLM path"
    print(
        f"{label}: median {median(times):.4f} s; range {min(times):.4f}–{max(times):.4f} s",
        flush=True,
    )
    del model

print(f"Time reduction: {100 * (1 - medians[1] / medians[0]):.1f}%")
