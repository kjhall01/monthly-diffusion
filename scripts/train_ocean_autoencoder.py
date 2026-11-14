"""Train the ocean autoencoder. Edit the values below and run this file."""

import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from monthly_diffusion.autoencoder import Autoencoder
from monthly_diffusion.mesh import (
    make_equiangular_grid_from_xarray,
    build_mesh_hierarchy,
)
from monthly_diffusion.rollout import fixed_rng
from workflows.run_files import check_script_arguments, run_inputs
from monthly_diffusion.checkpoints import load_latents
from workflows.data.sources import open_state_dataset, static_fields, open_co2
from monthly_diffusion.preprocessing import select_variables
from monthly_diffusion.preprocessing import (
    Preprocessor,
    validate_fields,
    validate_units,
)
from workflows.data.dataloader import ComponentStates
from monthly_diffusion.rollout import co2_values
from workflows.training.ema import EMA
from workflows.training.time_limit import TrainingTimeLimit, check_time_limit
from workflows.training.logging import training_logger
from workflows.training.losses import reconstruction_mse
from workflows.training.files import (
    load_preprocessor,
    save_autoencoder_setup,
    load_training_progress,
    save_best_weights,
    save_training_progress,
    add_autoencoder_history,
    select_best_weights,
    save_latent_cache,
)

check_script_arguments()

# Resolve beside this script, regardless of the shell's working directory.
RUN_DIR = Path(__file__).resolve().parent / "runs/test_gridpoint_log1p_mtpr_masked"
TIME_LIMIT_HOURS = 4.0
if not 0 < TIME_LIMIT_HOURS < float("inf"):
    raise ValueError("TIME_LIMIT_HOURS must be positive and finite")
# Count setup and training; allow the final model/cache export to finish.
deadline = perf_counter() + TIME_LIMIT_HOURS * 3600
COMPONENT = "ocean"

# Select physical fields here, in their input/output channel order.
VARIABLES = [
    "sst",
    "ssh",
    "sss",
    "mld",
    "sic",
    "sivol",
    "d20",
    "ohc300",
    "ohc700_minus_ohc300",
]
# Normalize physical values directly: no logit or inverse-softplus transforms.
SPECIFICATIONS = [
    replace(variable, transform="identity", scale=1.0)
    for variable in select_variables(VARIABLES, COMPONENT)
]
RESUME = False
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
THREADS = 4
LOAD_IN_MEMORY = True

SEED = 20260716

# Reuse the recorded paths/splits; these values can also be written explicitly.
DATA, CO2, TRAIN_PERIOD, VALIDATION_PERIOD = run_inputs(RUN_DIR)

# Open the Dataset here; select the variables and periods below.
dataset = open_state_dataset(DATA)  # or xr.open_zarr(DATA) for already-canonical fields
fields = dataset[[variable.name for variable in SPECIFICATIONS]]
statics = static_fields(dataset)
co2 = open_co2(CO2)
validate_units(fields, SPECIFICATIONS)
training_fields = fields.sel(time=TRAIN_PERIOD)
validation_fields = fields.sel(time=VALIDATION_PERIOD)
for split in (training_fields, validation_fields):
    validate_fields(split, [variable.name for variable in SPECIFICATIONS], statics)
if LOAD_IN_MEMORY:
    training_fields.load()
    validation_fields.load()
directory = RUN_DIR / COMPONENT
directory.mkdir(parents=True, exist_ok=True)
logger = training_logger(directory)
model_path = directory / "model.pt"
if model_path.exists() and not RESUME:
    raise FileExistsError("Selected AE exists; choose a new RUN_DIR or set RESUME=True")
torch.set_num_threads(THREADS)
device = torch.device(DEVICE)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Match the test_gridpoint atmosphere autoencoder's architecture.
WIDTH = 64
LATENT_CHANNELS = 32
REFINER_MLP_RATIO = 2  # channel mixing after every spatial refinement
MESH_LEVELS = (4, 3, 2)
REFINERS = (4, 6, 6)
ROTATION = (17, 23, 11)
CONDITION_ON_CO2 = True  # actual CO2 in both encoder and decoder, without jitter
CO2_FILM_LEVELS = ()  # used only when CONDITION_ON_CO2=True; empty means all levels
MASK_AWARE = True

# Fit normalization once, using training data only. Reuse it when resuming.
# "gridpoint": local time mean/std. "annual_zonal": zonal mean + pooled std.
CENTERING = "gridpoint"
# Minimum local std as a fraction of the variable's typical temporal std.
GRIDPOINT_STD_FLOOR = 0.05
# Exclude cells constant over the whole training period, before the std floor.
# Constant SIC=1 remains valid. Seasonal zeros remain valid wherever values vary.
MASK_CONSTANT_CELLS = True
OCEAN_SUPPORT = (
    "per_variable"  # "sst" applies one training-SST support to all ocean fields
)
if (directory / "setup.pt").exists():
    preprocessor = load_preprocessor(directory)
    if (
        preprocessor.variables != SPECIFICATIONS
        or preprocessor.centering != CENTERING
        or preprocessor.ocean_support != OCEAN_SUPPORT
        or getattr(preprocessor, "mask_constant_cells", False) != MASK_CONSTANT_CELLS
        or (
            CENTERING == "gridpoint"
            and preprocessor.gridpoint_std_floor != GRIDPOINT_STD_FLOOR
        )
    ):
        raise ValueError("Variables or preprocessing changed; choose a new run")
else:
    preprocessor = Preprocessor(
        SPECIFICATIONS,
        centering=CENTERING,
        ocean_support=OCEAN_SUPPORT,
        gridpoint_std_floor=GRIDPOINT_STD_FLOOR,
        mask_constant_cells=MASK_CONSTANT_CELLS,
    )
    preprocessor.fit_component(
        training_fields, COMPONENT, statics, co2_values(co2, training_fields.time)
    )

# Give the network fixed, normalized geography; it pools these fields just once.
# New AEs use any-wet support on the first mesh and preserve wet-node skips.
# This is saved with the architecture; use a fresh run for the corrected masks.
grid = make_equiangular_grid_from_xarray(statics.lat, statics.lon, name="grid")
geometry = build_mesh_hierarchy(
    nlat=len(statics.lat),
    nlon=len(statics.lon),
    grid=grid,
    mesh_levels=MESH_LEVELS,
    rotation_degrees=ROTATION,
)
autoencoder = Autoencoder(
    channels=len(SPECIFICATIONS),
    geometry=geometry,
    statics=preprocessor.static_tensor(statics),
    preprocessor=preprocessor,
    grid_shape=(len(statics.lat), len(statics.lon)),
    width=WIDTH,
    latent_channels=LATENT_CHANNELS,
    refiners=REFINERS,
    co2_film_levels=CO2_FILM_LEVELS,
    condition_on_co2=CONDITION_ON_CO2,
    mask_aware=MASK_AWARE,
    refiner_mlp_ratio=REFINER_MLP_RATIO,
    mask_policy="support",  # corrected wet-node masks and residual paths
).to(device)
print(f"{COMPONENT}: {sum(p.numel() for p in autoencoder.parameters()):,} parameters")
logger.info(
    "CO2 conditioning: %s | training time limit: %g hours",
    CONDITION_ON_CO2,
    TIME_LIMIT_HOURS,
)
logger.info(
    "Preprocessing: %s | variable transforms: %s",
    CENTERING,
    ", ".join(f"{v.name}={v.transform}" for v in SPECIFICATIONS),
)

if MASK_CONSTANT_CELLS:
    for variable in SPECIFICATIONS:
        stats = preprocessor.stats[variable.name]
        logger.info(
            "%s mask: %d learning cells; %d constant cells restored on decode",
            variable.name,
            int(stats["mask"].sum()),
            int(stats["constant_mask"].sum()),
        )

save_autoencoder_setup(
    directory,
    autoencoder,
    preprocessor,
    statics,
    training_fields,
    validation_fields,
    co2,
)
latitudes = torch.tensor(statics.lat.values, dtype=torch.float32, device=device)
training_data = ComponentStates(
    training_fields, COMPONENT, autoencoder.preprocessor, co2
)
validation_data = ComponentStates(
    validation_fields, COMPONENT, autoencoder.preprocessor, co2
)
# Training controls are ordinary values used by the loop below.
BATCH_SIZE = 16
EPOCHS = 1000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
OPTIMIZER_EPS = 1e-6
EMA_DECAY = 0.995
PATIENCE = 5
MIN_DELTA = 1e-4
VALIDATION_SEED = 271828

if min(BATCH_SIZE, EPOCHS, PATIENCE, THREADS) < 1:
    raise ValueError("Batch size, epochs, patience and threads must be positive")
validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False)

optimizer = torch.optim.AdamW(
    autoencoder.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    eps=OPTIMIZER_EPS,
)
ema = EMA(autoencoder, EMA_DECAY)
# Start the training random stream independently of network initialization.
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
progress = load_training_progress(
    directory,
    autoencoder,
    optimizer,
    ema,
    resume=RESUME,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    patience=PATIENCE,
    min_delta=MIN_DELTA,
    seed=SEED,
    validation_seeds=(VALIDATION_SEED,),
)
first_epoch = progress.epoch + 1
if model_path.exists():
    print("Keeping the selected AE; checking its latent cache.")
    first_epoch = EPOCHS + 1

try:
    for epoch in range(first_epoch, EPOCHS + 1):
        check_time_limit(deadline)
        if progress.stale_epochs >= PATIENCE:
            break
        started = perf_counter()
        autoencoder.train()
        generator = torch.Generator().manual_seed(SEED + epoch)
        training_loader = DataLoader(
            training_data, batch_size=BATCH_SIZE, shuffle=True, generator=generator
        )
        train_sum, train_count = 0.0, 0
        train_mse = torch.zeros(len(SPECIFICATIONS), dtype=torch.float64)
        train_physical_mse = torch.zeros_like(train_mse)
        for state, physical, month, co2_value in training_loader:
            check_time_limit(deadline)
            state = state.to(device)
            month = month.to(device)
            co2_value = co2_value.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, latent = autoencoder(state, month, co2_value)
            per_field = reconstruction_mse(prediction, state, latitudes)
            # Each variable has unit-sum spatial weights; average variables equally.
            loss = per_field.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite reconstruction loss")
            loss.backward()
            optimizer.step()
            progress.optimizer_steps += 1
            ema.update(autoencoder, progress.optimizer_steps)
            train_sum += len(state) * float(loss.detach())
            train_count += len(state)
            # Undo centering/scaling AND nonlinear transforms before physical RMSE.
            physical_mse = reconstruction_mse(
                autoencoder.to_physical(prediction.detach()),
                physical.to(device),
                latitudes,
            )
            if not torch.isfinite(physical_mse).all():
                raise FloatingPointError("Nonfinite physical-space training error")
            train_mse += len(state) * per_field.detach().cpu().double()
            train_physical_mse += len(state) * physical_mse.cpu().double()
        train_loss = train_sum / train_count

        autoencoder.eval()
        validation_sum, validation_count = 0.0, 0
        validation_mse = torch.zeros_like(train_mse)
        validation_physical_mse = torch.zeros_like(train_mse)
        with (
            ema.applied(autoencoder),
            torch.no_grad(),
            fixed_rng(VALIDATION_SEED, device),
        ):
            for state, physical, month, co2_value in validation_loader:
                check_time_limit(deadline)
                state = state.to(device)
                month = month.to(device)
                co2_value = co2_value.to(device)
                prediction, latent = autoencoder(state, month, co2_value)
                per_field = reconstruction_mse(prediction, state, latitudes)
                validation_sum += len(state) * float(per_field.mean())
                validation_count += len(state)
                physical_mse = reconstruction_mse(
                    autoencoder.to_physical(prediction), physical.to(device), latitudes
                )
                if not torch.isfinite(physical_mse).all():
                    raise FloatingPointError(
                        "Nonfinite physical-space validation error"
                    )
                validation_mse += len(state) * per_field.cpu().double()
                validation_physical_mse += len(state) * physical_mse.cpu().double()
        validation_loss = validation_sum / validation_count
        if not np.isfinite(validation_loss):
            raise FloatingPointError("Nonfinite validation loss")

        # Select the best EMA, and count epochs without a meaningful improvement.
        if validation_loss < progress.best_validation:
            progress.best_validation = validation_loss
            save_best_weights(directory, ema, epoch, validation_loss)
        if validation_loss < progress.patience_best - MIN_DELTA:
            progress.patience_best = validation_loss
            progress.stale_epochs = 0
        else:
            progress.stale_epochs += 1
        progress.epoch = epoch
        train_rmse = (train_physical_mse / train_count).sqrt()
        validation_rmse = (validation_physical_mse / validation_count).sqrt()
        train_normalized_rmse = (train_mse / train_count).sqrt()
        validation_normalized_rmse = (validation_mse / validation_count).sqrt()
        epoch_seconds = perf_counter() - started  # training and validation
        add_autoencoder_history(
            progress,
            train_loss,
            validation_loss,
            epoch_seconds,
            variables=SPECIFICATIONS,
            train_rmse=train_rmse,
            validation_rmse=validation_rmse,
            train_normalized_rmse=train_normalized_rmse,
            validation_normalized_rmse=validation_normalized_rmse,
        )
        save_training_progress(directory, autoencoder, optimizer, ema, progress)
        logger.info(
            f"Epoch {epoch:04d} | train {train_loss:.6f} | validation {validation_loss:.6f}"
            f" | {epoch_seconds:.1f} s/epoch"
            f" | stale {progress.stale_epochs}/{PATIENCE}"
        )

        for index, variable in enumerate(SPECIFICATIONS):
            units = variable.units or training_fields[variable.name].attrs.get(
                "units", "unspecified units"
            )
            logger.info(
                "  %s [%s] | physical RMSE train %.6g validation %.6g | normalized RMSE train %.6g validation %.6g",
                variable.name,
                units,
                train_rmse[index],
                validation_rmse[index],
                train_normalized_rmse[index],
                validation_normalized_rmse[index],
            )

except TrainingTimeLimit as error:
    if not (directory / "best.pt").exists():
        raise RuntimeError(
            "Time limit reached before any epoch finished validation; "
            "increase TIME_LIMIT_HOURS before restarting."
        ) from error
    logger.info(
        "Training time limit reached; exporting the best completed validation epoch."
    )
    # An unfinished epoch is discarded. latest.pt remains the last complete
    # epoch, and best.pt supplies the weights used by every following stage.

selected_epoch = select_best_weights(directory, autoencoder)

# Freeze the selected AE and encode each split once. The next script reads these
# tensors directly; it does not retrain the AE or refit its normalization.
if (directory / "latents.pt").exists():
    autoencoder.latent_normalizer = load_latents(RUN_DIR, COMPONENT)["normalizer"]
else:
    encoded_splits = []
    with torch.no_grad():
        for dataset in (training_data, validation_data):
            encoded_batches = []
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
            for state, physical, month, co2_value in loader:
                latent = autoencoder.encode(
                    state.to(device),
                    month.to(device),
                    co2_value.to(device),
                )
                encoded_batches.append(latent.cpu())
            encoded_splits.append(torch.cat(encoded_batches))
    training_latents, validation_latents = encoded_splits
    autoencoder.latent_normalizer.fit(training_latents)
    save_latent_cache(
        directory,
        training_latents,
        validation_latents,
        autoencoder.latent_normalizer,
        training_fields.time,
        validation_fields.time,
    )
# A portable file contains the selected AE, fitted preprocessing and latent scaler.
if not (directory / "autoencoder.pt").exists():
    autoencoder.save(directory / "autoencoder.pt")
logger.info(
    f"Selected epoch {selected_epoch}; saved AE and latent cache in {directory}"
)
