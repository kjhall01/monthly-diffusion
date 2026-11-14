"""Train the spherical AE on streamed six-hourly WeatherBench2 ERA5 anomalies.

Edit the source, variables and controls below. No raw/normalized dataset cache is
written. Statistics require one full training-data pass before the first epoch.
"""

import logging
import random
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from monthly_diffusion import Autoencoder, Variable
from monthly_diffusion.checkpoints import load_checkpoint, atomic_save
from monthly_diffusion.mesh import (
    make_equiangular_grid_from_xarray,
    build_mesh_hierarchy,
)
from monthly_diffusion.preprocessing import validate_fields, validate_units
from monthly_diffusion.rollout import fixed_rng
from monthly_diffusion.weather import WeatherPreprocessor
from workflows.data.weatherbench import (
    open_weatherbench,
    select_weather_fields,
    weather_statics,
    stream_weather,
    weather_tensors,
    record_weather_run,
    save_weather_setup,
)
from workflows.run_files import check_script_arguments
from workflows.training.ema import EMA
from workflows.training.files import (
    load_training_progress,
    save_best_weights,
    save_training_progress,
    add_autoencoder_history,
    select_best_weights,
)
from workflows.training.logging import training_logger
from workflows.training.losses import reconstruction_mse
from workflows.training.time_limit import TrainingTimeLimit, check_time_limit

check_script_arguments()

RUN_DIR = Path(__file__).resolve().parent / "runs/weatherbench2_anomalies"
DATA = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
CLIMATOLOGY = "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_240x121_equiangular_with_poles_conservative.zarr"
# Climatology years are contained in training; validation does not contribute.
TRAIN_PERIOD = slice("1990-01-01", "2017-12-31")
VALIDATION_PERIOD = slice("2018-01-01", "2019-12-31")
CLIMATOLOGY_YEARS = (1990, 2017)  # change together with CLIMATOLOGY

SURFACE_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_column_water_vapour",
    "total_precipitation_6hr",
]
PRESSURE_VARIABLES = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
# Six-hour precipitation remains metres of accumulated water, not a monthly rate.
# Anomalies can be negative: all 71 channels use identity transforms.

RESUME = False
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
THREADS = 4
SEED = 20260716
WIDTH = 64
LATENT_CHANNELS = 32
MESH_LEVELS = (4, 3, 2)
REFINERS = (4, 6, 6)
ROTATION = (17, 23, 11)
REFINER_MLP_RATIO = 2
GRIDPOINT_STD_FLOOR = 0.05
BATCH_SIZE = 16  # two native eight-timestep chunks; never smaller than eight
EPOCHS = 1000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
OPTIMIZER_EPS = 1e-6
EMA_DECAY = 0.995
PATIENCE = 5
MIN_DELTA = 1e-4
VALIDATION_SEED = 271828
TIME_LIMIT_HOURS = 4.0
# Fit latent scaling on a fixed random training subset AFTER selecting weights.
# This avoids a second full archive pass; no latent time-series cache is written.
LATENT_NORMALIZATION_BATCHES = 128

record_weather_run(
    RUN_DIR,
    source=DATA,
    climatology=CLIMATOLOGY,
    surface_variables=SURFACE_VARIABLES,
    pressure_variables=PRESSURE_VARIABLES,
    levels=LEVELS,
    training_period=TRAIN_PERIOD,
    validation_period=VALIDATION_PERIOD,
)
directory = RUN_DIR / "atmosphere"
directory.mkdir(parents=True, exist_ok=True)
logger = training_logger(directory)
statistics_logger = logging.getLogger("monthly_diffusion.weather")
statistics_logger.handlers = logger.handlers
statistics_logger.setLevel(logging.INFO)
statistics_logger.propagate = False
model_path = directory / "model.pt"
if not RESUME and any(
    (directory / name).exists() for name in ("model.pt", "latest.pt")
):
    raise FileExistsError("Training exists; set RESUME=True or choose a new RUN_DIR")
if min(BATCH_SIZE, EPOCHS, PATIENCE, THREADS, LATENT_NORMALIZATION_BATCHES) < 1:
    raise ValueError("Training counts must be positive")
if not 0 < TIME_LIMIT_HOURS < float("inf"):
    raise ValueError("TIME_LIMIT_HOURS must be positive and finite")
torch.set_num_threads(THREADS)
device = torch.device(DEVICE)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# These calls open metadata and coordinates; field arrays remain lazy.
dataset = open_weatherbench(DATA)
climatology_source = open_weatherbench(CLIMATOLOGY)
fields = select_weather_fields(dataset, SURFACE_VARIABLES, PRESSURE_VARIABLES, LEVELS)
climatology = select_weather_fields(
    climatology_source, SURFACE_VARIABLES, PRESSURE_VARIABLES, LEVELS
)
statics = weather_statics(dataset)
training_fields = fields.sel(time=TRAIN_PERIOD)
validation_fields = fields.sel(time=VALIDATION_PERIOD)
SPECIFICATIONS = [
    Variable(name, "atmosphere", fields[name].attrs.get("units", "")) for name in fields
]
validate_units(climatology, SPECIFICATIONS)
for split in (training_fields, validation_fields):
    validate_fields(split, list(fields), statics, cadence="6h")
if np.intersect1d(training_fields.time, validation_fields.time).size:
    raise ValueError("Training and validation dates overlap")
if (
    int(training_fields.time.dt.year.min()) > CLIMATOLOGY_YEARS[0]
    or int(training_fields.time.dt.year.max()) < CLIMATOLOGY_YEARS[1]
    or int(validation_fields.time.dt.year.min()) <= CLIMATOLOGY_YEARS[1]
):
    raise ValueError("Climatology years must lie within training and before validation")
for coord in ("lat", "lon"):
    if not np.array_equal(fields[coord], climatology[coord]):
        raise ValueError("ERA5 and climatology grids differ")
logger.info(
    "WeatherBench2: %d channels | %d training / %d validation timestamps | no disk data cache",
    len(SPECIFICATIONS),
    len(training_fields.time),
    len(validation_fields.time),
)

if (directory / "setup.pt").exists():
    preprocessor = load_checkpoint(directory / "setup.pt")["preprocessor"]
    if (
        preprocessor.variables != SPECIFICATIONS
        or preprocessor.gridpoint_std_floor != GRIDPOINT_STD_FLOOR
    ):
        raise ValueError("Weather preprocessing changed; choose a new RUN_DIR")
else:
    preprocessor = WeatherPreprocessor(
        SPECIFICATIONS, climatology, gridpoint_std_floor=GRIDPOINT_STD_FLOOR
    )
    logger.info("Fitting anomaly mean/std in one streaming pass through training data.")
    preprocessor.fit_batches(
        stream_weather(training_fields, BATCH_SIZE),
        statics,
        training_fields.time,
        training_fields.attrs,
    )

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
    grid_shape=(len(statics.lat), len(statics.lon)),
    statics=preprocessor.static_tensor(statics),
    preprocessor=preprocessor,
    width=WIDTH,
    latent_channels=LATENT_CHANNELS,
    refiners=REFINERS,
    refiner_mlp_ratio=REFINER_MLP_RATIO,
    mask_aware=True,
    mask_policy="support",
    residual_film_only=True,
    condition_on_co2=False,
    calendar="day_hour",
).to(device)
save_weather_setup(
    directory,
    autoencoder,
    preprocessor,
    statics,
    training_fields.time,
    validation_fields.time,
)
logger.info(
    "AE: %s parameters | year/day FiLM on residual updates | CO2 disabled",
    f"{sum(p.numel() for p in autoencoder.parameters()):,}",
)
latitudes = torch.tensor(statics.lat.values, dtype=torch.float32, device=device)

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
# The four-hour limit covers training/validation, after the one-time statistics pass.
deadline = perf_counter() + TIME_LIMIT_HOURS * 3600
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
        training_loader = stream_weather(
            training_fields, BATCH_SIZE, shuffle=True, seed=SEED + epoch
        )
        train_sum, train_count = 0.0, 0
        train_mse = torch.zeros(len(SPECIFICATIONS), dtype=torch.float64)
        train_physical_mse = torch.zeros_like(train_mse)
        for fields in training_loader:
            check_time_limit(deadline)
            state, physical, calendar = weather_tensors(fields, preprocessor, device)
            optimizer.zero_grad(set_to_none=True)
            prediction, latent = autoencoder(state, calendar, None)
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
                autoencoder.to_physical(prediction.detach(), time=fields.time),
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
            for fields in stream_weather(validation_fields, BATCH_SIZE):
                check_time_limit(deadline)
                state, physical, calendar = weather_tensors(
                    fields, preprocessor, device
                )
                prediction, latent = autoencoder(state, calendar, None)
                per_field = reconstruction_mse(prediction, state, latitudes)
                validation_sum += len(state) * float(per_field.mean())
                validation_count += len(state)
                physical_mse = reconstruction_mse(
                    autoencoder.to_physical(prediction, time=fields.time),
                    physical,
                    latitudes,
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

# Export the selected network, preprocessing and a training-only latent scaler.
# Only this bounded latent sample is kept in RAM; no data/latent cache is saved.
if not (directory / "autoencoder.pt").exists():
    encoded_batches = []
    with torch.no_grad():
        for fields in stream_weather(
            training_fields,
            BATCH_SIZE,
            shuffle=True,
            seed=SEED,
            max_batches=LATENT_NORMALIZATION_BATCHES,
        ):
            state, _, calendar = weather_tensors(fields, preprocessor, device)
            encoded_batches.append(autoencoder.encode(state, calendar, None).cpu())
    autoencoder.latent_normalizer.fit(torch.cat(encoded_batches))
    selected = load_checkpoint(model_path)
    selected["latent_normalizer"] = autoencoder.latent_normalizer
    atomic_save(selected, model_path)
    autoencoder.save(directory / "autoencoder.pt")
logger.info(
    "Selected epoch %d; saved weather AE in %s",
    selected_epoch,
    directory / "autoencoder.pt",
)
