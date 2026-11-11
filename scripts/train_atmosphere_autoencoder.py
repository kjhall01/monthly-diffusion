"""Train the atmosphere autoencoder. Edit the values below, then run this file."""

from pathlib import Path
from monthly_diffusion.run_files import check_script_arguments
from time import perf_counter
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from monthly_diffusion.modules.autoencoder import (
    MeshHierarchyAutoencoder,
    reconstruction_mse,
)
from monthly_diffusion.modules.mesh import (
    make_equiangular_grid_from_xarray,
    build_mesh_hierarchy_context,
)
from monthly_diffusion.data.configs import mdv1_variables
from monthly_diffusion.data.preprocessing import Preprocessor, LatentNormalizer
from monthly_diffusion.data.dataloader import ComponentStates, co2_values
from monthly_diffusion.data.sources import open_canonical_state, open_co2
from monthly_diffusion.inference.moments import component_moments
from monthly_diffusion.training.ema import EMA
from monthly_diffusion.training.checks import (
    check_component_data,
    check_resume_settings,
)
from monthly_diffusion.checkpoints import atomic_save
from monthly_diffusion.random_state import rng_state, restore_rng, fixed_rng
from monthly_diffusion.run_files import (
    create_run,
    file_digest,
    save_history,
    save_selection,
    load_latents,
)

# Use this SAME directory in every script belonging to this experiment.
check_script_arguments()

RUN_DIR = Path("runs/production")
COMPONENT = "atmosphere"
RESUME = False
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
THREADS = 4
LOAD_IN_MEMORY = False

# Network and preprocessing. These are passed directly to the actual AE below.
WIDTH = 102
LATENT_CHANNELS = 4
MESH_LEVELS = (4, 3, 2)
REFINERS = (6, 6, 4)
ROTATION = (17, 23, 11)
CO2_FILM_LEVELS = ()  # empty means all levels
CENTERING = "annual_zonal"
OCEAN_SUPPORT = (
    "per_variable"  # "sst" applies one training-SST support to all ocean fields
)
MASK_AWARE = False
SPECIFICATIONS = [v for v in mdv1_variables() if v.component == COMPONENT]

BATCH_SIZE = 4
EPOCHS = 1000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
OPTIMIZER_EPS = 1e-6
EMA_DECAY = 0.995
PATIENCE = 5
MIN_DELTA = 1e-4
SEED = 20260716
VALIDATION_SEED = 271828

# The first script records these paths/splits once for all subsequent stages.
DATA = Path(
    "/Users/kyle/Dev/monthly-diffusion-data/data/e3smv2-five-member-1850-2100/regular_latlon_1deg/e3smv2_monthly_1deg_r1i1p1f1_185001-210012.zarr"
)
CO2 = Path(
    "/Users/kyle/Dev/monthly-diffusion-data/data/co2-forcing-v1/products/co2_monthly_e3sm_ssp370_185001-210012.nc"
)
TRAIN_PERIOD = slice("1966-01", "2006-12")
VALIDATION_PERIOD = slice("1958-01", "1965-12")
source = dict(state_zarr=str(DATA), co2_netcdf=str(CO2))
run = create_run(RUN_DIR, source, TRAIN_PERIOD, VALIDATION_PERIOD)

# Read the pinned source and only transform this component's variables.
source = run["source"]
fields, _, statics = open_canonical_state(
    source["state_zarr"], expected_attrs=source.get("expected_attrs")
)
co2 = open_co2(source["co2_netcdf"], expected_sha256=source.get("co2_sha256"))
training_fields = fields.sel(time=slice(*run["training_period"]))
validation_fields = fields.sel(time=slice(*run["validation_period"]))
statics = statics[["phi_s", "lsm"]].transpose("lat", "lon").load()
if LOAD_IN_MEMORY:
    training_fields.load()
    validation_fields.load()

# One self-contained component directory; no other networks are constructed.
directory = RUN_DIR / COMPONENT
directory.mkdir(parents=True, exist_ok=True)
model_path, setup_path = directory / "model.pt", directory / "setup.pt"
if model_path.exists() and not RESUME:
    raise FileExistsError(
        f"{model_path} already exists; set RESUME=True or choose a new RUN_DIR"
    )
parameters = dict(
    width=WIDTH,
    latent_channels=LATENT_CHANNELS,
    mesh_levels=MESH_LEVELS,
    refiners=REFINERS,
    rotation=ROTATION,
    co2_film_levels=CO2_FILM_LEVELS,
    mask_aware=MASK_AWARE,
)
settings = dict(
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    optimizer_eps=OPTIMIZER_EPS,
    ema_decay=EMA_DECAY,
    patience=PATIENCE,
    min_delta=MIN_DELTA,
    seed=SEED,
    validation_seed=VALIDATION_SEED,
)
if min(BATCH_SIZE, EPOCHS, PATIENCE, THREADS) < 1:
    raise ValueError("Batch size, epochs, patience and threads must be positive")
torch.set_num_threads(THREADS)
device = torch.device(DEVICE)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if setup_path.exists():
    setup = torch.load(setup_path, map_location="cpu", weights_only=False)
    check_component_data(
        training_fields, validation_fields, statics, SPECIFICATIONS, co2, setup
    )
    preprocessor = setup["preprocessor"]
    if (
        setup["parameters"] != parameters
        or preprocessor.centering != CENTERING
        or preprocessor.ocean_support != OCEAN_SUPPORT
    ):
        raise ValueError("AE or preprocessing settings changed; choose a new run")
else:
    check_component_data(
        training_fields, validation_fields, statics, SPECIFICATIONS, co2
    )
    preprocessor = Preprocessor(
        SPECIFICATIONS, centering=CENTERING, ocean_support=OCEAN_SUPPORT
    )
    preprocessor.fit_component(
        training_fields, COMPONENT, statics, co2_values(co2, training_fields.time)
    )
    setup = dict(
        parameters=parameters,
        preprocessor=preprocessor,
        statics=statics,
        validation_times=validation_fields.time.values,
        validation_co2=co2_values(co2, validation_fields.time),
        validation_attrs=dict(validation_fields.attrs),
        initialization_moments=component_moments(
            preprocessor, training_fields, COMPONENT
        ),
    )
    atomic_save(setup, setup_path)
setup_digest = file_digest(setup_path)

grid = make_equiangular_grid_from_xarray(statics.lat, statics.lon, name="grid")
geometry = build_mesh_hierarchy_context(
    nlat=len(statics.lat),
    nlon=len(statics.lon),
    mesh_levels=MESH_LEVELS,
    rotation_degrees=ROTATION,
    grid=grid,
)
autoencoder = MeshHierarchyAutoencoder(
    channels=len(SPECIFICATIONS),
    geometry=geometry,
    grid_shape=(len(statics.lat), len(statics.lon)),
    width=WIDTH,
    latent_channels=LATENT_CHANNELS,
    mesh_levels=MESH_LEVELS,
    refiners=REFINERS,
    rotation=ROTATION,
    co2_film_levels=CO2_FILM_LEVELS,
    mask_aware=MASK_AWARE,
).to(device)
print(
    f"{COMPONENT}: {sum(p.numel() for p in autoencoder.parameters()):,} parameters; saving to {directory}"
)
static_tensor = preprocessor.static_tensor(statics).to(device)[None]
latitudes = torch.tensor(statics.lat.values, device=device)
weights = torch.tensor([v.weight for v in SPECIFICATIONS], device=device)
training_data = ComponentStates(training_fields, COMPONENT, preprocessor, co2)
validation_data = ComponentStates(validation_fields, COMPONENT, preprocessor, co2)
validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False)

# The optimizer, reconstruction loss, validation and selection are all here.
optimizer = torch.optim.AdamW(
    autoencoder.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    eps=OPTIMIZER_EPS,
)
ema = EMA(autoencoder, EMA_DECAY)
# Separate weight initialization from the training RNG, as in the original trainer.
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
history, step, first_epoch, stale = [], 0, 1, 0
best, patience_best = float("inf"), float("inf")
latest_path = directory / "latest.pt"
if latest_path.exists() and not model_path.exists():
    if not RESUME:
        raise FileExistsError("Interrupted training exists; set RESUME=True")
    saved = torch.load(latest_path, map_location=device, weights_only=False)
    check_resume_settings(saved["settings"], settings)
    if saved["setup_digest"] != setup_digest:
        raise ValueError("Preprocessing changed since the interrupted training")
    autoencoder.load_state_dict(saved["model"])
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
if model_path.exists():
    print("Keeping the selected AE; checking its latent cache.")
    first_epoch = EPOCHS + 1

for epoch in range(first_epoch, EPOCHS + 1):
    if stale >= PATIENCE:
        break
    started = perf_counter()
    autoencoder.train()
    generator = torch.Generator().manual_seed(SEED + epoch)
    training_loader = DataLoader(
        training_data, batch_size=BATCH_SIZE, shuffle=True, generator=generator
    )
    train_sum, train_count = 0.0, 0
    for batch in training_loader:
        batch = {name: value.to(device) for name, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        prediction, _ = autoencoder(
            batch["state"],
            static_tensor.expand(len(batch["state"]), -1, -1, -1),
            batch["month"],
            batch["co2"],
        )
        per_field = reconstruction_mse(prediction, batch["state"], latitudes)
        loss = (per_field * weights).sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite reconstruction loss")
        loss.backward()
        optimizer.step()
        step += 1
        ema.update(autoencoder, step)
        train_sum += len(batch["state"]) * float(loss.detach())
        train_count += len(batch["state"])

    autoencoder.eval()
    validation_sum, validation_count = 0.0, 0
    with ema.applied(autoencoder), torch.no_grad(), fixed_rng(VALIDATION_SEED, device):
        for batch in validation_loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            prediction, _ = autoencoder(
                batch["state"],
                static_tensor.expand(len(batch["state"]), -1, -1, -1),
                batch["month"],
                batch["co2"],
            )
            per_field = reconstruction_mse(prediction, batch["state"], latitudes)
            validation_sum += len(batch["state"]) * float((per_field * weights).sum())
            validation_count += len(batch["state"])
    score = validation_sum / validation_count
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
    history.append(
        dict(
            epoch=epoch,
            optimizer_steps=step,
            train_loss=train_sum / train_count,
            val_loss=score,
            seconds=perf_counter() - started,
            stale_epochs=stale,
        )
    )
    atomic_save(
        dict(
            model=autoencoder.state_dict(),
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
        f"Epoch {epoch:04d} | train {train_sum / train_count:.6f} | validation {score:.6f} | stale {stale}/{PATIENCE}"
    )

if not model_path.exists():
    save_selection(
        directory,
        parameters=parameters,
        setup_digest=setup_digest,
        training_settings=settings,
    )
selected = torch.load(model_path, map_location=device, weights_only=False)
if selected["setup_digest"] != setup_digest:
    raise ValueError("Selected AE belongs to different preprocessing")
autoencoder.load_state_dict(selected["model"])
autoencoder.eval().requires_grad_(False)

# Encode the selected AE once. Scaling is fitted ONLY on training latents and
# saved with this component, so diffusion and reconstruction use identical units.
if (directory / "latents.pt").exists():
    load_latents(RUN_DIR, COMPONENT)
else:
    latents = {}
    with torch.no_grad():
        for split, dataset in (
            ("training", training_data),
            ("validation", validation_data),
        ):
            encoded = []
            for batch in DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False):
                batch = {name: value.to(device) for name, value in batch.items()}
                z = autoencoder.encode(
                    batch["state"],
                    static_tensor.expand(len(batch["state"]), -1, -1, -1),
                    batch["month"],
                    batch["co2"],
                )
                encoded.append(z.cpu())
            latents[split] = torch.cat(encoded)
    normalizer = LatentNormalizer().fit(latents["training"])
    atomic_save(
        dict(
            training=normalizer.normalize(latents["training"]),
            validation=normalizer.normalize(latents["validation"]),
            normalizer=normalizer,
            training_time=training_fields.time,
            validation_time=validation_fields.time,
            model_digest=file_digest(model_path),
            setup_digest=setup_digest,
        ),
        directory / "latents.pt",
    )
print(
    f"Selected epoch {selected['epoch']}; saved AE and normalized latent cache in {directory}"
)
