from pathlib import Path
import sys, atexit, csv
from datetime import datetime
import torch
import xarray as xr
from torch.utils.data import DataLoader, TensorDataset
from time import perf_counter
from random_word import RandomWords
from monthly_diffusion.model import ( MeshStage, RefinedIcosahedralMeshEncoder, 
    Preprocessor, RefinedIcosahedralMeshDecoder, 
    ScalarToLatentChannels
)
from monthly_diffusion.utilities import Tee, close_diagnostics


DATA = Path("/Users/kyle/Dev/downloads/e3smv2_monthly_1deg_r1i1p1f1_185001-210012.nc")
VARIABLES = [
    'd20', 'mld', 'ohc300', 'ohc300_700', 'sic', 'sit', 'ssh', 'sss', 'sst'
]
BATCH_SIZE = 8
LATENT_CHANNELS = 18

N_SCALARS = 2 
CONDITION_CHANNELS = LATENT_CHANNELS
EPOCHS = 500
LR = 1e-3

TRAIN_PERIOD = slice("1966-01", "2006-12")
VAL_PERIOD = slice("1958-01", "1965-12")
PATIENCE = 10


### Overwrite this if you have already made the directory
HOME_DIR = Path("../output/heptagon") #+ RandomWords().get_random_word())
RUN_DIR = HOME_DIR / "ocean"

# Keep a console transcript alongside the metrics and checkpoint for each run.
RUN_DIR.mkdir(parents=True, exist_ok=True)
diagnostics_file = (RUN_DIR / "ocean_training_diagnostics.txt").open("w", encoding="utf-8", buffering=1)
console_stdout, console_stderr = sys.stdout, sys.stderr

sys.stdout = Tee(console_stdout, diagnostics_file)
sys.stderr = Tee(console_stderr, diagnostics_file)
atexit.register(close_diagnostics, console_stdout, console_stderr, diagnostics_file)
print(f"Saving results to {RUN_DIR}")

# Select months first; keep the raw data lazy while fitting statistics.
ds = xr.open_dataset(DATA)
train_da = ( 
    ds[VARIABLES]
    .sel(time=TRAIN_PERIOD)
    .to_array("variable")
)


val_da =  ( 
    ds[VARIABLES].sel(time=VAL_PERIOD)
    .to_array("variable")
)

statics = xr.concat([
    ds.lsm,
    (ds.phi_s - ds.phi_s.mean()) / ds.phi_s.std() 
], 'variable').assign_coords({'variable': ['lsm', 'phi_s']})
statics = torch.tensor(
    statics.stack(point=('lat', 'lon')).transpose('point', 'variable').values, 
    dtype=torch.float32
)

train_months = torch.tensor(train_da.time.month.values, dtype=torch.float32).reshape(-1, 1)
val_months = torch.tensor(val_da.time.month.values, dtype=torch.float32).reshape(-1, 1)



train_conditions = torch.hstack([
    torch.cos(train_months / 12 * 2 * torch.pi), 
    torch.sin(train_months / 12 * 2 * torch.pi)
])

val_conditions = torch.hstack([
    torch.cos(val_months / 12 * 2 * torch.pi), 
    torch.sin(val_months / 12 * 2 * torch.pi)
])

preprocessor = Preprocessor(train_da, compact=True)
statics = preprocessor.select_points(statics, dim=0)
train_torch = preprocessor.xr_to_torch(train_da)
train_torch, train_observed = preprocessor.preprocess_torch(train_torch, return_mask=True)
del train_da # to clean up extra data

val_torch = preprocessor.xr_to_torch(val_da)
val_torch, val_observed = preprocessor.preprocess_torch(val_torch, return_mask=True)
del val_da # to clean up extra data

encoder_stages = [
    MeshStage(name="m4", refinement=4, width=128, refiners=4, map_to_next_stage="radius"),
    MeshStage(name="m3", refinement=3, width=128, refiners=4, map_to_next_stage="radius"),
    MeshStage(name="m2", refinement=2, width=128, refiners=4, map_to_next_stage="radius"),
]

decoder_stages = [
    MeshStage(name="m2", refinement=2, width=128, refiners=4, map_to_next_stage="barycentric"),
    MeshStage(name="m3", refinement=3, width=128, refiners=4, map_to_next_stage="barycentric"),
    MeshStage(name="m4", refinement=4, width=128, refiners=4, map_to_next_stage="barycentric"),
]


encoder = RefinedIcosahedralMeshEncoder(
    hierarchy=encoder_stages, 
    preprocessor = preprocessor,
    channels=preprocessor.channels + CONDITION_CHANNELS, 
    latent_channels=LATENT_CHANNELS,
)

decoder = RefinedIcosahedralMeshDecoder(
    hierarchy=decoder_stages,
    preprocessor = preprocessor,
    channels = preprocessor.channels, 
    latent_channels=LATENT_CHANNELS + CONDITION_CHANNELS,
)

ec = ScalarToLatentChannels(
    N_SCALARS,
    encoder.grid,
    out_channels = CONDITION_CHANNELS
)

dc = ScalarToLatentChannels(
    N_SCALARS,
    decoder.meshes[0],
    out_channels = CONDITION_CHANNELS
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# CPU tensors for the loaders; only the current batch goes to the device.
train_loader = DataLoader(
    TensorDataset(
        train_torch.to(device),
        train_conditions.to(device),
        train_observed.to(device),
    ), 
    batch_size=BATCH_SIZE, shuffle=True
)

val_loader = DataLoader(
    TensorDataset(
        val_torch.to(device),
        val_conditions.to(device),
        val_observed.to(device),
    ),
    batch_size=BATCH_SIZE, shuffle=False
)

if not (torch.isfinite(preprocessor.mean_torch).all()
        and torch.isfinite(preprocessor.std_torch).all()
        and (preprocessor.std_torch > 0).all()):
    raise ValueError("Every variable needs a finite training mean and positive SD.")

encoder = encoder.to(device)
decoder = decoder.to(device)
ec.to(device)
dc.to(device)
statics = statics.to(device)

# Train both sets of weights.
optimizer = torch.optim.AdamW(
    list(encoder.parameters()) + list(decoder.parameters()) + list(ec.parameters()) + list(dc.parameters()),
    lr=LR,
)

training_start = perf_counter()

log_path = RUN_DIR / "loss.csv"
checkpoint_path = RUN_DIR / "best.pt"

with log_path.open("w", newline="") as f:
    csv.writer(f).writerow([
        "epoch", "train_mse", "val_mse",
        *[f"val_rmse_{v}" for v in preprocessor.variables],
    ])

best_val_loss = float("inf")
EPOCHS_without_improvement = 0

for epoch in range(EPOCHS):
    epoch_start = perf_counter()

    encoder.train()
    decoder.train()
    ec.train()
    dc.train()

    total_loss = torch.zeros(preprocessor.channels, device=device)
    sample_count = torch.zeros(preprocessor.channels, device=device)

    for state, conditions, observed in train_loader:

        optimizer.zero_grad(set_to_none=True)

        in_conditions = ec(conditions)
        state_plus_conditions = torch.cat([state, in_conditions], dim=-1)
        latent = encoder(state_plus_conditions, statics)
        latent_conditions = dc(conditions)
        latent_plus_conditions = torch.cat([latent, latent_conditions], dim=-1)
        reconstruction = decoder(latent_plus_conditions, statics)

        mse_per_sample, present = preprocessor.masked_mse(
            reconstruction, state, observed,
        )
        counts = present.sum(dim=0)
        if not present.any():
            continue
        # Average observed samples per variable, then give variables equal weight.
        loss = (mse_per_sample.sum(dim=0) / counts.clamp_min(1))[counts > 0].mean()
        loss.backward()
        optimizer.step()

        # Accumulate detached metrics for the whole epoch.
        total_loss += mse_per_sample.detach().sum(dim=0)
        sample_count += counts

    # Validation uses the same preprocessing and loss weights.
    encoder.eval()
    decoder.eval()
    ec.eval()
    dc.eval()

    
    val_mse_sum = torch.zeros(preprocessor.channels, device=device)
    val_physical_mse_sum = torch.zeros(preprocessor.channels, device=device)
    val_samples = torch.zeros(preprocessor.channels, device=device)

    with torch.no_grad():
        for state, conditions, observed in val_loader:
            state = state.to(device)
            conditions = conditions.to(device)

            in_conditions = ec(conditions)
            state_plus_conditions = torch.cat([state, in_conditions], dim=-1)
            latent = encoder(state_plus_conditions, statics)

            latent_conditions = dc(conditions)
            latent_plus_conditions = torch.cat([latent, latent_conditions], dim=-1)
            reconstruction = decoder(latent_plus_conditions, statics)

            mse_per_sample, present = preprocessor.masked_mse(
                reconstruction, state, observed,
            )

            val_mse_sum += mse_per_sample.sum(dim=0)

            # Backgrounds cancel in the error; restore physical units directly.
            physical_mse_per_sample = mse_per_sample * preprocessor.std_torch.reshape(-1).square()

            val_physical_mse_sum += physical_mse_per_sample.sum(dim=0)
            val_samples += present.sum(dim=0)

    # An entirely unobserved variable has NaN metrics and is excluded from the mean.
    val_mse_per_variable = val_mse_sum / val_samples
    val_loss = val_mse_per_variable.nanmean().item()
    epoch_loss = (total_loss / sample_count).nanmean().item()

    # Average squared errors over all validation samples, THEN take the root.
    val_rmse = (
        val_physical_mse_sum / val_samples
    ).sqrt().cpu().tolist()

    epoch_seconds = perf_counter() - epoch_start
    elapsed_seconds = perf_counter() - training_start
    minutes, seconds = divmod(int(elapsed_seconds), 60)
    hours, minutes = divmod(minutes, 60)

    print(
        f"Epoch {epoch + 1:03d}/{EPOCHS} | "
        f"train MSE {epoch_loss:.6f} | "
        f"val MSE {val_loss:.6f} | "
        f"{epoch_seconds:.1f}s/epoch | "
        f"elapsed {hours:02d}:{minutes:02d}:{seconds:02d}"
    )

    for variable, rmse in zip(preprocessor.variables, val_rmse):
        print(f"  {variable}: validation RMSE = {rmse:.6g}")

    # Append one row per epoch; closing the file flushes it to disk.
    with log_path.open("a", newline="") as f:
        csv.writer(f).writerow([
            epoch + 1, epoch_loss, val_loss, *val_rmse,
        ])

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        EPOCHS_without_improvement = 0

        torch.save({
            "config": {
                "variables": VARIABLES,
                "encoder_stages": [vars(stage) for stage in encoder_stages],
                "decoder_stages": [vars(stage) for stage in decoder_stages],
                "latent_channels": LATENT_CHANNELS,
                "condition_channels": CONDITION_CHANNELS,
                "n_scalars": N_SCALARS,
                "train_period": [TRAIN_PERIOD.start, TRAIN_PERIOD.stop], 
                "val_period": [VAL_PERIOD.start, VAL_PERIOD.stop],
            },
            "coordinates": {
                dim: torch.tensor(preprocessor.coords[dim].values)
                for dim in ("lat", "lon")
            },
            "epoch": epoch + 1,
            "val_loss": val_loss,
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "ec": ec.state_dict(),
            "dc": dc.state_dict(),
            "preprocessor": preprocessor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "statics": statics,
        }, checkpoint_path)

        print(f"Saved best checkpoint: val MSE = {val_loss:.6f}")

    else:
        EPOCHS_without_improvement += 1
        print(f"No improvement: {EPOCHS_without_improvement}/{PATIENCE}")

        if EPOCHS_without_improvement >= PATIENCE:
            print(f"Early stopping. Best val MSE = {best_val_loss:.6f}")
            break
