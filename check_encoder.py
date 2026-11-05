"""From the repo root: python check_encoder.py (with monthly_diffusion installed)."""
from pathlib import Path

import torch
import xarray as xr
from torch.utils.data import DataLoader, TensorDataset
from time import perf_counter
from monthly_diffusion.architecture import MeshStage, RefinedIcosahedralMeshEncoder, Preprocessor, RefinedIcosahedralMeshDecoder , ScalarToLatent

DATA = Path("../downloads/mdv1-e3smv2-r1i1p1f1-training-validation-1deg-1958-2006.nc").expanduser()
VARIABLES = [
    "sfcWind", "t2m", "tcwv", "mtpr", "mslp", "olr", "z500", "tau300_700",
    "t850", "t500", "u200", "v200", "u50", "ssrd", "tau_x", "tau_y"
]
BATCH_SIZE = 8
LATENT_CHANNELS = 32


# Select months first; keep the raw data lazy while fitting statistics.
ds = xr.open_dataset(DATA)
train_da = ( 
    ds[VARIABLES]
    .sel(time=slice("1966-01", "2006-12"))
    .to_array("variable")
)


val_da =  ( 
    ds[VARIABLES].sel(time=slice("1958-01", "1965-12"))
    .to_array("variable")
)



train_months = torch.tensor(train_da.time.month.values, dtype=torch.float32).reshape(-1, 1)
val_months = torch.tensor(val_da.time.month.values, dtype=torch.float32).reshape(-1, 1)



train_co2 = torch.tensor(
    ds.sel(time=slice("1966-01", "2006-12")).co2.values,
    dtype=torch.float32
).reshape(-1, 1)

co2_mean = train_co2.mean() 
co2_std = train_co2.std() 

val_co2 = torch.tensor(
    ds.sel(time=slice("1958-01", "1965-12")).co2.values,
    dtype=torch.float32
).reshape(-1, 1)

train_co2 = (train_co2 - co2_mean) / co2_std 
val_co2 = (val_co2 - co2_mean) / co2_std 


train_conditions = torch.hstack([
    train_co2, 
    torch.cos(train_months / 12 * 2 * torch.pi), 
    torch.sin(train_months / 12 * 2 * torch.pi)
])

val_conditions = torch.hstack([
    val_co2, 
    torch.cos(val_months / 12 * 2 * torch.pi), 
    torch.sin(val_months / 12 * 2 * torch.pi)
])

preprocessor = Preprocessor(train_da)
train_torch = preprocessor.xr_to_torch(train_da)
train_torch = preprocessor.preprocess_torch(train_torch)
del train_da # to clean up extra data

val_torch = preprocessor.xr_to_torch(val_da)
val_torch = preprocessor.preprocess_torch(val_torch)
del val_da # to clean up extra data

encoder_stages = [
    MeshStage(name="m4", refinement=4, width=128, refiners=4, map_to_next_stage="radius"),
    MeshStage(name="m3", refinement=3, width=128, refiners=6, map_to_next_stage="radius"),
    MeshStage(name="m2", refinement=2, width=128, refiners=6, map_to_next_stage="radius"),
]

decoder_stages = [
    MeshStage(name="m2", refinement=2, width=128, refiners=6, map_to_next_stage="barycentric"),
    MeshStage(name="m3", refinement=3, width=128, refiners=6, map_to_next_stage="barycentric"),
    MeshStage(name="m4", refinement=4, width=128, refiners=4, map_to_next_stage="barycentric"),
]

N_SCALARS = 3
CONDITION_CHANNELS = 1

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

ec = ScalarToLatent(
    N_SCALARS,
    encoder.grid,
    out_channels = CONDITION_CHANNELS
)

dc = ScalarToLatent(
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
    ), 
    batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(
        val_torch.to(device),
        val_conditions.to(device),
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

n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
print(f"Encoder trainable parameters: {n_params:,}")
n_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
print(f"Dncoder trainable parameters: {n_params:,}")
preprocessor = preprocessor.to(device)

# Train both sets of weights.
optimizer = torch.optim.AdamW(
    list(encoder.parameters()) + list(decoder.parameters()) + list(ec.parameters()) + list(dc.parameters()),
    lr=1e-3,
)

epochs = 200
training_start = perf_counter()

for epoch in range(epochs):
    epoch_start = perf_counter()

    encoder.train()
    decoder.train()
    ec.train()
    dc.train()

    total_loss = torch.zeros((), device=device)
    total_physical_mse = torch.zeros(
        preprocessor.channels, device=device
    )
    sample_count = 0

    for (state, conditions) in train_loader:
        #state = state.to(device)                    # [B, H*W, C]

        optimizer.zero_grad(set_to_none=True)

        in_conditions = ec(conditions)
        state_plus_conditions = torch.cat([state, in_conditions], dim=-1)
        latent = encoder(state_plus_conditions)
        latent_conditions = dc(conditions)
        latent_plus_conditions = torch.cat([latent, latent_conditions], dim=-1)
        reconstruction = decoder(latent_plus_conditions)

        # Exclude invalid locations before squaring.
        error = torch.where(
            preprocessor.valid_mask_torch,
            reconstruction - state,
            0.0,
        )

        # Spatial weights sum to 1 separately for each variable.
        mse_per_sample = (
            error.square() * preprocessor.loss_weights
        ).sum(dim=1)                                # [B, C]

        loss = mse_per_sample.mean()                # Mean over batch and variables
        loss.backward()
        optimizer.step()

        # Accumulate detached metrics for the whole epoch.
        batch_size = state.shape[0]
        total_loss += loss.detach() * batch_size

        # Physical error = normalized error * training SD.
        scale_squared = preprocessor.std_torch.reshape(-1)#.square()
        total_physical_mse += (
            mse_per_sample.detach() * scale_squared
        ).sum(dim=0)
        sample_count += batch_size

    # Validation uses the same preprocessing and loss weights.
    encoder.eval()
    decoder.eval()
    ec.eval()
    dc.eval()

    val_mse_sum = torch.zeros(preprocessor.channels, device=device)
    val_samples = 0

    with torch.no_grad():
        for (state, conditions ) in val_loader:
            in_conditions = ec(conditions)
            state_plus_conditions = torch.cat([state, in_conditions], dim=-1)
            latent = encoder(state_plus_conditions)
            latent_conditions = dc(conditions)
            latent_plus_conditions = torch.cat([latent, latent_conditions], dim=-1)
            reconstruction = decoder(latent_plus_conditions)

            error = torch.where(
                preprocessor.valid_mask_torch,
                reconstruction - state,
                0.0,
            )
            mse_per_sample = (
                error.square() * preprocessor.loss_weights
            ).sum(dim=1)                        # [B, C]

            val_mse_sum += mse_per_sample.sum(dim=0)
            val_samples += state.shape[0]

    val_mse_per_variable = val_mse_sum / val_samples
    val_loss = val_mse_per_variable.mean().item()

    epoch_loss = (total_loss / sample_count).item()

    # Convert normalized RMSE into each variable's physical units.
    val_rmse = (
        val_mse_per_variable
        * preprocessor.std_torch.reshape(-1).square()
    ).sqrt().cpu().tolist()

    # Reading the metrics above waits for device computations to finish,
    # so these timings include the completed training and validation work.
    epoch_seconds = perf_counter() - epoch_start
    elapsed_seconds = perf_counter() - training_start
    minutes, seconds = divmod(int(elapsed_seconds), 60)
    hours, minutes = divmod(minutes, 60)

    print(
        f"Epoch {epoch + 1:03d}/{epochs} | "
        f"train MSE {epoch_loss:.6f} | "
        f"val MSE {val_loss:.6f} | "
        f"{epoch_seconds:.1f}s/epoch | "
        f"elapsed {hours:02d}:{minutes:02d}:{seconds:02d}"
    )

    for variable, rmse in zip(preprocessor.variables, val_rmse):
        print(f"  {variable}: validation RMSE = {rmse:.6g}")
