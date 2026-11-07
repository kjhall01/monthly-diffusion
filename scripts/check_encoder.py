"""From the repo root: python check_encoder.py (with monthly_diffusion installed)."""
from pathlib import Path

import torch
import xarray as xr
from torch.utils.data import DataLoader, TensorDataset
from time import perf_counter
from monthly_diffusion.architecture import MeshStage, RefinedIcosahedralMeshEncoder, Preprocessor, RefinedIcosahedralMeshDecoder , ScalarToLatentChannels

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
#    MeshStage(name="m5", refinement=5, width=128, refiners=2, map_to_next_stage="radius"),
    MeshStage(name="m4", refinement=4, width=64, refiners=1, map_to_next_stage="radius"),
    MeshStage(name="m3", refinement=3, width=64, refiners=1, map_to_next_stage="radius"),
    MeshStage(name="m2", refinement=2, width=64, refiners=1, map_to_next_stage="radius"),
]

decoder_stages = [
    MeshStage(name="m2", refinement=2, width=64, refiners=1, map_to_next_stage="barycentric"),
    MeshStage(name="m3", refinement=3, width=64, refiners=1, map_to_next_stage="barycentric"),
    MeshStage(name="m4", refinement=4, width=64, refiners=1, map_to_next_stage="barycentric"),
#    MeshStage(name="m5", refinement=5, width=128, refiners=2, map_to_next_stage="barycentric"),

]

N_SCALARS = 3
CONDITION_CHANNELS = 3

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
statics = statics.to(device)

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

import csv
from datetime import datetime

# Separate output folder for each run.
run_dir = Path("output") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
run_dir.mkdir(parents=True, exist_ok=True)

log_path = run_dir / "loss.csv"
checkpoint_path = run_dir / "best.pt"

with log_path.open("w", newline="") as f:
    csv.writer(f).writerow([
        "epoch", "train_mse", "val_mse",
        *[f"val_rmse_{v}" for v in preprocessor.variables],
    ])

best_val_loss = float("inf")
epochs_without_improvement = 0
patience = 5

print(f"Saving results to {run_dir}")

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
        latent = encoder(state_plus_conditions, statics)
        latent_conditions = dc(conditions)
        latent_plus_conditions = torch.cat([latent, latent_conditions], dim=-1)
        reconstruction = decoder(latent_plus_conditions, statics)

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
    val_physical_mse_sum = torch.zeros(preprocessor.channels, device=device)
    val_samples = 0

    with torch.no_grad():
        for state, conditions in val_loader:
            state = state.to(device)
            conditions = conditions.to(device)

            in_conditions = ec(conditions)
            state_plus_conditions = torch.cat([state, in_conditions], dim=-1)
            latent = encoder(state_plus_conditions, statics)

            latent_conditions = dc(conditions)
            latent_plus_conditions = torch.cat([latent, latent_conditions], dim=-1)
            reconstruction = decoder(latent_plus_conditions, statics)

            # Normalized error: comparable to the training objective.
            error = torch.where(
                preprocessor.valid_mask_torch,
                reconstruction - state,
                0.0,
            )
            mse_per_sample = (
                error.square() * preprocessor.loss_weights
            ).sum(dim=1)  # [B, C]

            val_mse_sum += mse_per_sample.sum(dim=0)

            # Restore physical units AND the fixed zonal background.
            physical_state = preprocessor.postprocess_torch(state)
            physical_reconstruction = preprocessor.postprocess_torch(reconstruction)

            # Postprocessing restores NaNs at invalid locations;
            # exclude those before squaring.
            physical_error = torch.where(
                preprocessor.valid_mask_torch,
                physical_reconstruction - physical_state,
                0.0,
            )
            physical_mse_per_sample = (
                physical_error.square() * preprocessor.loss_weights
            ).sum(dim=1)  # [B, C]

            val_physical_mse_sum += physical_mse_per_sample.sum(dim=0)
            val_samples += state.shape[0]

    val_mse_per_variable = val_mse_sum / val_samples
    val_loss = val_mse_per_variable.mean().item()
    epoch_loss = (total_loss / sample_count).item()

    # Average squared errors over all validation samples, THEN take the root.
    val_rmse = (
        val_physical_mse_sum / val_samples
    ).sqrt().cpu().tolist()

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

    # Append one row per epoch; closing the file flushes it to disk.
    with log_path.open("a", newline="") as f:
        csv.writer(f).writerow([
            epoch + 1, epoch_loss, val_loss, *val_rmse,
        ])

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save({
            "epoch": epoch + 1,
            "val_loss": val_loss,
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "ec": ec.state_dict(),
            "dc": dc.state_dict(),
            "preprocessor": preprocessor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "co2_mean": co2_mean,
            "co2_std": co2_std,
            "statics": statics,
        }, checkpoint_path)

        print(f"Saved best checkpoint: val MSE = {val_loss:.6f}")

    else:
        epochs_without_improvement += 1
        print(f"No improvement: {epochs_without_improvement}/{patience}")

        if epochs_without_improvement >= patience:
            print(f"Early stopping. Best val MSE = {best_val_loss:.6f}")
            break

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True,
)

encoder.load_state_dict(checkpoint["encoder"])
decoder.load_state_dict(checkpoint["decoder"])
ec.load_state_dict(checkpoint["ec"])
dc.load_state_dict(checkpoint["dc"])
preprocessor.load_state_dict(checkpoint["preprocessor"])

print(
    f"Restored epoch {checkpoint['epoch']} "
    f"(val MSE = {checkpoint['val_loss']:.6f})"
)

###### CO2 RESPONSE 
import numpy as np
import matplotlib.pyplot as plt

encoder.eval()
decoder.eval()
ec.eval()
dc.eval()

# Recover physical ppm from your standardized validation conditions.
co2_center = float(co2_mean.detach().cpu())
co2_scale = float(co2_std.detach().cpu())

validation_ppm = (
    val_conditions[:, 0].detach().cpu() * co2_scale + co2_center
)
reference_ppm = float(validation_ppm.median())

# Constant decoder CO₂ values. Adjust as desired.
sweep_ppm = np.array(sorted(set([
    280.0, 320.0, 360.0, 400.0, 560.0, 800.0, reference_ppm
])))
reference_index = int(np.argmin(abs(sweep_ppm - reference_ppm)))

n_variables = preprocessor.channels
sweep_sum = torch.zeros(
    len(sweep_ppm), n_variables, device=device
)
truth_sum = torch.zeros(n_variables, device=device)
reconstruction_sum = torch.zeros(n_variables, device=device)
n_samples = 0

# Weights sum to one over valid locations, separately per variable.
weights = preprocessor.loss_weights
valid = preprocessor.valid_mask_torch

def physical_global_mean(normalized):
    """[B, N, C] standardized fields -> [B, C] physical means."""
    physical = preprocessor.postprocess_torch(normalized)
    physical = torch.where(valid, physical, 0.0)
    return (physical * weights).sum(dim=1)

with torch.no_grad():
    for state, conditions in val_loader:
        state = state.to(device)
        conditions = conditions.to(device)

        # Encode ONCE using the actual conditions of each validation sample.
        latent = encoder(
            torch.cat([state, ec(conditions)], dim=-1),
            statics,
        )

        truth_sum += physical_global_mean(state).sum(dim=0)

        # Ordinary reconstruction at each sample's actual CO₂.
        actual_reconstruction = decoder(
            torch.cat([latent, dc(conditions)], dim=-1),
            statics,
        )
        reconstruction_sum += physical_global_mean(
            actual_reconstruction
        ).sum(dim=0)

        # Same latent, same month; only decoder CO₂ changes.
        for i, ppm in enumerate(sweep_ppm):
            changed_conditions = conditions.clone()
            changed_conditions[:, 0] = (
                float(ppm) - co2_center
            ) / co2_scale

            prediction = decoder(
                torch.cat(
                    [latent, dc(changed_conditions)], dim=-1
                ),
                statics,
            )
            sweep_sum[i] += physical_global_mean(
                prediction
            ).sum(dim=0)

        n_samples += state.shape[0]

truth_mean = (truth_sum / n_samples).cpu().numpy()
actual_mean = (reconstruction_sum / n_samples).cpu().numpy()
sweep_mean = (sweep_sum / n_samples).cpu().numpy()

# Response relative to decoding the SAME latents at reference_ppm.
response = sweep_mean - sweep_mean[reference_index]

# Genuine reconstruction bias: decoder uses actual sample CO₂.
actual_bias = actual_mean - truth_mean

# Counterfactual offset from the observed validation mean.
# This is NOT an error against known counterfactual truth.
reference_offset = sweep_mean - truth_mean[None, :]

train_ppm = (
    train_conditions[:, 0].detach().cpu() * co2_scale + co2_center
)
train_min, train_max = float(train_ppm.min()), float(train_ppm.max())

print(f"\nTraining CO₂ range: {train_min:.1f}–{train_max:.1f} ppm")
print(f"Constant-CO₂ response baseline: {reference_ppm:.1f} ppm")

for j, variable in enumerate(preprocessor.variables):
    print(f"\n{variable}: actual-CO₂ global-mean bias = {actual_bias[j]:+.5g}")
    print(" CO₂ ppm | global mean | Δ from baseline | offset from validation")
    for i, ppm in enumerate(sweep_ppm):
        print(
            f" {ppm:7.1f} | {sweep_mean[i, j]:11.5g} | "
            f"{response[i, j]:+15.5g} | {reference_offset[i, j]:+22.5g}"
        )

# Plot T2M. Change this name to inspect another variable.
variable = "t2m"
j = list(preprocessor.variables).index(variable)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(sweep_ppm, sweep_mean[:, j], "o-", label="Fixed validation latents")
axes[0].axhline(
    truth_mean[j], color="black", linestyle="--",
    label="Observed validation mean",
)
axes[0].set_ylabel(f"{variable}: global mean, physical units")
axes[0].legend()

axes[1].plot(sweep_ppm, response[:, j], "o-")
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_ylabel(f"{variable}: change from {reference_ppm:.1f} ppm")

for ax in axes:
    ax.axvspan(train_min, train_max, color="gray", alpha=0.15)
    ax.set_xlabel("Decoder CO₂ (ppm)")
    ax.grid(alpha=0.2)

fig.suptitle("Decoder CO₂ sensitivity — fixed validation latents and months")
fig.tight_layout()
plt.show()