"""Train: python check_encoder.py. Evaluate: python check_encoder.py --evaluate RUN/best.pt."""
from pathlib import Path
import sys
from evaluate_encoder import load_models

# Evaluation is a separate path: never fit statistics, train, or select checkpoints
# using the extended archive. Keep the original training workflow below intact.
if __name__ == "__main__" and len(sys.argv) > 1:
    from evaluate_encoder import main
    main()
    raise SystemExit(0)

import atexit
import csv
from datetime import datetime




class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


def close_diagnostics():
    sys.stdout, sys.stderr = console_stdout, console_stderr
    diagnostics_file.close()




import torch
import xarray as xr
from torch.utils.data import DataLoader, TensorDataset
from time import perf_counter
from monthly_diffusion.architecture import MeshStage, RefinedIcosahedralMeshEncoder, Preprocessor, RefinedIcosahedralMeshDecoder , ScalarToLatentChannels

do_train = False
DATA = Path(
    "/Users/kyle/Dev/downloads/"
    "e3smv2_monthly_1deg_r1i1p1f1_185001-210012.nc"
)
VARIABLES = [
    "sfcWind", "t2m", "tcwv", "mtpr", "mslp", "olr", "z500", "tau300_700",
    "t850", "t500", "u200", "v200", "u50", "ssrd", "tau_x", "tau_y"
]
BATCH_SIZE = 8
LATENT_CHANNELS = 32

N_SCALARS = 2
CONDITION_CHANNELS = LATENT_CHANNELS
epochs = 500
lr = 1e-3

train_period = slice("1966-01", "2006-12")
val_period = slice("1958-01", "1965-12")
patience = 10
checkpoint_override = Path("/Users/kyle/Dev/monthly-diffusion/output/20261106_111306_426321/best.pt")

if do_train: 
    # Keep a console transcript alongside the metrics and checkpoint for each run.
    run_dir = Path("output") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_file = (run_dir / "diagnostics.txt").open("w", encoding="utf-8", buffering=1)
    console_stdout, console_stderr = sys.stdout, sys.stderr
    
    sys.stdout = Tee(console_stdout, diagnostics_file)
    sys.stderr = Tee(console_stderr, diagnostics_file)
    atexit.register(close_diagnostics)
    print(f"Saving results to {run_dir}")

    # Select months first; keep the raw data lazy while fitting statistics.
    ds = xr.open_dataset(DATA)
    train_da = ( 
        ds[VARIABLES]
        .sel(time=train_period)
        .to_array("variable")
    )


    val_da =  ( 
        ds[VARIABLES].sel(time=val_period)
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
        ds.sel(time=train_period).co2.values,
        dtype=torch.float32
    ).reshape(-1, 1)

    co2_mean = train_co2.mean() 
    co2_std = train_co2.std() 

    val_co2 = torch.tensor(
        ds.sel(time=val_period).co2.values,
        dtype=torch.float32
    ).reshape(-1, 1)

    train_co2 = (train_co2 - co2_mean) / co2_std 
    val_co2 = (val_co2 - co2_mean) / co2_std 


    train_conditions = torch.hstack([
    # train_co2,
        torch.cos(train_months / 12 * 2 * torch.pi), 
        torch.sin(train_months / 12 * 2 * torch.pi)
    ])

    val_conditions = torch.hstack([
    # val_co2,
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
        MeshStage(name="m4", refinement=4, width=128, refiners=4, map_to_next_stage="radius"),
        MeshStage(name="m3", refinement=3, width=128, refiners=4, map_to_next_stage="radius"),
        MeshStage(name="m2", refinement=2, width=128, refiners=4, map_to_next_stage="radius"),
    ]

    decoder_stages = [
        MeshStage(name="m2", refinement=2, width=128, refiners=4, map_to_next_stage="barycentric"),
        MeshStage(name="m3", refinement=3, width=128, refiners=4, map_to_next_stage="barycentric"),
        MeshStage(name="m4", refinement=4, width=128, refiners=4, map_to_next_stage="barycentric"),
    #    MeshStage(name="m5", refinement=5, width=128, refiners=2, map_to_next_stage="barycentric"),

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
        lr=lr,
    )

    
    training_start = perf_counter()

    log_path = run_dir / "loss.csv"
    checkpoint_path = run_dir / "best.pt"

    with log_path.open("w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_mse", "val_mse",
            *[f"val_rmse_{v}" for v in preprocessor.variables],
        ])

    best_val_loss = float("inf")
    epochs_without_improvement = 0


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
            scale_squared = preprocessor.std_torch.reshape(-1).square()
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
                "config": {
                    "variables": VARIABLES,
                    "encoder_stages": [vars(stage) for stage in encoder_stages],
                    "decoder_stages": [vars(stage) for stage in decoder_stages],
                    "latent_channels": LATENT_CHANNELS,
                    "condition_channels": CONDITION_CHANNELS,
                    "n_scalars": N_SCALARS,
                    "train_period": ["1966-01", "2006-12"],
                    "val_period": ["1958-01", "1965-12"],
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

if checkpoint_override is not None:
    checkpoint_path = checkpoint_override 

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True,
)

config = checkpoint["config"]
coords = xr.Dataset(coords={
    **{
        name: values.numpy()
        for name, values in checkpoint["coordinates"].items()
    },
    "variable": config["variables"],
}).coords

preprocessor = Preprocessor.from_state_dict(
    checkpoint["preprocessor"], coords
)

encoder, decoder, ec, dc = load_models(
    checkpoint, preprocessor, config, device
)
statics = checkpoint["statics"].to(device)

print(
    f"Restored epoch {checkpoint['epoch']} "
    f"(val MSE = {checkpoint['val_loss']:.6f})"
)

# Models and preprocessor have already been constructed and restored.
for module in (encoder, decoder, ec, dc, preprocessor):
    module.to(device).eval()

BATCH_SIZE = 24

output_dir = Path("reconstructed_batches")
output_dir.mkdir(parents=True, exist_ok=True)
with xr.open_dataset(DATA) as ds:
    times = ds.time.values.copy()

    with torch.no_grad():
        for start in range(0, ds.sizes["time"], BATCH_SIZE):
            batch = ds.isel(time=slice(start, start + BATCH_SIZE))
            batch_da = batch[VARIABLES].to_array("variable")
            print(batch_da.time)
            # Coordinate ordering is checked by xr_to_torch().
            state = preprocessor.xr_to_torch(batch_da, device=device)
            state = preprocessor.preprocess_torch(state)
            # New missing cells outside the training period must not propagate
            # through the network. Zero uses the saved training mean/background.
            state = torch.where(torch.isfinite(state), state, 0.0)

            months = torch.tensor(
                batch.time.dt.month.values,
                dtype=torch.float32,
                device=device,
            ).reshape(-1, 1)

            season = torch.cat([
                torch.cos(months / 12 * 2 * torch.pi),
                torch.sin(months / 12 * 2 * torch.pi),
            ], dim=-1)

            # Must match the conditioning used during training.
            if ec.in_proj.in_features == 3:
                co2 = torch.tensor(
                    batch.co2.values,
                    dtype=torch.float32,
                    device=device,
                ).reshape(-1, 1)

                co2 = (
                    co2 - float(checkpoint["co2_mean"])
                ) / float(checkpoint["co2_std"])

                conditions = torch.cat([co2, season], dim=-1)

            elif ec.in_proj.in_features == 2:
                conditions = season

            else:
                raise ValueError("Unexpected conditioning input size.")

            latent = encoder(
                torch.cat([state, ec(conditions)], dim=-1),
                statics,
            )

            reconstruction = decoder(
                torch.cat([latent, dc(conditions)], dim=-1),
                statics,
            )

            # Restore units, background, and invalid-location NaNs.
            if not torch.isfinite(reconstruction).all():
                raise ValueError(f"Non-finite reconstruction in batch starting at {start}; not saving")
            physical = preprocessor.postprocess_torch(reconstruction)
            
            # Inside the batch loop, after computing physical:
            recon_da = preprocessor.torch_to_xr(
                physical.cpu(),
                times=batch.time.values,
            )
            recon_ds = recon_da.to_dataset(dim="variable")

            recon_ds.to_netcdf(
                output_dir / f"reconstruction_{start:05d}.nc"
            )

reconstructed = xr.open_mfdataset(
    "reconstructed_batches/reconstruction_*.nc",
    combine="by_coords",
    chunks={"time": BATCH_SIZE},
)
