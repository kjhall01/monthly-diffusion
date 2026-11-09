import xarray as xr 
import torch 
from pathlib import Path 
from torch.utils.data import DataLoader
from time import perf_counter
import random

from monthly_diffusion.utilities import load_models
from monthly_diffusion.training import FiveMonthDataset , three_month_loss
from monthly_diffusion.model import CoupledDiffusion 


BATCH_SIZE = 16
MAX_EPOCHS = 500
PATIENCE = 10 

HOME_DIR = Path(__file__).resolve().parents[1] / "output/heptagon"
RUN_DIR = HOME_DIR / "latent_diffusion"
RUN_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# load these for the config, and for physical-space loss reporting 
atm_encoder, atm_dec, atm_ec, atm_dc, atm_preprocessor, config = load_models(HOME_DIR, 'atmosphere')
ocn_encoder, ocn_dec, ocn_ec, ocn_dc, ocn_preprocessor, config = load_models(HOME_DIR, 'ocean')

atmos_latents = xr.open_mfdataset(
    str(HOME_DIR / "atmosphere/ae_trajectory/latent*.nc"),
    combine="by_coords", chunks={'time': 24},
).latent

ocean_latents = xr.open_mfdataset(
    str(HOME_DIR / "ocean/ae_trajectory/latent*.nc"),
    combine="by_coords", chunks={'time': 24},
).latent

co2 = xr.open_dataset(
    "/Users/kyle/Dev/monthly-diffusion-data/data/"
    "co2-forcing-v1/products/co2_monthly_e3sm_ssp370_185001-210012.nc"
).co2

co2_train = co2.sel(time=slice(*config['train_period']))
co2_valid = co2.sel(time=slice(*config['VAL_PERIOD']))

train_months = torch.tensor(co2_train.time.month.values, dtype=torch.float32).reshape(-1, 1)
valid_months = torch.tensor(co2_valid.time.month.values, dtype=torch.float32).reshape(-1, 1)

co2_train = torch.tensor(co2_train.values, dtype=torch.float32).reshape(-1,1)
co2_valid = torch.tensor(co2_valid.values, dtype=torch.float32).reshape(-1,1)

atmos_train = atmos_latents.sel(time=slice(*config['train_period']))
atmos_valid = atmos_latents.sel(time=slice(*config['VAL_PERIOD'])) # in future this will need to be lowercase 

atmos_train_torch = torch.tensor(
    atmos_train.transpose('time', 'node', 'channel').values,
    dtype=torch.float32
)

atmos_valid_torch = torch.tensor(
    atmos_valid.transpose('time', 'node', 'channel').values,
    dtype=torch.float32
)

ocean_train = ocean_latents.sel(time=slice(*config['train_period']))
ocean_valid = ocean_latents.sel(time=slice(*config['VAL_PERIOD'])) # in future this will need to be lowercase 

ocean_train_torch = torch.tensor(
    ocean_train.transpose('time', 'node', 'channel').values,
    dtype=torch.float32
)

ocean_valid_torch = torch.tensor(
    ocean_valid.transpose('time', 'node', 'channel').values,
    dtype=torch.float32
)




train_loader = DataLoader(
    FiveMonthDataset(
        atmos_train_torch.to(device),
        ocean_train_torch.to(device),
        train_months.to(device),
        co2_train.to(device)
    ), 
    batch_size=BATCH_SIZE, shuffle=True
)

valid_loader = DataLoader(
    FiveMonthDataset(
        atmos_valid_torch.to(device),
        ocean_valid_torch.to(device),
        valid_months.to(device),
        co2_valid.to(device)
    ), 
    batch_size=BATCH_SIZE, shuffle=False
)

model_config = dict(
    atmos_latent_channels=32,
    ocean_latent_channels=18,
    n_atmos_nodes=162, #M2
    n_ocean_nodes=162, #M2
    width=128,
    depth=4,
    n_heads=4,
    ffn_ratio=4.0,
    time_embedding_channels=64,
    co2_dropout=0.1,
    atmosphere_tau=0.5,
    ocean_tau=6.0,
)
model = CoupledDiffusion(**model_config)

model.fit_preprocessing(atmos_train_torch, ocean_train_torch, co2_scale=280)
model = model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
)

modes = (
    "fully_coupled",
    "prescribed_ocean",
    "prescribed_atmosphere",
)

for name, network in (
    ("Atmosphere", model.atmosphere),
    ("Ocean", model.ocean),
):
    count = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"{name} trainable parameters: {count:,}")

total = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total:,}")

best_val_loss = float("inf")
epochs_without_improvement = 0
for epoch in range(MAX_EPOCHS):
    started = perf_counter()
    model.train()

    train_sum = {mode: 0.0 for mode in modes}
    train_count = {mode: 0 for mode in modes}

    for batch in train_loader:
        batch = {
            name: tensor.to(device)
            for name, tensor in batch.items()
        }

        # One mode per batch; balanced in expectation.
        mode = random.choice(modes)

        optimizer.zero_grad(set_to_none=True)

        loss = three_month_loss(model, batch, mode)
        loss.backward()

        # Limit unusually large updates through the unrolled sampler.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = batch["atmosphere"].shape[0]
        train_sum[mode] += loss.detach().item() * batch_size
        train_count[mode] += batch_size

    # Evaluate all three tasks separately, with explicit CO2 available.
    model.eval()
    val_sum = {mode: 0.0 for mode in modes}
    val_count = 0

    with torch.no_grad():
        for batch in valid_loader:
            batch = {
                name: tensor.to(device)
                for name, tensor in batch.items()
            }
            batch_size = batch["atmosphere"].shape[0]

            for mode in modes:
                loss = three_month_loss(model, batch, mode)
                val_sum[mode] += loss.item() * batch_size

            val_count += batch_size

    print(
        f"Epoch {epoch + 1:03d}/{MAX_EPOCHS} "
        f"| {perf_counter() - started:.1f}s"
    )

    for mode in modes:
        train_mean = (
            train_sum[mode] / train_count[mode]
            if train_count[mode] else float("nan")
        )
        val_mean = val_sum[mode] / val_count

        print(
            f"  {mode:23s} "
            f"| train {train_mean:.6f} "
            f"| val {val_mean:.6f}"
        )

    val_loss = val_sum["fully_coupled"] / val_count
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save({
            "config": model_config,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "val_loss": val_loss,
            "validation_mode": "fully_coupled",
            "train_period": config["train_period"],
            "val_period": config["VAL_PERIOD"],
        }, RUN_DIR / "best.pt")
        print(f"Saved best checkpoint: fully-coupled val loss = {val_loss:.6f}")

    else:
        epochs_without_improvement += 1
        print(f"No improvement: {epochs_without_improvement}/{PATIENCE}")
        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping. Best fully-coupled val loss = {best_val_loss:.6f}")
            break


