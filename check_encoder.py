"""From the repo root: python check_encoder.py (with monthly_diffusion installed)."""
from pathlib import Path

import torch
import xarray as xr
from torch.utils.data import DataLoader, TensorDataset

from monthly_diffusion.architecture import MeshStage, RefinedIcosahedralMeshEncoder

DATA = Path("~/Desktop/mdv1-e3smv2-r1i1p1f1-training-validation-1deg-1958-2006.nc").expanduser()
VARIABLES = [
    "sfcWind", "t2m", "tcwv", "mtpr", "mslp", "olr", "z500", "tau300_700",
    "t850", "t500", "u200", "v200", "u50", "ssrd", "tau_x", "tau_y",
]
BATCH_SIZE = 8
LATENT_CHANNELS = 256




stages = [
    MeshStage(name="m4", refinement=4, width=32, refiners=4),
    MeshStage(name="m3", refinement=3, width=64, refiners=6),
    MeshStage(name="m2", refinement=2, width=128, refiners=6),
]


# Select months first; keep the raw data lazy while fitting statistics.
ds = xr.open_dataset(DATA)

train =  ( 
    ds[VARIABLES].sel(time=slice("1966-01", "2006-12"))
    .to_array("variable")
    .transpose("time", "lat", "lon", "variable")
)

val =  ( 
    ds[VARIABLES].sel(time=slice("1958-01", "1965-12"))
    .to_array("variable")
    .transpose("time", "lat", "lon", "variable")
)


encoder = RefinedIcosahedralMeshEncoder(
    hierarchy=stages, train_da=train, latent_channels=LATENT_CHANNELS,
)

train_prepped = encoder.preprocess_torch( encoder.xr_to_torch(train))
val_prepped = encoder.preprocess_torch( encoder.xr_to_torch(val))


# CPU tensors for the loaders; only the current batch goes to the device.
train_loader = DataLoader(TensorDataset(train),
                            batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(val),
                        batch_size=BATCH_SIZE, shuffle=False)

if not (torch.isfinite(encoder.mean_torch).all()
        and torch.isfinite(encoder.std_torch).all()
        and (encoder.std_torch > 0).all()):
    raise ValueError("Every variable needs a finite training mean and positive SD.")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
encoder.to(device)

state = next(iter(train_loader))
latent = encoder(state)


