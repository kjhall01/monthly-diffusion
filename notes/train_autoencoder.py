import xarray as xr 
import torch 

from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path 


import monthly_diffusion as md 

# folder where things will land 
output = Path("output")
output.mkdir(exist_ok=True, parents=True)

# netcdf4 file containing training dataset
data_path = Path("~/Desktop/mdv1-e3smv2-r1i1p1f1-training-validation-1deg-1958-2006.nc") 

# set train/val/test split - here no test
train_period = slice("1966-01-01", "2006-12-01")
val_period = slice("1958-01-01", "1965-12-01")

# select variables 
prognostic_variables = [ 
    'sfcWind', 't2m', 'tcwv', 
    'mtpr', 'mslp', 'olr', 
    'z500', 'tau300_700', 't850', 
    't500', 'u200', 'v200', 
    'u50', 'ssrd', 'tau_x', 
    'tau_y', 
]

# ocean_variables = [
#   'sst', 'ssh', 'sss', 
#   'mld', 'sic', 'sivol', 
#   'd20', 'ohc300', 'ohc700_minus_ohc300'
# ]

#prognostic_variables = ['sst', 't2m', 'mtpr', 'ssh']
static_variables = ['lsm', 'phi_s']


train_ds = xr.open_dataset(data_path).sel(time=train_period)

# make cosine-latitude area-weights 
latitude_weights = torch.cos(
    torch.deg2rad(
        torch.tensor(train_ds.lat.values.squeeze())
    )
)
latitude_weights = latitude_weights / latitude_weights.sum() 
latitude_weights = latitude_weights[None, :, None, None]

##### TRAINING DATA 
# train_data will be of shape (B, H, W, C)
train_data = train_ds[prognostic_variables].to_array('variable')
train_data = train_data.transpose("time", "lat", "lon", "variable")
train_data = torch.tensor(train_data.values)


# co2_train will be of shape (B, 1)
co2_train = torch.tensor(
    train_ds.co2.values
)

# statics will be of shape (H, W, C)
statics = torch.tensor(
    train_ds[static_variables].to_array('variable')
    .transpose('lat', 'lon', 'variable').values
)

train_months = torch.tensor(train_ds.time.month.values, dtype=torch.float)


##### VALIDATION DATA 
val_ds = xr.open_dataset(data_path).sel(time=val_period)

# train_data will be of shape (B, H, W, C)
val_data = val_ds[prognostic_variables].to_array('variable')
val_data = val_data.transpose("time", "lat", "lon", "variable")
val_data = torch.tensor(val_data.values)

# co2_val will be of shape (B, 1)
co2_val = torch.tensor(
    val_ds.co2.values
)

val_months = torch.tensor(val_ds.time.month.values, dtype=torch.float)

print(f"Train Data: {train_data.shape}")
print(f"Train CO2: {co2_train.shape}")
print(f"Train Months: {train_months.shape}")
print()
print(f"Statics: {statics.shape}")
print()
print(f"Val Data: {val_data.shape}")
print(f"Val CO2: {co2_val.shape}")
print(f"Val Months: {val_months.shape}")


BATCH_SIZE=8
train_loader = DataLoader(
    TensorDataset(train_data_prepped, train_months, co2_train),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

valid_loader = DataLoader(
    TensorDataset(val_data, val_months, co2_val),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)


e = (
    len(prognostic_variables), #channels
    kernel_hidden_dim = 128, 
    down_maps = [grid_to_m4, m4_to_m3, m3_to_m2],
    refiner_maps = [m4_to_m4, m3_to_m3, m2_to_m2], 
    n_refiners = [4,6,6],
    widths = [32, 64, 128],
    latent_channels=256
)


state, months, co2 = next(iter(train_loader))

with torch.no_grad():
    latent = e(state)

print(latent.shape)
# Expected: torch.Size([16, 162, 256])