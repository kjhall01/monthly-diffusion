import md as src 
print("VERSION: ", src.__version__)
import xarray as xr 
import numpy as np 
import torch 
import pandas as pd
import torch.nn as nn 
from pathlib import Path

torch.cuda.set_device(1)
mps_device = torch.device("cuda")

train_period = ("1985-01-01", "2014-12-01")
val_period =  ("1979-01-01", "1984-12-01")
seasonality_dim=3
forcing_variables = ['SSTKSFC', "CISFC", "nanmask"]
group_levels = False
dataset = '1p5x1p5'
training = True
model_path = "MD-1p5.pth"
data_dir = "/glade/work/khall/ERA5/AIMIP-Data/"


n_epochs = 103
batch_size = 4
learning_rate = 1e-3

train, statistics, mask, statics = src.open_era5_mini(
    start=train_period[0], end=train_period[1], 
    save_params_to=f"variable_statistics_{pd.Timestamp(train_period[0]).strftime('%Y%m')}-{pd.Timestamp(train_period[1]).strftime('%Y%m')}.json",
    mask=None,
    mode='conv',
    group_levels= group_levels,
    return_statics = True,
    dataset = dataset,
    data_dir = data_dir
)


train_forcing = train.sel(varlev=forcing_variables)
aimip_forcings = xr.open_dataset(Path(data_dir) / f'aimip-forcings-flat-{dataset}.nc')
aimip_forcings_prepped, dct, msk = src.preprocess_by_variables(aimip_forcings.da)

# need to do this to fix the mask calculations - they are messed up in preprocessing 
train_nanmask = train_forcing.sel(varlev='nanmask').where(train_forcing.sel(varlev='nanmask') == 1, other=0)
train_forcing = xr.concat(
    [
        train_forcing.sel(varlev=['SSTKSFC', 'CISFC']), 
        ((1 - train_nanmask) * aimip_forcings_prepped.sel(varlev='LSMSFC', drop=True).mean('time')).expand_dims('varlev') ],
    'varlev'
)


train = train.sel(varlev=[i for i in train.varlev.values if i not in forcing_variables ])

encoding_variables = [_ for _ in train.varlev.values ]
static_variables = [ _ for _ in statics.varlev.values ]
forcing_variables2 = forcing_variables + [f'time{_}SFC' for _ in range(seasonality_dim) ]

all_variable_names = []
all_variable_names.extend(encoding_variables)
all_variable_names.extend(forcing_variables2)
all_variable_names.extend(static_variables)
all_variable_names[all_variable_names.index('nanmask')] = "nanmaskSFC"

print(all_variable_names)
print(statics)
statics = torch.tensor(statics.da.values).to(mps_device)

val, statistics, mask, _ = src.open_era5_mini(
    start=val_period[0], end=val_period[1],
    save_params_to=None, dict_of_params=statistics,
    mask = mask,
    mode='conv',
    group_levels= group_levels,
    dataset = dataset ,
    data_dir = data_dir

)

val_forcing = val.sel(varlev = forcing_variables)
aimip_forcings = xr.open_dataset(Path(data_dir) / f'aimip-forcings-flat-{dataset}.nc')
aimip_forcings_prepped, dct, msk = src.preprocess_by_variables(aimip_forcings.da)

val_nanmask = val_forcing.sel(varlev='nanmask').where(val_forcing.sel(varlev='nanmask') == 1, other=0)
val_forcing = xr.concat(
    [ 
        val_forcing.sel(varlev=['SSTKSFC', 'CISFC']), 
        ((1 - val_nanmask) * aimip_forcings_prepped.sel(varlev='LSMSFC', drop=True).mean('time')).expand_dims('varlev') ],
    'varlev'
)

val = val.sel(varlev=[i for i in val.varlev.values  if i not in forcing_variables])


train_months = np.asarray([i.month for i in pd.date_range(train_period[0], train_period[1], freq="MS") ])
val_months = np.asarray([i.month for i in pd.date_range(val_period[0], val_period[1], freq="MS") ])
train_months_t0, train_months_t1 = train_months[:-1].reshape(-1,1), train_months[1:].reshape(-1,1) 
val_months_t0, val_months_t1 = val_months[:-1].reshape(-1,1), val_months[1:].reshape(-1,1) 

train_t0, train_t1 = train.values[:-1, ...], train.values[1:, ...]
val_t0, val_t1 = val.values[:-1, ...], val.values[1:, ...]

train_forcing_t0, train_forcing_t1 = train_forcing.values[:-1, :, :, :], train_forcing.values[1:, :, :, :]
val_forcing_t0, val_forcing_t1 = val_forcing.values[:-1, :, : ,:], val_forcing.values[1:, :, :, :]

training_data = {
    'x': train_t0, 
    'y': train_t1, 
    'x_forcing': train_forcing_t0, 
    'y_forcing': train_forcing_t1,
    'x_months':  train_months_t0,
    'y_months': train_months_t1
}

val_data = {
    'x': val_t0, 
    'y': val_t1, 
    'x_forcing': val_forcing_t0, 
    'y_forcing': val_forcing_t1,
    'x_months':  val_months_t0,
    'y_months': val_months_t1
}


pvae = src.CVAE(
    forcing_dim = train_forcing_t0.shape[1], 
    data_dimensions = train_t0[0,...].shape,
    activation=nn.GELU, 
    device=mps_device,
    seasonality_dim = seasonality_dim,
    seasonality_basis=seasonality_dim,
    hidden_channels=[ 32 ],
    squeeze_factors = [ 3],
    kl_weight = 5e-3,
    recon_weight = 1,
    prediction_weight = 0.5,
    n_statics = statics.shape[0],
    statics= statics,
    variable_names= all_variable_names,

    encoder_sfno_embed_dim = 128,
    encoder_rank = 64,
    encoder_conditioning_rank = 2,
    encoder_conditioning_operator_type = 'spectral_fc',

    decoder_sfno_embed_dim = 128,
    decoder_rank = 256,
    decoder_conditioning_rank = 4,
    decoder_conditioning_operator_type = 'spectral_fc',

    diffusion_sfno_embed_dim = 128,
    diffusion_rank = 128, 
    diffusion_conditioning_rank = 2,
    diffusion_conditioning_operator_type= 'spectral_fc'
)

best_overall, best_recon, best_pred = pvae.train_model(
    training_data = training_data,
    validation_data=val_data,
    num_epochs=n_epochs,
    batch_size=batch_size, 
    lr=learning_rate
)
pvae = src.load_model(src.CVAE, best_overall)
src.save_model(pvae, model_path)



