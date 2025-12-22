import md as src 
print("VERSION: ", src.__version__)
import xarray as xr 
import numpy as np 
import torch 
import pandas as pd
import torch.nn as nn 
from pathlib import Path

#torch.cuda.set_device(0)
mps_device = torch.device("mps")

train_period = ("1985-01-01", "2014-12-01")
val_period =  ("1979-01-01", "1984-12-01")
seasonality_dim=3
forcing_variables = ['SSTKSFC', "CISFC", "nanmask"]
group_levels = False
dataset = '5p0x5p0'
training = True
model_path = "MD-5p0.pth"
data_dir = "~/Desktop/aimip-data/"

observed_forcing_output_folder = f'observed_forcings-{model_path.split(".")[0]}'
p2k_forcing_output_folder = f"p2k_forcings-{model_path.split(".")[0]}"
p4k_forcing_output_folder = f"p4k_forcings-{model_path.split(".")[0]}"


train, statistics, mask, statics = src.open_era5_mini(
    start=train_period[0], end=train_period[1], 
    save_params_to="variable_statistics_197901-201412.json",
    mask=None,
    mode='conv',
    group_levels= group_levels,
    return_statics = True,
    dataset = dataset ,
    data_dir = data_dir
)


train_forcing = train.sel(varlev=forcing_variables)
aimip_forcings = xr.open_dataset(Path(data_dir) / f'aimip-forcings-flat-{dataset}.nc')
aimip_forcings_prepped, dct, msk = src.preprocess_by_variables(aimip_forcings.da)

train_nanmask = train_forcing.sel(varlev='nanmask').where(train_forcing.sel(varlev='nanmask') == 1, other=0)
train_forcing = xr.concat(
    [ 
        train_forcing.sel(varlev=['SSTKSFC', 'CISFC']), 
        ((1 - train_nanmask) * aimip_forcings_prepped.sel(varlev='LSMSFC', drop=True).mean('time')).expand_dims('varlev') ],
    'varlev'
)



print('training forcing means', train_forcing.mean(['lat', 'lon', 'time']).values)
print('training forcing stds', train_forcing.std(['lat', 'lon', 'time']).values)



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

print('validation forcing means', val_forcing.mean(['lat', 'lon', 'time']).values)
print('validation forcing stds', val_forcing.std(['lat', 'lon', 'time']).values)


train_months = np.asarray([i.month for i in pd.date_range(train_period[0], train_period[1], freq="MS") ])
val_months = np.asarray([i.month for i in pd.date_range(val_period[0], val_period[1], freq="MS") ])
train_months_t0, train_months_t1 = train_months[:-1].reshape(-1,1), train_months[1:].reshape(-1,1) 
val_months_t0, val_months_t1 = val_months[:-1].reshape(-1,1), val_months[1:].reshape(-1,1) 

train_t0, train_t1 = train.values[:-1, ...], train.values[1:, ...]
val_t0, val_t1 = val.values[:-1, ...], val.values[1:, ...]

      
train_forcing_t0, train_forcing_t1 = train_forcing.values[:-1, :, :, :], train_forcing.values[1:, :, :, :]
val_forcing_t0, val_forcing_t1 = val_forcing.values[:-1, :, : ,:], val_forcing.values[1:, :, :, :]

print("HELLO", train_forcing_t0.shape, train_t0.shape, statics.shape)

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

latent_dim = 16
kl_weight = 1e-7
n2n_weight = 1e-3
pr_weight = 1e-3

ne=200
bs=4
lr=1e-3

if training:
    pvae = src.CVAE(
        forcing_dim = train_forcing_t0.shape[1], 
        data_dimensions = train_t0[0,...].shape,
        activation=nn.GELU, 
        device=mps_device,
        seasonality_dim = seasonality_dim,
        seasonality_basis=seasonality_dim,
        hidden_channels=[ 32 ],
        squeeze_factors = [ 2],
        kl_weight = 5e-3,
        recon_weight = 1,
        prediction_weight = 0.5,
        n_statics = statics.shape[0],
        statics= statics,
        variable_names= all_variable_names,

        encoder_sfno_embed_dim = 256,
        encoder_rank = 256,
        encoder_conditioning_rank = 2,
        encoder_conditioning_operator_type = 'spectral_fc',
    
        decoder_sfno_embed_dim = 256,
        decoder_rank = 256,
        decoder_conditioning_rank = 2,
        decoder_conditioning_operator_type = 'spectral_fc',

        diffusion_sfno_embed_dim = 128,
        diffusion_rank = 128, 
        diffusion_conditioning_rank = 2,
        diffusion_conditioning_operator_type= 'spectral_fc'
    )

    best_overall, best_recon, best_pred = pvae.train_model(
        training_data = training_data,
        validation_data=val_data,
        num_epochs=ne,
        batch_size=bs, 
        lr=lr
    )
    pvae = src.load_model(src.CVAE, best_overall)
    src.save_model(pvae, model_path)
else:
    pvae = src.load_model(src.CVAE, model_path) 
    #src.save_model(pvae, model_path)



import sys; sys.exit() 
ensemble_size=10
ic = torch.tensor(val_t0, dtype=torch.float32)[0,...].unsqueeze(0)

forcings, statistics, mask, _ = src.open_era5_mini(
    start=None, end=None,
    save_params_to=None, dict_of_params=statistics,
    mask = mask,
    mode='conv',
    group_levels= group_levels,
    dataset = dataset 


)

forcings = forcings.sel(varlev=forcing_variables)
forcings.to_netcdf(f"{observed_forcing_output_folder}.nc")

months = np.asarray([ pd.Timestamp(i).month for i in forcings.time.values ]).reshape(-1,1)


print('observed forcing means', forcings.mean(['lat', 'lon', 'time']).values)
print('observed forcing stds', forcings.std(['lat', 'lon', 'time']).values)
forcings = forcings.values

template = xr.concat([xr.ones_like(val.isel(time=0)) for _ in range(ensemble_size) ], 'time')

if True:

    ic = torch.tensor(val.isel(time=0).values, dtype=torch.float32).unsqueeze(0)


    print(ic.shape, forcings.shape, months.shape)
    pvae.forced_run(
        ic,
        forcings,
        months,
        ensemble_size=ensemble_size,
        output_template=template,
        dict_of_params=statistics,
        mask=mask,
        output_files=f"{observed_forcing_output_folder}2",
        use_noise=False,
        group_levels=group_levels
    )

    forcings, _, _, _= src.open_era5_mini(
        start=None, end=None,
        save_params_to=None, dict_of_params=statistics,
        mask = mask,
        mode='conv',
        group_levels= group_levels,
        dataset = dataset 

    )
    forcings = forcings.sel(varlev=forcing_variables)

    forcings = src.undo_preprocess_by_variables(
        forcings, 
        dict_to_undo=statistics,
        group_levels=group_levels
    )

    plus2k = forcings.sel(varlev='SSTKSFC') + forcings.time.copy(data=2 * torch.linspace(0, 1, forcings.time.shape[0]))
    forcings = xr.concat([plus2k, forcings.sel(varlev=[ "CISFC", "nanmask"])], 'varlev')
    forcings = forcings.sel(varlev=forcing_variables).transpose('time', 'varlev', 'lat', 'lon')

    forcings, _, _ = src.preprocess_by_variables(
        forcings,
        dict_to_undo=statistics,
        group_levels=group_levels
    )

    forcings.to_netcdf(f"{p2k_forcing_output_folder}.nc")

    pvae.forced_run(
        ic,
        forcings.values,
        months = months,
        ensemble_size=ensemble_size,
        output_template=template,
        dict_of_params=statistics,
        mask=mask,
        output_files=f"{p2k_forcing_output_folder}2",
        use_noise=False,
        group_levels=group_levels
    )


