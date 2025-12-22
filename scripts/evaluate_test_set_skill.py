import md as src 
print("VERSION: ", src.__version__)
import xarray as xr 
import numpy as np 
import torch 
from torch import nn 
import pandas as pd 
from torch.utils.data import DataLoader, TensorDataset
from pandas.tseries.offsets import DateOffset 
import json 

train_period = ("1985-01-01", "2014-12-01")
val_period =  ("1979-01-01", "1984-12-01")
test_period = ("2015-01-01", None)
forcing_variables = ['SSTKSFC', "CISFC", "nanmask"]
group_levels = False
dataset = "1p5x1p5"


model_path = "MD-1p5.pth"
mps_device = torch.device("cuda")
model = src.load_model(src.CVAE, model_path)
model.to(mps_device)

# load precalculated train period population statistics 
train_period = ("1985-01-01", "2014-12-01")
params_to_load = f"variable_statistics_{pd.Timestamp(train_period[0]).strftime('%Y%m')}-{pd.Timestamp(train_period[1]).strftime('%Y%m')}.json"
dict_of_params = json.load(open(params_to_load, 'r'))


test, statistics, mask, _ = src.open_era5_mini(
    start=test_period[0], end=test_period[1],
    save_params_to=None, dict_of_params=dict_of_params,
    mask = None,
    mode='conv' ,
    dataset=dataset,
    group_levels = group_levels
)

test_forcing = test.sel(varlev=forcing_variables)
aimip_forcings = xr.open_dataset(f'/glade/work/khall/ERA5/AIMIP-Data/aimip-forcings-flat-{dataset}.nc')
aimip_forcings_prepped, dct, msk = src.preprocess_by_variables(aimip_forcings.da)

test_nanmask = test_forcing.sel(varlev='nanmask').where(test_forcing.sel(varlev='nanmask') == 1, other=0)
test_forcing = xr.concat(
    [ 
        test_forcing.sel(varlev=['SSTKSFC', 'CISFC']), 
        ((1 - test_nanmask) * aimip_forcings_prepped.sel(varlev='LSMSFC', drop=True).mean('time')).expand_dims('varlev') ],
    'varlev'
)




test = test.sel(varlev=[i for i in test.varlev.values  if i not in forcing_variables])
print(test.time.min(), test.time.max())

test_months = np.asarray([ pd.Timestamp(i).month for i in test.time.values ]).reshape(-1,1)
test_t0, test_t1 = test.values[:, ...], test.values[1:, ...]

test_forcing_t0, test_forcing_t1 = test_forcing.values[:, :], test_forcing.values[1:, :]


bs = 16 
test_dataset = TensorDataset(
    torch.tensor(test_forcing_t0, dtype=torch.float32).to(mps_device), 
    torch.tensor(test_t0, dtype=torch.float32).to(mps_device),
    torch.tensor(test_months, dtype=torch.float32).to(mps_device)
)
test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False)



ensemble_size = 5

sz = 0
result = [] 
for f_t0, x_t0, m_t0 in test_dataloader:
    template = xr.concat([xr.ones_like(test.isel(time=slice(None, x_t0.shape[0]))) for _ in range(ensemble_size) ], 'member')
    print([pd.Timestamp(template.time.values[_]) + DateOffset(months=sz) for _ in range(x_t0.shape[0])])
    print(f_t0.shape)

    m_t0 = model.seasonality_embedding(m_t0) 
    f_t0 = torch.cat([f_t0, m_t0], dim=1) 

    statics = model.statics

    mu_t0, lv_t0, cs = model.encoder(x_t0, f_t0)
    tc = []

    dec = model.decoder(mu_t0, cs).unsqueeze(0)
    tc.append( dec )

    for _ in range(ensemble_size-1):
        montecarlo_sample = torch.randn_like(lv_t0).to(lv_t0.device)
        z_t0 = model.reparametrize(mu_t0, lv_t0, eps=montecarlo_sample)
        dec = model.decoder(z_t0,  cs).unsqueeze(0)
        tc.append( dec ) 

    decoded_physical_states = torch.vstack(tc).cpu().detach().numpy()

    decoded_physical_states = template.copy(data=decoded_physical_states)
    decoded_physical_states = decoded_physical_states.assign_coords({'member': np.arange(ensemble_size) + 1})
    decoded_physical_states = decoded_physical_states.assign_coords({'time': [pd.Timestamp(template.time.values[_]) + DateOffset(months=sz) for _ in range(x_t0.shape[0])]})

    decoded_physical_states = decoded_physical_states * mask
    decoded_physical_states = src.undo_preprocess_by_variables(
        decoded_physical_states, 
        dict_to_undo=statistics,
        group_levels = group_levels
    )                

    result.append(decoded_physical_states)
    sz += x_t0.shape[0]

result = xr.concat(result, 'time')
reference = xr.open_dataset(f"/glade/work/khall/ERA5/AIMIP-Data/era5-flat-{dataset}.nc").da.sel(time=slice(test_period[0], test_period[1]))

mu = result.sel(member=1).drop_vars('member') 

# squared error, mean in time -> mse over space for each varlev
mse = ((mu - reference) ** 2).mean(dim='time')  # dims: lat, lon, varlev
spread = ((mu - result.isel(member=slice(1, None)) )**2).mean(dim='member').mean(dim='time')  # dims: lat, lon, varlev

# cosine-latitude weights (broadcast to lat,lon)
lat = mse['lat']
weights_lat = np.cos(np.deg2rad(lat))
weights = xr.DataArray(weights_lat, coords={'lat': lat}, dims=['lat'])
weighted_mse = mse.weighted(weights).mean(['lat', 'lon'])
weighted_spread = spread.weighted(weights).mean(['lat', 'lon']) 

rmse = np.sqrt(weighted_mse)  # dims: varlev
rspread = np.sqrt(weighted_spread)  # dims: varlev

# pretty table
table = rmse.to_series().to_frame(name='rmse (mu)').sort_index()

# compute cosine-latitude weighted mean of reference (average in time, then spatially)
mean_ref = np.sqrt(reference**2).mean(dim='time')  # dims: lat, lon, varlev
lat = mean_ref['lat']
weights_lat = np.cos(np.deg2rad(lat))
weights_da = xr.DataArray(weights_lat, coords={'lat': lat}, dims=['lat'])
weighted_ref_mean = mean_ref.weighted(weights_da).mean(['lat', 'lon'])  # dims: varlev

# temporal variance, then cosine-latitude weighted spatial mean per varlev
ref_var = reference.std(dim='time')
weighted_ref_var = ref_var.weighted(weights_da).mean(['lat', 'lon'])
table['obs_std'] = weighted_ref_var.to_series().reindex(table.index)

# add as column to table, aligning index
table['obs_mean'] = weighted_ref_mean.to_series().reindex(table.index)
table['rspread (ens)'] = rspread.to_series().reindex(table.index)
def _fmt(x):
    try:
        return f"{x:0.8f}"
    except (ValueError, TypeError):
        return x

print(table.applymap(_fmt))