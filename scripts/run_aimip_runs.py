import md as src 
print("VERSION: ", src.__version__)
import xarray as xr 
import numpy as np 
import torch 
import pandas as pd
import json 

torch.cuda.set_device(2)
mps_device = torch.device("cuda")

dataset = '1p5x1p5'
model_path = "MD-1p5.pth"

forcing_variables = ['SSTKSFC', "CISFC", "nanmask"]
group_levels = False

# load precalculated train period population statistics 
train_period = ("1985-01-01", "2014-12-01")
params_to_load = f"variable_statistics_{pd.Timestamp(train_period[0]).strftime('%Y%m')}-{pd.Timestamp(train_period[1]).strftime('%Y%m')}.json"
dict_of_params = json.load(open(params_to_load, 'r'))

data, statistics, mask, statics = src.open_era5_mini(
    start="1978-10-01", end=None, 
    mode='conv',
    dict_of_params = dict_of_params,
    group_levels= group_levels,
    return_statics = True,
    dataset = dataset 
)

aimip_forcings = xr.open_dataset(f'/glade/work/khall/ERA5/AIMIP-Data/aimip-forcings-flat-{dataset}.nc').sel(varlev=['SSTKSFC', "CISFC", "LSMSFC"])

_, aimip_forcing_params, _ = src.preprocess_by_variables(
    aimip_forcings.da.sel(
        time=slice(train_period[0], train_period[1])  # standardize according to training period 
    ),
    group_levels=group_levels
)

aimip_forcings_f, _, _ = src.preprocess_by_variables(
    aimip_forcings.da, # this is full dataset from 1978-10 
    dict_to_undo=aimip_forcing_params, 
    group_levels=group_levels
)

aimip_forcings_f = aimip_forcings_f.fillna(0).transpose('time', 'varlev', 'lat', 'lon')
data = data.transpose('time', 'varlev', 'lat', 'lon')

data = data.sel(varlev=[i for i in data.varlev.values if i not in forcing_variables ])

pvae = src.load_model(src.CVAE, model_path)

# aimip requests 5 members
ensemble_size = 5 
ic = torch.tensor(data.values, dtype=torch.float32)[0, ...].unsqueeze(0)
months = np.asarray([ pd.Timestamp(i).month for i in pd.date_range("1978-10-01", "2024-12-01", freq="MS") ]).reshape(-1,1)

template = xr.concat([xr.ones_like(data.isel(time=0)) for _ in range(ensemble_size) ], 'time')

print(ic.shape, aimip_forcings_f.values.shape, months.shape) 
print("FROM TRAIN: rch.Size([1, 42, 121, 240]) (876, 3, 121, 240) (876, 1)")

pvae.forced_run(
    ic,
    aimip_forcings_f.values,
    months,
    ensemble_size=ensemble_size,
    output_template=template,
    dict_of_params=statistics,
    mask=mask,
    output_files=f'observed_forcings-{model_path[:-4]}',
    use_noise=False,
    group_levels=group_levels
)

for uniform_increase in [2, 4]:
    print(f" Doing plus{uniform_increase}k uniform increase run")

    plus2k_sst = aimip_forcings.da.sel(varlev='SSTKSFC') + uniform_increase

    forcings = xr.concat(
        [plus2k_sst, aimip_forcings.da.sel(varlev=["CISFC", "LSMSFC"])],
        'varlev'
    )

    forcings = forcings.sel(varlev=['SSTKSFC', "CISFC", "LSMSFC"]).transpose('time', 'varlev', 'lat', 'lon')#.fillna(0)

    forcings, _, _ = src.preprocess_by_variables(
        forcings,
        dict_to_undo=aimip_forcing_params,
        group_levels=group_levels
    )

    forcings.to_netcdf(f"plus{uniform_increase}k_forcings.nc")

    pvae.forced_run(
        ic,
        forcings.fillna(0).values,
        months=months,
        ensemble_size=ensemble_size,
        output_template=template,
        dict_of_params=statistics,
        mask=mask,
        output_files=f"p{uniform_increase}k_forcings-{model_path[:-4]}",
        use_noise=False,
        group_levels=group_levels
    )

