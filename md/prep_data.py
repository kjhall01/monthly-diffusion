import torch 
import xarray as xr
import src 
import json 
from .preprocessing import preprocess_by_variables
from pathlib import Path 


def open_era5_mini(
        start=None, 
        end=None, 
        dict_of_params=None, 
        save_params_to=None,
        mask = None,
        save_mask_to=None,
        mode = 'flat',
        fillvalue=0,
        group_levels = True, 
        return_statics = False,
        dataset = '1x1',
        data_dir="/glade/work/khall/ERA5/AIMIP-Data/"
    ):
    
    """
    opens era5-mini and returns requested dates, scaled-by-variable with NaNs dropped in a torch.tensor of shape (N, M)
    where N is the number of dates and M is the number of Non-NaN features in era5. 
    * note that all of the variable names in ERA5 must adhere to the conventions set in preprocess_by_variables

    kwargs:
    start - pd.Timestamp for xr.DataArray.sel
    end - pd.Timestamp for xr.DataArray.sel 
    dict_of_params - dictionary full of statistics per variable produced by src.preprocess_by_variables
    save_params_to - string for with open(save_params_to, 'w') i.e. writing dict_of_params to file
    mask - xr.DataArray binary of shape (lat, lon) with zeros indicating NaN fields to be masked during preprocessing
    """

    # open ERA5 training data 

    ds = xr.open_dataset(Path(data_dir) / f"era5-flat-{dataset}.nc").da

    ds = ds.sel(time=slice(start, end)) 

    ds, dict_of_params, mask = preprocess_by_variables(
        ds,
        mask=mask,
        dict_to_undo=dict_of_params,
        group_levels= group_levels
    )

    if save_mask_to is not None: 
        mask.to_netcdf(save_mask_to)

    if save_params_to is not None:
        with open(save_params_to, "w") as f:
            json.dump(dict_of_params, f, indent=4) 

    if mode == 'flat':
        ds = ds.sortby('time')
        ds = ds.stack(feature=('lat', 'lon', 'varlev'))
        ds = ds.dropna('feature', how='any')
        ds = ds.transpose('time', 'feature')
    else: 
        ds = xr.concat([ds, mask.fillna(0).mean('varlev').expand_dims({'varlev':['nanmask']})], 'varlev')
        ds = ds.transpose('time', 'varlev', 'lat', 'lon')
        dict_of_params['nanmask'] = {}

    statics = None 
    if return_statics:
        statics = xr.open_dataset(Path(data_dir) / f"era5-statics-flat-{dataset}.nc")
        statics = (statics - statics.mean(['lat', 'lon'])) / statics.std(['lat', 'lon'])
        statics = statics.transpose('varlev', 'lat', 'lon')
        statics = statics.rename({'SDOR': 'da'}) # I forgot to rename the data arrays when I made the file - but they're all there.


    return ds.transpose('time', 'varlev', 'lat', 'lon').fillna(fillvalue), dict_of_params, mask, statics 