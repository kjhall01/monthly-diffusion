import xarray as xr 
from scipy.special import logit, expit
import numpy as np 
from collections import defaultdict

def preprocess_continuous_variable(da, mu=None, sigma=None):
    """
    preprocess functions for variables will always return:
    1. the preprocessed data
    2. a dictionary containing the parameters needed by the 'reverse_preprocess' function for that variable 
    """
    
    mu = float(da.mean().values) if mu is None else mu
    sigma = float(da.std().values) if sigma is None else sigma
    return (da - mu) / sigma, {'mu': mu, 'sigma': sigma} 

def undo_preprocess_continuous_variable(da, mu=None, sigma=None):
    return da * sigma + mu 

def preprocess_strictly_positive_variable(da, mu=None, sigma=None):
    """
    i.e., for q and precip we will apply a sqrt and then do the standardization
    """
    da = np.sqrt(da) 
    mu = float(da.mean().values) if mu is None else mu
    sigma = float(da.std().values) if sigma is None else sigma
    return (da - mu) / sigma, {'mu': mu, 'sigma': sigma} 

def undo_preprocess_strictly_positive_variable(da, mu=None, sigma=None):
    return ( da * sigma + mu ) **2 

def preprocess_bounded_variable(da, epsilon=1e-3, mu=None, sigma=None):
    """
    ie., for sea ice we will clip values to (epsilon, 1-epsilon) and then apply a logit transformation 
    and then a normalization
    """
    #clamp values bc infinity is bad
    da = da.where(da > epsilon, other=epsilon) 
    da = da.where(da < (1-epsilon), other=(1-epsilon)) 
    da = logit(da) 
    mu = float(da.mean().values) if mu is None else mu
    sigma = float(da.std().values) if sigma is None else sigma
    return (da - mu) / sigma, {'mu': mu, 'sigma': sigma} 

def undo_preprocess_bounded_variable(da, mu=None, sigma=None):
    return expit(da*sigma + mu)

# detect distinct variables 
def group_varlevs_by_variable(varlevs):
    grouped = defaultdict(list)
    for v in varlevs:
        if v.endswith('SFC'):
            key = v.split('SFC')[0].replace('VAR_', '').lower()
        else:
            key = ''.join([c for c in v if not c.isdigit() and c != '.']).lower()
        grouped[key].append(v)
    return dict(grouped)

# detect distinct variables 
def group_varlevs_by_variable_individual(varlevs):
    return { k: [k] for k in varlevs}

def preprocess_by_variables(
    ds, 
    dict_to_undo = None,
    mask = None,
    group_levels = True
):
    """
        designed for era5-flat.zarr, which has all the variables and levels pooled in a single dimension
        returns preprocessed data, dict for undoing preprocessing 
    """

    how_to_preprocess = {
        'CISFC': preprocess_bounded_variable,
        'MTPRSFC' :preprocess_strictly_positive_variable,
        'SKTSFC'  :preprocess_continuous_variable,
        'SPSFC'  :preprocess_continuous_variable,
        'MSLSFC': preprocess_continuous_variable,
        'SSTKSFC'  :preprocess_continuous_variable,
        'VAR_10USFC'  :preprocess_continuous_variable,
        'VAR_10VSFC' :preprocess_continuous_variable,
        'VAR_2DSFC' :preprocess_continuous_variable,
        'VAR_2TSFC'  :preprocess_continuous_variable,
        'q100.0' :preprocess_strictly_positive_variable,
        'q1000.0' :preprocess_strictly_positive_variable,
        'q250.0' :preprocess_strictly_positive_variable,
        'q50.0' :preprocess_strictly_positive_variable,
        'q500.0':preprocess_strictly_positive_variable,
        'q700.0' :preprocess_strictly_positive_variable,
        'q850.0' :preprocess_strictly_positive_variable,
        't100.0'  :preprocess_continuous_variable,
        't1000.0'  :preprocess_continuous_variable,
        't250.0'  :preprocess_continuous_variable,
        't50.0'  :preprocess_continuous_variable,
        't500.0'  :preprocess_continuous_variable,
        't700.0' :preprocess_continuous_variable,
        't850.0'  :preprocess_continuous_variable,
        'u100.0'  :preprocess_continuous_variable,
        'u1000.0' :preprocess_continuous_variable,
        'u250.0' :preprocess_continuous_variable,
        'u50.0' :preprocess_continuous_variable,
        'u500.0' :preprocess_continuous_variable,
        'u700.0' :preprocess_continuous_variable,
        'u850.0': preprocess_continuous_variable,
        'v100.0' :preprocess_continuous_variable,
        'v1000.0' :preprocess_continuous_variable,
        'v250.0' :preprocess_continuous_variable,
        'v50.0' : preprocess_continuous_variable,
        'v500.0' : preprocess_continuous_variable,
        'v700.0': preprocess_continuous_variable,
        'v850.0': preprocess_continuous_variable,

        'z' :preprocess_continuous_variable,

        'z100.0' :preprocess_continuous_variable,
        'z1000.0' :preprocess_continuous_variable,
        'z250.0' :preprocess_continuous_variable,
        'z50.0' : preprocess_continuous_variable,
        'z500.0' : preprocess_continuous_variable,
        'z700.0': preprocess_continuous_variable,
        'z850.0': preprocess_continuous_variable,
        
        'q': preprocess_strictly_positive_variable,
        't': preprocess_continuous_variable,
        'u': preprocess_continuous_variable, 
        'v': preprocess_continuous_variable, 
        '10u': preprocess_continuous_variable, 
        '10v': preprocess_continuous_variable, 
        '2d': preprocess_continuous_variable, 
        '2t': preprocess_continuous_variable, 
        'mtpr': preprocess_strictly_positive_variable,
        'skt': preprocess_continuous_variable, 
        'sp': preprocess_continuous_variable, 
        'sstk': preprocess_continuous_variable, 
        'nanmask': lambda x: (x, {}),
        'nanmaskSFC': lambda x: (x, {}),
        'lsm': lambda x: (x, {}),
        'lsmSFC': lambda x: (x, {}),
        'LSMSFC': lambda x: (x, {}),

        'ci': preprocess_bounded_variable
    }
    need_to_fit_params = False
    if dict_to_undo is None:
        need_to_fit_params = True 
        dict_to_undo = {}


    if group_levels:
        variables = group_varlevs_by_variable(ds.varlev.values)
    else:
        variables = group_varlevs_by_variable_individual(ds.varlev.values)

    if mask is None: 
        mask = [] 
        for varlev in ds.varlev.values:
            mask_for_var = xr.ones_like(ds.sel(varlev=varlev).mean('time'))
            mask_for_var = mask_for_var.where(~np.isnan(ds.sel(varlev=varlev).min('time', skipna=False)), other=np.nan)
            mask.append(mask_for_var)
        mask = xr.concat(mask, 'varlev')
    ds = ds * mask 

    tc = []
    for variable in variables.keys(): 
        da = ds.sel(varlev=variables[variable])
        if need_to_fit_params:
            da, parameters = how_to_preprocess[variable](da) 
            dict_to_undo[variable] = parameters 
        else: 
            da, parameters = how_to_preprocess[variable](da, **dict_to_undo[variable])
        tc.append(da) 

    return xr.concat(tc, 'varlev'), dict_to_undo, mask

def undo_preprocess_by_variables(
    ds, 
    plevels=[1000 , 850, 700, 500, 250, 100, 50], 
    pressure_variables = ['q', 'u', 'v', 't', 'z' ],
    surface_variables = ['10u',  '10v',  '2d', '2t', 'ci', 'mtpr', 'skt', 'sp', 'sstk', 'nanmask'],
    dict_to_undo={},
    group_levels=True
):
    """
        designed for era5-flat.zarr, which has all the variables and levels pooled in a single dimension
        returns preprocessed data, dict for undoing preprocessing 
    """

    how_to_preprocess = {
        'CISFC': undo_preprocess_bounded_variable,
        'MTPRSFC' :undo_preprocess_strictly_positive_variable,
        'SKTSFC'  :undo_preprocess_continuous_variable,
        'SPSFC'  :undo_preprocess_continuous_variable,
        'MSLSFC': undo_preprocess_continuous_variable,
        'SSTKSFC'  :undo_preprocess_continuous_variable,
        'VAR_10USFC'  :undo_preprocess_continuous_variable,
        'VAR_10VSFC' :undo_preprocess_continuous_variable,
        'VAR_2DSFC' :undo_preprocess_continuous_variable,
        'VAR_2TSFC'  :undo_preprocess_continuous_variable,
        'q100.0' :undo_preprocess_strictly_positive_variable,
        'q1000.0' :undo_preprocess_strictly_positive_variable,
        'q250.0' :undo_preprocess_strictly_positive_variable,
        'q50.0' :undo_preprocess_strictly_positive_variable,
        'q500.0':undo_preprocess_strictly_positive_variable,
        'q700.0' :undo_preprocess_strictly_positive_variable,
        'q850.0' :undo_preprocess_strictly_positive_variable,
        't100.0'  :undo_preprocess_continuous_variable,
        't1000.0'  :undo_preprocess_continuous_variable,
        't250.0'  :undo_preprocess_continuous_variable,
        't50.0'  :undo_preprocess_continuous_variable,
        't500.0'  :undo_preprocess_continuous_variable,
        't700.0' :undo_preprocess_continuous_variable,
        't850.0'  :undo_preprocess_continuous_variable,
        'u100.0'  :undo_preprocess_continuous_variable,
        'u1000.0' :undo_preprocess_continuous_variable,
        'u250.0' :undo_preprocess_continuous_variable,
        'u50.0' :undo_preprocess_continuous_variable,
        'u500.0' :undo_preprocess_continuous_variable,
        'u700.0' :undo_preprocess_continuous_variable,
        'u850.0': undo_preprocess_continuous_variable,
        'v100.0' :undo_preprocess_continuous_variable,
        'v1000.0' :undo_preprocess_continuous_variable,
        'v250.0' :undo_preprocess_continuous_variable,
        'v50.0' : undo_preprocess_continuous_variable,
        'v500.0' : undo_preprocess_continuous_variable,
        'v700.0': undo_preprocess_continuous_variable,
        'v850.0': undo_preprocess_continuous_variable,
        'z' :undo_preprocess_continuous_variable,

        'z100.0' :undo_preprocess_continuous_variable,
        'z1000.0' :undo_preprocess_continuous_variable,
        'z250.0' :undo_preprocess_continuous_variable,
        'z50.0' : undo_preprocess_continuous_variable,
        'z500.0' : undo_preprocess_continuous_variable,
        'z700.0': undo_preprocess_continuous_variable,
        'z850.0': undo_preprocess_continuous_variable,
        'q': undo_preprocess_strictly_positive_variable,
        't': undo_preprocess_continuous_variable,
        'u': undo_preprocess_continuous_variable, 
        'v': undo_preprocess_continuous_variable, 
        '10u': undo_preprocess_continuous_variable, 
        '10v': undo_preprocess_continuous_variable, 
        '2d': undo_preprocess_continuous_variable, 
        '2t': undo_preprocess_continuous_variable, 
        'mtpr': undo_preprocess_strictly_positive_variable,
        'skt': undo_preprocess_continuous_variable, 
        'sp': undo_preprocess_continuous_variable, 
        'sstk': undo_preprocess_continuous_variable, 
        'nanmask': lambda x: x,
        'nanmaskSFC': lambda x: x,
        'lsm': lambda x: x,
        'lsmSFC': lambda x: x,
        'LSMSFC': lambda x: x,

        'ci': undo_preprocess_bounded_variable
    }

    if group_levels:
        variables = group_varlevs_by_variable(ds.varlev.values)
    else:
        variables = group_varlevs_by_variable_individual(ds.varlev.values)

    tc = []
    for variable in variables.keys(): 
        da = ds.sel(varlev=variables[variable])
        da = how_to_preprocess[variable](da, **dict_to_undo[variable]) 
        tc.append(da) 
    return xr.concat(tc, 'varlev')#.transpose('time', 'varlev', 'lat', 'lon')

