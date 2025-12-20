import numpy as np 
import xarray as xr 
import pandas as pd 
from scipy.stats import norm
import torch 


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal 

def deniell(fourier_coeffs, frequencies, m_for_deniell_smoothing=15):
    initial_shape = frequencies.shape[0]
    if m_for_deniell_smoothing == 0:
        return fourier_coeffs, frequencies
    padded = F.pad(fourier_coeffs.t(), (m_for_deniell_smoothing//2 , m_for_deniell_smoothing//2), mode='reflect').t()
    freqs = torch.vstack([frequencies.reshape(1,-1), frequencies.reshape(1,-1)])
    padded_freqs = F.pad(freqs, (m_for_deniell_smoothing//2, m_for_deniell_smoothing//2), mode='reflect')
    tc = []
    tc2 = []
    smoothing_weights = norm.pdf(np.linspace(-2,2, m_for_deniell_smoothing))
    smoothing_weights = smoothing_weights / smoothing_weights.sum()
    for i in range(m_for_deniell_smoothing):
        tcc =  padded[i: i+fourier_coeffs.shape[0], : ]
        tcc2 = padded_freqs[0, i:i+fourier_coeffs.shape[0]].squeeze()
        tc.append(tcc * smoothing_weights[i])
        tc2.append(tcc2 * smoothing_weights[i])

    fourier_coeffs = torch.dstack(tc).sum(dim=-1)
    frequencies = torch.dstack(tc2).sum(dim=-1).squeeze()
    assert frequencies.shape[0] == initial_shape, 'initial shape changes - {} vs {}'.format(initial_shape, frequencies.shape[0])
    return fourier_coeffs, frequencies

def global_detrend(da, time_dim='time', spatial_dims=['lat', 'lon'], deg=2, p=None):
    if p is not None:
        return da - xr.polyval(da['time'],  p.polyfit_coefficients), None, None, p
    else:
        # detrend along a single dimension
        weights = np.cos(np.deg2rad(da.lat))
        weights.name = "weights"

        # Apply the weights and compute the mean across lat/lon
        da_weighted = da.weighted(weights)
        weighted_mean = da_weighted.mean(dim=spatial_dims)
        p = weighted_mean.polyfit(dim=time_dim, deg=deg)
        fit = xr.polyval(da[time_dim], p.polyfit_coefficients)
        return da - fit, fit, weighted_mean, p #.polyfit_coefficient


def detrend(da, dim='time', deg=2):
    # detrend along a single dimension
    # below assumes monthly data, returns trend in per-year
    da = da.assign_coords({dim: [i/12 for i in range(da.coords[dim].values.shape[0])]})
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit, p.polyfit_coefficients.sel(degree=1), p

def remove_climo(monthly, dim='time', sub=None, monthly_climatology=None):
    if monthly_climatology is None:
        if sub is not None: 
            monthly_climatology = monthly.sel({dim:sub}).groupby(f'{dim}.month').mean()
        else:
            monthly_climatology = monthly.groupby(f'{dim}.month').mean()
    
    toconcat = []
    for year in sorted(list(set( [ pd.Timestamp(i).year for i in monthly.coords[dim].values] ))):
        ds_yearly = monthly.sel(time=slice(pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12,31))).groupby(f'{dim}.month').mean() - monthly_climatology
        ds_yearly = ds_yearly.assign_coords({'month': [ pd.Timestamp(year, j, 1) for j in ds_yearly.coords['month'].values ] } ).rename({'month': dim})
        toconcat.append(ds_yearly)
    monthly_anom = xr.concat(toconcat, dim).sortby(dim)
    return monthly_anom, monthly_climatology


def intermediate_shapes(data_dimensions, hidden_layers=[], squeeze_factors=None, downsample_factor=1):
    if squeeze_factors is None:
        downsample_factor = [ downsample_factor for _ in range(len(hidden_layers))]
    else:
        downsample_factor = squeeze_factors
    in_c, in_h, in_w = data_dimensions
    shapes = [(in_c, in_h, in_w)]
    for _, l in enumerate(hidden_layers):
        in_h = int(in_h // downsample_factor[_])
        in_w = int(in_w // downsample_factor[_])
        shapes.append((l, in_h, in_w))
    return shapes[::-1] # small to big  c


def gaussian_crps(mu, sigma, y):
    """
    Calculates the Gaussian CRPS using the analytical solution.

    Args:
        mu (torch.Tensor): Predicted means of the Gaussian distributions.
        sigma (torch.Tensor): Predicted standard deviations (positive) of the Gaussian distributions.
        y (torch.Tensor): Observed ground truth values.

    Returns:
        torch.Tensor: The CRPS value.
    """
    # Create standard normal distribution for Phi and phi
    standard_normal = Normal(0, 1)
    
    # Standardize the observation
    z = (y - mu) / sigma
    
    # Calculate the components of the analytical solution
    phi_z = torch.exp(standard_normal.log_prob(z))
    Phi_z = standard_normal.cdf(z)
    
    # Apply the CRPS formula
    crps_value = sigma * (
        (z * (2 * Phi_z - 1)) + (2 * phi_z) - (1 / torch.sqrt(torch.tensor(torch.pi)))
    )
    
    return crps_value.mean() # Return the average CRPS over the batch
