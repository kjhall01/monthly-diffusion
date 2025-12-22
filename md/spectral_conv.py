# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Modified by Kyle Hall (kylehall@umd.edu) on 11/14/2025 

from torch_harmonics import RealSHT, InverseRealSHT
import torch 
from torch import nn 
from torch.nn import functional as F
import math 
from .spectral_fc import SpectralFC 

class SpectralResample(nn.Module):
    """
    Just separating the grid resampling via 
    SHT transformation from the actual convolution
    So I can use it for linear downscaling of 
    conditions
    """

    def __init__(self, 
            in_shape, 
            out_shape, 
        ):

        super(SpectralResample, self).__init__()

        self.forward_transform = RealSHT(*in_shape).float()
        self.inverse_transform = InverseRealSHT(
            *out_shape, 
            lmax=self.forward_transform.lmax, 
            mmax=self.forward_transform.mmax
        ).float()

        self.modes_lat = self.inverse_transform.lmax
        self.modes_lon = self.inverse_transform.mmax

        assert self.inverse_transform.lmax == self.modes_lat
        assert self.inverse_transform.mmax == self.modes_lon


    def forward(self, x):
        dtype = x.dtype
        x = x.float()

        if x.device == 'cuda':
            with torch.autocast(device_type="cuda", enabled=False):
                x = self.forward_transform(x)
                x = self.inverse_transform(x)
        else:
            x = self.forward_transform(x)
            x = self.inverse_transform(x)
        x = x.type(dtype)

        return x

class SpectralConvS2(nn.Module):
    """
    Spectral Convolution according to Driscoll & Healy. Designed for convolutions on the two-sphere S2
    using the Spherical Harmonic Transforms in torch-harmonics, but supports convolutions on the periodic
    domain via the RealFFT2 and InverseRealFFT2 wrappers.
    
    Parameters
    ----------

    in_shape : int
        (C, H, W) for input
    out_shape : int
        (C, H, W) for outuput 
    gain : float, optional
        Gain factor for weight initialization, by default 2.0
    operator_type : str, optional
        Type of spectral operator ("driscoll-healy", "diagonal", "block-diagonal"), by default "driscoll-healy"
    bias : bool, optional
        Whether to use bias, by default False
    """
    
    def __init__(self, 
            in_shape, 
            out_shape, 
            gain=2.0, 
            operator_type='spectral_fc', #"driscoll-healy", 
            bias=False,
            return_tuple=True,
            rank=16
        ):
        super(SpectralConvS2, self).__init__()
        self.return_tuple = return_tuple
        in_channels, out_channels = in_shape[0], out_shape[0]
        in_spatial_shape = (in_shape[1], in_shape[2])
        out_spatial_shape = (out_shape[1], out_shape[2])

        self.spectral_resample = SpectralResample(in_spatial_shape, out_spatial_shape)

        # if we need to do the SHT downsampling god this is cool 
        self.scale_residual = (self.spectral_resample.forward_transform.nlat != self.spectral_resample.inverse_transform.nlat) or (self.spectral_resample.forward_transform.nlon != self.spectral_resample.inverse_transform.nlon)
        
        # remember factorization details
        self.operator_type = operator_type
        weight_shape = [out_channels, in_channels]

        if self.operator_type == 'spectral_fc':
            self.spectral_fc = SpectralFC(in_channels, out_channels, self.spectral_resample.modes_lat, self.spectral_resample.modes_lon, rank=rank)
        else:
            if self.operator_type == "diagonal":
                weight_shape += [self.spectral_resample.modes_lat, self.spectral_resample.modes_lon]
                self.contract_func = "...ilm,oilm->...olm"
            elif self.operator_type == "block-diagonal":
                weight_shape += [self.spectral_resample.modes_lat, self.spectral_resample.modes_lon, self.spectral_resample.modes_lon]
                self.contract_func = "...ilm,oilnm->...oln"
            elif self.operator_type == "driscoll-healy":
                weight_shape += [self.spectral_resample.modes_lat]
                self.contract_func = "...ilm,oil->...olm"
            else:
                raise NotImplementedError(f"Unkown operator type f{self.operator_type}")

            # form weight tensors
            scale = math.sqrt(gain / in_channels)
            self.weight = nn.Parameter(scale * torch.randn(*weight_shape, dtype=torch.complex64))
            if bias:
                self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, x):

        dtype = x.dtype
        x = x.float()
        residual = x

        if x.device == 'cuda':
            with torch.autocast(device_type="cuda", enabled=False):
                x = self.spectral_resample.forward_transform(x)
                if self.scale_residual:
                    residual = self.spectral_resample.inverse_transform(x)
        else:
            x = self.spectral_resample.forward_transform(x)
            if self.scale_residual:
                residual = self.spectral_resample.inverse_transform(x)

        if self.operator_type == 'spectral_fc':
            x = self.spectral_fc(x)
        else:
            x = torch.einsum(self.contract_func, x, self.weight)

        if x.device == 'cuda':
            with torch.autocast(device_type="cuda", enabled=False):
                x = self.spectral_resample.inverse_transform(x)
        else:
            x = self.spectral_resample.inverse_transform(x)

        if hasattr(self, "bias"):
            x = x + self.bias
        x = x.type(dtype)

        if self.return_tuple:
            return x, residual
        else:
            return x
        

