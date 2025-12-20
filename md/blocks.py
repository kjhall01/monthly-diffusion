import torch_harmonics as th 
import torch 
from torch import nn 
from torch.nn import functional as F
import math 
from .spectral_conv import SpectralConvS2, SpectralResample
from .spatial_conditional_norm import SpatialConditionalRMSNorm
from .positional import AddChannelEmbedding

class ConcatCondishBlock(nn.Module):
    """
    A block which:
    1. Concatenates conditioning tensor w/ x along channel dim
    2. Concatenates static channels w/ x along channel dim 
    3. Applies SpectralConvS2 convolution, turning
        (in_c + static_c + condition_c, in_h, in_w) 
        into
        (out_c, out_h, out_w)
    4. Applies Nonlinear Activation (GELU)
    5. Does a linear spectral resample of conditioning tensor to (condition_c, out_h, out_w)
    6. Applies Spatial Conditional RMS Norm to the output with downsampled conditioning tensor
    7. returns both nonlinear x  and downsampled condition tensor 

    static tensor is treated differently because it is not used for the conditional RMS norm
    """

    def __init__(
        self,
        in_shape,
        out_shape,
        n_conditioning_channels,
        n_static_channels = 0,
        activation = nn.GELU,
        is_sfno_block = False,
        sfno_hidden_layer = 32,
        has_resampled_condition = False,
        varnames = None,
        rank = 32,
        operator_type="spectral_fc",
        conditioning_operator_type = "driscoll-healy",
        conditioning_hidden_channels = 4,
        conditioning_rank = 4
    ):
        super(ConcatCondishBlock, self).__init__()
        self.in_c, self.in_h, self.in_w = in_shape
        self.condition_c = n_conditioning_channels
        self.static_c = n_static_channels
        self.varnames = varnames 

        in_c_with_condition_and_statics = self.in_c + n_conditioning_channels + n_static_channels

        out_c, out_h, out_w = out_shape
        in_condition_shape = (n_conditioning_channels, self.in_h, self.in_w)
        out_condition_shape = (n_conditioning_channels, out_h, out_w)

        self.in_conv = SpectralConvS2(
            in_shape = (in_c_with_condition_and_statics, self.in_h, self.in_w),
            out_shape = out_shape,
            rank=rank,
            operator_type=operator_type
        )
        self.activation = activation() 

        self.has_resampled_condition = has_resampled_condition
        if not self.has_resampled_condition:
            in_spatial_shape = (in_condition_shape[1], in_condition_shape[2])
            out_spatial_shape = (out_condition_shape[1], out_condition_shape[2])
            self.condition_resample = SpectralResample(in_spatial_shape, out_spatial_shape)

        self.spatial_conditioning = SpatialConditionalRMSNorm(
            out_shape,
            out_condition_shape,
            hidden_channels = conditioning_hidden_channels,
            conditioning_operator_type = conditioning_operator_type,
            rank = conditioning_rank
        )

        self.spatial_conditioning2 = SpatialConditionalRMSNorm(
            out_shape,
            out_condition_shape,
            hidden_channels = conditioning_hidden_channels,
            conditioning_operator_type = conditioning_operator_type,
            rank = conditioning_rank
        )

        self.is_sfno_block = is_sfno_block
        if self.is_sfno_block:
            self.in_c_with_condition_and_statics = in_c_with_condition_and_statics
            self.mlp1_layer1 = nn.Conv2d(in_c_with_condition_and_statics, sfno_hidden_layer, kernel_size = 1)
            self.mlp1_layer2 = nn.Conv2d(sfno_hidden_layer, out_c, kernel_size = 1)
        self.out = nn.Conv2d(out_c, out_c, kernel_size=1)
        
        if self.varnames is not None:
            self.embedding = AddChannelEmbedding( varnames, len(varnames), embed_dim=8)


    def forward(self, x, c, statics=None, c_resampled=None ):
        B, C, H, W = x.shape 
        assert C == self.in_c and H == self.in_h and W == self.in_w, f"Incorrect image shape - got ({C}, {H}, {W}) and expected ({self.in_c}, {self.in_h}, {self.in_w})"
        cB, cC, H, W = c.shape 
        assert cC == self.condition_c and H == self.in_h and W == self.in_w, f"Incorrect condition shape - got ({cC}, {H}, {W}) and expected ({self.condition_c}, {self.in_h}, {self.in_w})"
        x = torch.cat([x, c], dim=1)

        if statics is not None:
            C, H, W = statics.shape 
            assert C == self.static_c and H == self.in_h and W == self.in_w, f"Incorrect statics shape - got ({C}, {H}, {W}) and expected ({self.static_c}, {self.in_h}, {self.in_w})"
            x = torch.cat([x, (torch.ones(x.shape[0], *statics.shape).to(x.device) * statics.unsqueeze(0))], dim=1)
        
        if self.varnames is not None:
            x = self.embedding(x)

        x, res = self.in_conv(x)
        x = self.activation(x)

        if c_resampled is None:
            c_resampled = self.condition_resample(c)

        x = self.spatial_conditioning(x, c_resampled)
        
        if self.is_sfno_block:
            res_nl = self.mlp1_layer1(res)
            res_nl = self.activation(res_nl)
            res_nl = self.mlp1_layer2(res_nl)
            x = x + res_nl 
            x = self.spatial_conditioning2(x, c_resampled)
        return self.out(x), c_resampled


class JustCondishBlock(nn.Module):
    """
    A block which:
    1. Applies SpectralConvS2, turning
        (in_c, in_h, in_w) 
        into
        (out_c, out_h, out_w)
    2. Applies Nonlinear Activation (GELU)
    3. Does a linear spectral resample of conditioning tensor to (condition_c, out_h, out_w)
    4. Applies Spatial Conditional RMS Norm to the output with downsampled conditioning tensor
    5. returns both nonlinear x  and downsampled condition tensor 

    static tensor is treated differently because it is not used for the conditional RMS norm
    """
    def __init__(
        self,
        in_shape,
        out_shape,
        n_conditioning_channels,
        activation = nn.GELU,
        is_sfno_block = False,
        sfno_hidden_layer = 32,
        has_resampled_condition = False,
        rank = 32,
        conditioning_operator_type = "driscoll-healy",
        conditioning_hidden_channels = 4,
        conditioning_rank = 4,
    ):
        super(JustCondishBlock, self).__init__()
        self.in_c, self.in_h, self.in_w = in_shape
        self.condition_c = n_conditioning_channels

        self.out_c, self.out_h, self.out_w = out_shape
        in_condition_shape = (n_conditioning_channels, self.in_h, self.in_w)
        out_condition_shape = (n_conditioning_channels, self.out_h, self.out_w)

        self.in_conv = SpectralConvS2(
            in_shape = in_shape ,
            out_shape = out_shape,
            rank = rank
        )
        self.activation = activation() 

        self.has_resampled_condition = has_resampled_condition
        if not self.has_resampled_condition:
            in_spatial_shape = (in_condition_shape[1], in_condition_shape[2])
            out_spatial_shape = (out_condition_shape[1], out_condition_shape[2])
            self.condition_resample = SpectralResample(in_spatial_shape, out_spatial_shape)

        self.spatial_conditioning = SpatialConditionalRMSNorm(
            out_shape,
            out_condition_shape,
            hidden_channels = conditioning_hidden_channels,
            conditioning_operator_type = conditioning_operator_type,
            rank = conditioning_rank
        )

        self.spatial_conditioning2 = SpatialConditionalRMSNorm(
            out_shape,
            out_condition_shape,
            hidden_channels = conditioning_hidden_channels,
            conditioning_operator_type = conditioning_operator_type,
            rank = conditioning_rank
        )
        
        self.is_sfno_block = is_sfno_block
        if self.is_sfno_block:
            self.mlp1_layer1 = nn.Conv2d(self.in_c, sfno_hidden_layer, kernel_size = 1)
            self.mlp1_layer2 = nn.Conv2d(sfno_hidden_layer, self.out_c, kernel_size = 1)
        self.out = nn.Conv2d(self.out_c, self.out_c, kernel_size=1)


    def forward(self, x, c, c_is_resampled=False):
        B, C, H, W = x.shape 
        assert C == self.in_c and H == self.in_h and W == self.in_w, f"Incorrect image shape - got ({C}, {H}, {W}) and expected ({self.in_c}, {self.in_h}, {self.in_w})"
       
        cB, cC, H, W = c.shape 
    
        x, res = self.in_conv(x)

        if c_is_resampled:
            c_resampled = c 
        else:
            c_resampled = self.condition_resample(c)
            
        x = self.spatial_conditioning(x, c_resampled)
        x = self.activation(x)

        if self.is_sfno_block:
            res_nl = self.mlp1_layer1(res)
            res_nl = self.activation(res_nl)
            res_nl = self.mlp1_layer2(res_nl)
            x = x + res_nl 
            x = self.spatial_conditioning2(x, c_resampled)
        return self.out(x), c_resampled