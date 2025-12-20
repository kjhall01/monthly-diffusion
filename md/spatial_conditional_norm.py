
import torch
import torch.nn as nn
import torch.nn.functional as F
from .spectral_conv import SpectralConvS2

class SpatialConditionalRMSNorm(nn.Module):
    """
    Conditional RMSNorm for convolutional feature maps,
    where the conditioning is another spatial map (C_c, H, W).

    Computes per-feature RMS normalization, and modulates using
    gamma/beta maps derived from the conditioning image.
    """

    def __init__(self, 
            in_shape, 
            cond_shape, 
            eps=1e-8, 
            hidden_channels=3,
            conditioning_operator_type = "driscoll-healy",
            rank = 4
        ):
        super().__init__()
        self.eps = eps
        in_channels = in_shape[0]
        hidden_channels = hidden_channels or max(in_shape[0] // 2, cond_shape[0])

        # small conv net to map cond -> (gamma, beta)
        hidden_shape = (hidden_channels, in_shape[1], in_shape[2])
        out_shape = (2 * in_channels, in_shape[1], in_shape[2])
        self.to_gamma_beta = nn.Sequential(
            SpectralConvS2(cond_shape, hidden_shape, operator_type=conditioning_operator_type, rank=rank, return_tuple=False),
            nn.ReLU(),
            SpectralConvS2(hidden_shape, out_shape, operator_type=conditioning_operator_type, rank=rank, return_tuple=False)
        )

        # learnable base parameters (unconditional RMSNorm)
        self.weight = nn.Parameter(torch.ones(in_channels))
        self.bias = nn.Parameter(torch.zeros(in_channels))

    def forward(self, x, cond):
        """
        x:    [B, Cx, H, W] -- feature map to normalize
        cond: [B, Cc, H, W] -- conditioning feature map
        """
        B, C, H, W = x.shape

        # compute RMS across channels and spatial dims
        rms = x.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt() + self.eps
        x_norm = x / rms

        # get spatial gamma and beta from conditioning map
        gamma_beta = self.to_gamma_beta(cond)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)  # [B, Cx, H, W]

        # combine base scale/shift with conditional modulation
        weight = self.weight.view(1, C, 1, 1)
        bias = self.bias.view(1, C, 1, 1)
        out = x_norm * (weight * gamma + 1.0) + (bias + beta)
        return out