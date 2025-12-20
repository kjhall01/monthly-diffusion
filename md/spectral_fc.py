import torch
import torch.nn as nn
import math

class SpectralFC(nn.Module):
    """
    SFNO-style spectral kernel with tensor-product structure:    
    Applies a low-rank global spectral operator to X[..., C_in, L, M] -> X[..., C_out, L, M].
    
    Shapes:
        X          : [..., C_in, L, M] (complex)
        W_in       : [rank, C_in]
        W_out      : [C_out, rank]
        A (out)    : [rank, L, M]
        B (in)     : [rank, L, M]
    """
    def __init__(self, in_channels, out_channels, L, M, rank=32, gain=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.L = L
        self.M = M
        self.rank = rank

        # Channel mixing: C_in -> rank
        self.W_in = nn.Parameter(1e-3 *
            torch.randn(rank, in_channels, dtype=torch.complex64) 
            * math.sqrt(gain / in_channels)
        )
        # Channel mixing: rank -> C_out
        self.W_out = nn.Parameter(1e-3 *
            torch.randn(out_channels, rank, dtype=torch.complex64)
            * math.sqrt(gain / rank)
        )

        # Spectral patterns for output and input sides
        # A: output "basis" over (l, m)
        self.A = nn.Parameter(1e-3 *
            torch.randn(rank, L, M, dtype=torch.complex64) 
            * (gain / math.sqrt(L * M))
        )
        # B: input "basis" over (l, m)
        self.B = nn.Parameter(1e-3 *
            torch.randn(rank, L, M, dtype=torch.complex64) 
            * (gain / math.sqrt(L * M))
        )

    def forward(self, X):
        """
        Effectively, Fully-connected Linear Autoencoder in Spectral Space
        (with reduced dimensionality in latent - true FC would be too many params) 
        
        X: [..., C_in, L, M] complex
        Returns: [..., C_out, L, M] complex
        """
        
        # ensure complex
        X = X.to(torch.complex64)

        #  Channel mixing: C_in -> rank
        X_rank = torch.einsum("...iLM,ri->...rLM", X, self.W_in)  # [..., R, L, M]

        #  Compress spectral info for each rank using B:
        s = torch.einsum("...rLM,rLM->...r", X_rank, self.B)  # [..., R]

        # 3) Expand back to full spectral grid with A:
        Y_rank = torch.einsum("...r,rLM->...rLM", s, self.A)  # [..., R, L, M]

        # 4) Channel mixing: rank -> C_out
        Y = torch.einsum("...rLM,or->...oLM", Y_rank, self.W_out)  # [..., C_out, L, M]

        return Y
