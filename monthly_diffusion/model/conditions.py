import torch 
from torch import nn
import torch.nn.functional as F


class ScalarToLatent(nn.Module):
    """
        maps scalar inputs directly to all NxC latent dimensions
    """
    def __init__(
        self, 
        n_scalars, 
        mesh,
        out_channels = 1,
        hidden_dim = 128
    ):
        super().__init__()
        self.out_channels = out_channels
        self.mesh = mesh
        self.in_proj = nn.Linear(n_scalars, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_channels * mesh.xyz.shape[0])

    def forward(self, conditions):
        h = self.in_proj(conditions)
        h = F.silu(h)
        h = self.out_proj(h)
        return h.reshape(conditions.shape[0], self.mesh.xyz.shape[0], self.out_channels)
    

class ScalarToLatentChannels(nn.Module):
    """
        maps scalar inputs directly to C latent dimensions, broadcast ofer N nodes
    """
    def __init__(
        self, 
        n_scalars, 
        mesh,
        out_channels = 1,
        hidden_dim = 128
    ):
        super().__init__()
        self.out_channels = out_channels
        self.mesh = mesh
        self.in_proj = nn.Linear(n_scalars, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_channels)

    def forward(self, conditions):
        # conditions: [B, n_scalars]
        h = F.silu(self.in_proj(conditions))  # [B, hidden_dim]
        h = self.out_proj(h)                 # [B, out_channels]

        return h[:, None, :].expand(
            -1, self.mesh.xyz.shape[0], -1
        )  # [B, N, out_channels]

