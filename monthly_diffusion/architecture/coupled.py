"""Explicit component assembly. No coupling schedule, loss, or sampler here."""
from torch import nn
from .autoencoder import AutoencoderConfig, MeshHierarchyAutoencoder


class CoupledAutoencoder(nn.Module):
    """Independent atmosphere/ocean networks sharing fixed geometry only.

    Call .atmosphere(...) or .ocean(...) explicitly. This container does not
    itself exchange information; latent coupling belongs to the denoisers.
    """
    def __init__(self, atmosphere_channels, ocean_channels, geometry, grid_shape,
                 config=AutoencoderConfig(), *, ocean_mask_aware=False,
                 share_conditioning=False):
        super().__init__()
        self.atmosphere = MeshHierarchyAutoencoder(
            atmosphere_channels, geometry, grid_shape, config,
            share_conditioning=share_conditioning)
        self.ocean = MeshHierarchyAutoencoder(
            ocean_channels, geometry, grid_shape, config,
            mask_aware=ocean_mask_aware, share_conditioning=share_conditioning)


class CoupledDenoiser(nn.Module):
    """Two denoisers with caller-assembled history/cross-component context.

    Both receive context explicitly: no hidden Jacobi/Gauss-Seidel choice.
    Inference scheduling and context concatenation stay in the rollout code.
    """
    def __init__(self, atmosphere, ocean):
        super().__init__()
        self.atmosphere = atmosphere
        self.ocean = ocean

    def forward(self, atmosphere_noisy, ocean_noisy, *,
                atmosphere_context, ocean_context, condition, time):
        return (
            self.atmosphere(atmosphere_noisy, condition, atmosphere_context, time),
            self.ocean(ocean_noisy, condition, ocean_context, time),
        )
