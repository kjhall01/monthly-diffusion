"""A small composition example for power users; no training or data required."""

import torch
from monthly_diffusion.modules.mesh import build_mesh_hierarchy_context
from monthly_diffusion.modules.autoencoder import (
    MeshHierarchyAutoencoder,
    AutoencoderConfig,
)
from monthly_diffusion.components.latent_diffusion import (
    LatentDiffusion,
    LatentDiffusionConfig,
)


geometry = build_mesh_hierarchy_context(nlat=18, nlon=36, mesh_levels=(2, 1, 0))
autoencoder = MeshHierarchyAutoencoder(
    3,
    geometry,
    (18, 36),
    AutoencoderConfig(width=16, mesh_levels=(2, 1, 0), refiners=(1, 1, 1)),
)
# One independent component sees only its two 4-channel history states.
diffusion = LatentDiffusion(
    LatentDiffusionConfig(
        n_latent_tokens=12,
        condition_channels=8,
        width=16,
        depth=2,
        time_embedding_channels=16,
    )
)
x = torch.randn(2, 3, 18, 36)
static, month, co2 = torch.zeros(2, 2, 18, 36), torch.randn(2, 2), torch.randn(2, 1)
reconstruction, raw_z = autoencoder(x, static, month, co2)
# Real training fits fixed normalization after AE selection. This is merely
# a dimensional smoke example, not a scientific training recipe.
history = torch.cat((raw_z.detach(), raw_z.detach()), -1)
forecast = diffusion.sample(torch.randn(2, 6), history, torch.randn_like(raw_z))
print("Reconstruction", reconstruction.shape, "latent forecast", forecast.shape)
