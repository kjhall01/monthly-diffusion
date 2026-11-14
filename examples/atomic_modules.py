"""A small composition example for power users; no training or data required."""

import torch
from monthly_diffusion.mesh import build_mesh_hierarchy
from monthly_diffusion.autoencoder import Autoencoder
from monthly_diffusion.diffusion import LatentDiffusion


geometry = build_mesh_hierarchy(nlat=18, nlon=36, mesh_levels=(2, 1, 0))
autoencoder = Autoencoder(
    3, geometry, (18, 36), statics=torch.zeros(2, 18, 36), width=16, refiners=(1, 1, 1)
)
# One independent component sees only its two 4-channel history states.
diffusion = LatentDiffusion(
    n_latent_tokens=12,
    condition_channels=8,
    width=16,
    depth=2,
    time_embedding_channels=16,
)
x = torch.randn(2, 3, 18, 36)
month, co2 = torch.randn(2, 2), torch.randn(2, 1)
reconstruction, raw_z = autoencoder(x, month, co2)
# Real training fits fixed normalization after AE selection. This is merely
# a dimensional smoke example, not a scientific training recipe.
history = torch.cat((raw_z.detach(), raw_z.detach()), -1)
forecast = diffusion.sample(torch.randn(2, 6), history, torch.randn_like(raw_z))
print("Reconstruction", reconstruction.shape, "latent forecast", forecast.shape)
