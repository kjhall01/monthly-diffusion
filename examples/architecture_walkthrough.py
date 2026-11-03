"""Run from repository root: python -m examples.architecture_walkthrough.

Synthetic architecture walkthrough only: inputs here are already preprocessed.
No climate training or climate realism is claimed by this example.
"""
import torch
from monthly_diffusion.architecture import (
    AutoencoderConfig, MeshHierarchyAutoencoder,
    DenoiserConfig, SphericalDiT, build_mesh_hierarchy_context,
)

def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    # Fixed connectivity is constructed once, not inside forward().
    geometry = build_mesh_hierarchy_context(
        nlat=8, nlon=16, mesh_levels=(1, 0),
        radius_query_fraction_edge_length=1.0,
    )
    config = AutoencoderConfig(width=8, latent_channels=2,
                               mesh_levels=(1,0), refiners=(1,1))
    autoencoder = MeshHierarchyAutoencoder(
        channels=2, geometry=geometry, grid_shape=(8,16), config=config,
        share_conditioning=False,
    )
    # [batch, variable, latitude, longitude].
    state = torch.randn(2,2,8,16)
    statics = torch.randn(2,2,8,16)
    month = torch.tensor([[0.,1.],[1.,0.]])  # precomputed month harmonics
    co2 = torch.tensor([[0.],[1.]])         # standardized CO2, NOT raw ppm
    reconstructed, latent = autoencoder(state,statics,month,co2)
    print('state -> latent -> reconstructed:',state.shape,latent.shape,reconstructed.shape)
    loss=(reconstructed-state).square().mean()
    loss.backward()
    print('Synthetic reconstruction MSE:',float(loss.detach()))

    # The network takes supplied noisy latents, history, and diffusion time.
    # A later diffusion package will define the noise schedule and sampler.
    denoiser = SphericalDiT(DenoiserConfig(
        latent_channels=2,n_latent_tokens=latent.shape[1],condition_channels=4,
        width=8,depth=1,n_heads=2,n_forcings=2,time_embedding_channels=8))
    history=torch.cat([latent.detach(),latent.detach()],dim=-1)
    prediction=denoiser(torch.randn_like(latent),month,history,t=.5)
    print('Denoiser prediction:',prediction.shape)

if __name__=='__main__': main()
