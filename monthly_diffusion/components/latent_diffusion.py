"""Single-component full-state v-prediction and differentiable seven-NFE sampling."""
import torch
from torch.nn import functional as F
from ..modules.spherical_dit import SphericalDiT, LatentDiffusionConfig
from .sampling import _advance

TIMESTEPS = (999, 674, 373, 120, 29, 6, 0)


class LatentDiffusion(SphericalDiT):
    def diffuse(self, z, t=None, noise=None):
        t = self.prepare_timestep(t, z.shape[0], z.device)
        noise = torch.randn_like(z) if noise is None else noise
        alpha, sigma = self.schedule(t, z)
        return alpha*z + sigma*noise, alpha*noise - sigma*z, t

    def loss(self, z, calendar, history, *, t=None, noise=None):
        noisy, target, t = self.diffuse(z, t, noise)
        return F.mse_loss(self(noisy, calendar, history, t), target)

    def sample_for_training(self, calendar, history, noise, *, timesteps=TIMESTEPS):
        """Keep autograd through all denoising steps; callers detach deliberately."""
        if timesteps[0] != self.diffusion_steps-1 or timesteps[-1] != 0 or any(
                a <= b for a, b in zip(timesteps[:-1], timesteps[1:])):
            raise ValueError("Descending sampler grid must include both diffusion endpoints")
        grid = torch.tensor(timesteps, device=noise.device)
        z, previous_data, previous_step = noise, None, None
        for i, step in enumerate(grid):
            t = step.expand(z.shape[0])
            v = self(z, calendar, history, t)
            alpha, sigma = self.schedule(t, z)
            data = alpha*z - sigma*v
            if i == len(grid)-1:
                return data
            z = _advance(z, data, previous_data, head=self, step=step,
                         next_step=grid[i+1], previous_step=previous_step,
                         method="dpmpp2m_midpoint")
            previous_data, previous_step = data, step

    @torch.no_grad()
    def sample(self, calendar, history, noise, *, timesteps=TIMESTEPS):
        return self.sample_for_training(calendar, history, noise, timesteps=timesteps)
