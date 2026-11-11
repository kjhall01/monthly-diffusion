"""Two diffusion heads with shared dates, independent noise, and explicit coupling.

Histories are concatenated [current, previous] on the channel axis. Calendar is
[current sin/cos, previous sin/cos, target sin/cos]. No CO2 enters either head.
"""
from dataclasses import dataclass, replace
import torch
from torch import nn
from torch.nn import functional as F
from .latent_diffusion import LatentDiffusion, LatentDiffusionConfig, TIMESTEPS
from .coupling import Coupling, ar1_noise
from .sampling import denoise_paired_with_timesteps


@dataclass(frozen=True)
class CoupledDiffusionConfig:
    atmosphere_latent_channels: int = 4
    ocean_latent_channels: int = 4
    n_latent_tokens: int = 162
    width: int = 132
    depth: int = 4
    n_heads: int = 4
    ffn_ratio: float = 1.55
    time_embedding_channels: int = 64
    atmosphere_tau: float = .5
    ocean_tau: float = 6.
    coupling: str = "jacobi"
    cross_gradients: bool = False


class CoupledLatentDiffusion(nn.Module):
    def __init__(self, config=CoupledDiffusionConfig()):
        super().__init__()
        self.config = config
        self.coupling = Coupling(config.coupling)
        a, o = config.atmosphere_latent_channels, config.ocean_latent_channels
        common = LatentDiffusionConfig(n_latent_tokens=config.n_latent_tokens,
                    width=config.width, depth=config.depth, n_heads=config.n_heads,
                    ffn_ratio=config.ffn_ratio, time_embedding_channels=config.time_embedding_channels)
        self.atmosphere = LatentDiffusion(replace(common, latent_channels=a,
            condition_channels=2*a+2*o+(o if self.coupling == Coupling.JACOBI else 0)))
        self.ocean = LatentDiffusion(replace(common, latent_channels=o, condition_channels=2*o+3*a))
        self.n_latent_tokens, self.diffusion_steps = config.n_latent_tokens, common.diffusion_steps

    def _atmosphere_context(self, ah, oh, noisy_ocean=None):
        if not self.config.cross_gradients:
            oh = oh.detach()
            noisy_ocean = None if noisy_ocean is None else noisy_ocean.detach()
        values = (ah, oh, noisy_ocean) if self.coupling == Coupling.JACOBI else (ah, oh)
        return torch.cat(values, -1)

    def _ocean_context(self, oh, ah, atmosphere):
        if not self.config.cross_gradients:
            ah, atmosphere = ah.detach(), atmosphere.detach()
        return torch.cat((oh, ah, atmosphere), -1)

    def loss_terms(self, a, o, calendar, ah, oh, *, t=None, noise_a=None, noise_o=None,
                   atmosphere_condition=None):
        t = self.atmosphere.prepare_timestep(t, a.shape[0], a.device)
        an, av, _ = self.atmosphere.diffuse(a, t, noise_a)
        on, ov, _ = self.ocean.diffuse(o, t, noise_o)
        if self.coupling == Coupling.ATMOSPHERE_FIRST and atmosphere_condition is None:
            raise ValueError("Atmosphere-first loss requires a GENERATED atmosphere, not its target")
        ao = an if self.coupling == Coupling.JACOBI else atmosphere_condition
        ap = self.atmosphere(an, calendar, self._atmosphere_context(ah, oh, on), t)
        op = self.ocean(on, calendar, self._ocean_context(oh, ah, ao), t)
        al, ol = F.mse_loss(ap, av), F.mse_loss(op, ov)
        return {"transition": (al+ol)/2, "atmosphere_transition": al, "ocean_transition": ol}

    def terminal_noise(self, ah, oh, previous=(None, None)):
        a = ar1_noise(ah[..., :self.config.atmosphere_latent_channels], previous[0],
                      tau_months=self.config.atmosphere_tau)
        o = ar1_noise(oh[..., :self.config.ocean_latent_channels], previous[1],
                      tau_months=self.config.ocean_tau)
        return a, o

    def sample_for_training(self, calendar, ah, oh, *, previous_noise=(None, None), noise=None):
        na, no = self.terminal_noise(ah, oh, previous_noise) if noise is None else noise
        if self.coupling == Coupling.JACOBI:
            a, o = denoise_paired_with_timesteps(self, calendar, ah, oh,
                timesteps=torch.tensor(TIMESTEPS, device=ah.device), method="dpmpp2m_midpoint",
                atmosphere_noise=na, ocean_noise=no)
        else:
            a = self.atmosphere.sample_for_training(calendar, self._atmosphere_context(ah, oh), na)
            o = self.ocean.sample_for_training(calendar, self._ocean_context(oh, ah, a), no)
        return a, o, na, no

    @torch.no_grad()
    def sample(self, *args, **kwargs):
        return self.sample_for_training(*args, **kwargs)

    def rollout_loss(self, batch, *, loss_weight=.01):
        """Equal-weight horizons; own rollout gradients live, cross paths explicit.

        Each input sequence is chronological [previous, current, future...].
        The tensor concatenation below deliberately reverses only the history.
        """
        a, o, months = batch["atmosphere"], batch["ocean"], batch["month"]
        ac, ap, oc, op = a[:, 1], a[:, 0], o[:, 1], o[:, 0]
        noise, terms = (None, None), {}
        steps = a.shape[1]-2
        for step in range(steps):
            ah, oh = torch.cat((ac, ap), -1), torch.cat((oc, op), -1)
            calendar = torch.cat((months[:, step+1], months[:, step], months[:, step+2]), -1)
            if self.coupling == Coupling.ATMOSPHERE_FIRST:
                na, no = self.terminal_noise(ah, oh, noise)
                predicted_a = self.atmosphere.sample_for_training(calendar, self._atmosphere_context(ah, oh), na)
                result = self.loss_terms(a[:, step+2], o[:, step+2], calendar, ah, oh,
                                         atmosphere_condition=predicted_a)
                if step+1 < steps:
                    predicted_o = self.ocean.sample_for_training(calendar,
                                            self._ocean_context(oh, ah, predicted_a), no)
                    noise = na, no
            else:
                result = self.loss_terms(a[:, step+2], o[:, step+2], calendar, ah, oh)
                if step+1 < steps:
                    predicted_a, predicted_o, na, no = self.sample_for_training(calendar, ah, oh, previous_noise=noise)
                    noise = na, no
            for name, value in result.items():
                terms[f"{name}_step{step+1}"] = value
            if step+1 < steps:
                ac, ap, oc, op = predicted_a, ac, predicted_o, oc
        for name in ("transition", "atmosphere_transition", "ocean_transition"):
            terms[name] = torch.stack([terms[f"{name}_step{i+1}"] for i in range(steps)]).mean()
        terms["loss"] = loss_weight*terms["transition"]
        return terms
