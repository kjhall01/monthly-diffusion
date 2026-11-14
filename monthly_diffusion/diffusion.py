"""Coupled latent diffusion, from attention blocks through the complete sampler.

Histories are [current, previous] along channels. Calendar is current/previous/
target month sin/cos. Optional target-month CO2 enters each block through FiLM.
Conditioning dropout and CFG remove only CO2, retaining history and calendar.
denoise() keeps gradients;
sample() uses the same solver without gradients for inference.
"""

from __future__ import annotations
import math
from enum import Enum
from typing import Literal
import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, width: int, ratio: float):
        super().__init__()
        hidden = max(1, int(round(width * ratio)))
        self.net = nn.Sequential(
            nn.Linear(width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FiLM(nn.Module):
    """Feature-wise affine conditioning initialized as an identity map."""

    def __init__(self, width: int, cond_width: int):
        super().__init__()
        self.projection = nn.Linear(cond_width, 2 * width)
        nn.init.normal_(self.projection.weight, std=1.0e-3)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.projection(cond).chunk(2, dim=-1)
        return x * (1.0 + scale.unsqueeze(-2)) + shift.unsqueeze(-2)


class DiTBlock(nn.Module):
    def __init__(self, width: int, n_heads: int, ffn_ratio: float, cond_width: int):
        super().__init__()
        self.self_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(width, n_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(width, n_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = FeedForward(width, ffn_ratio)
        self.film = FiLM(width, cond_width)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        h = self.self_norm(x)
        h, _ = self.self_attention(h, h, h, need_weights=False)
        x = x + h
        q = self.cross_norm(x)
        kv = self.context_norm(context)
        h, _ = self.cross_attention(q, kv, kv, need_weights=False)
        x = x + h
        x = self.film(x, cond)
        return x + self.ffn(self.ffn_norm(x))


class SphericalDiT(nn.Module):
    """Predict v from noisy latent tokens, history and calendar conditioning."""

    def __init__(
        self,
        *,
        latent_channels=4,
        n_latent_tokens=162,
        condition_channels=20,
        width=132,
        n_forcings=6,
        depth=4,
        n_heads=4,
        ffn_ratio=1.55,
        diffusion_steps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        time_embedding_channels=64,
        condition_on_co2=False,
        co2_mean=0.0,
        co2_std=1.0,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.n_latent_tokens = n_latent_tokens
        self.condition_channels = condition_channels
        self.hidden_channels = width
        self.n_forcings = n_forcings
        self.n_layers = depth
        self.diffusion_steps = diffusion_steps
        self.time_embedding_channels = time_embedding_channels
        self.condition_on_co2 = bool(condition_on_co2)
        self.co2_mean, self.co2_std = float(co2_mean), float(co2_std)
        if (
            not math.isfinite(self.co2_mean)
            or not math.isfinite(self.co2_std)
            or self.co2_std <= 0
        ):
            raise ValueError("CO2 normalization needs a finite mean and positive std")

        self.input_proj = nn.Linear(latent_channels, width)
        self.context_proj = nn.Linear(condition_channels, width)
        self.token_identity = nn.Parameter(torch.randn(n_latent_tokens, width) * 0.02)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_channels, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.forcing_mlp = nn.Sequential(
            nn.Linear(n_forcings, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(width, n_heads, ffn_ratio, width) for _ in range(depth)]
        )
        self.output_norm = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, latent_channels)
        if self.condition_on_co2:
            self.co2_mlp = nn.Sequential(
                nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width)
            )
            # A learned missing-condition embedding, distinct from mean CO2.
            self.null_co2 = nn.Parameter(torch.zeros(width))

        betas = torch.linspace(beta_start, beta_end, diffusion_steps)
        alpha_bar = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("alpha", torch.sqrt(alpha_bar), persistent=False)
        self.register_buffer("sigma", torch.sqrt(1.0 - alpha_bar), persistent=False)

    def forward(self, z_noisy, f, z_condition, t, *, co2=None, co2_keep=None):
        batch_size = z_noisy.shape[0]
        self.check_latent(z_noisy, self.latent_channels, "noisy latent")
        f = self.prepare_forcings(f, batch_size, z_noisy)
        context = self.prepare_condition(z_condition, batch_size, z_noisy)
        t = self.prepare_timestep(t, batch_size, z_noisy.device)
        identity = self.token_identity.to(dtype=z_noisy.dtype).unsqueeze(0)
        h = self.input_proj(z_noisy) + identity
        context = self.context_proj(context) + identity
        cond = self.time_mlp(self.timestep_embedding(t, z_noisy)) + self.forcing_mlp(f)
        if self.condition_on_co2:
            cond = cond + self.co2_embedding(co2, co2_keep, z_noisy)
        for block in self.blocks:
            h = block(h, context, cond)
        return self.output_proj(self.output_norm(h))

    def co2_embedding(self, co2, keep, like):
        """Accept ppm; use saved training statistics, or the learned null token."""
        batch = like.shape[0]
        missing = self.null_co2.to(like).expand(batch, -1)
        if co2 is None:
            return missing
        ppm = torch.as_tensor(co2, device=like.device, dtype=like.dtype)
        if ppm.ndim == 0:
            ppm = ppm.expand(batch)
        if ppm.shape == (batch,):
            ppm = ppm[:, None]
        if ppm.shape != (batch, 1):
            raise ValueError("CO2 must be a scalar or one ppm value per sample")
        embedded = self.co2_mlp((ppm - self.co2_mean) / self.co2_std)
        if keep is None:
            return embedded
        if keep.shape != (batch,) or keep.dtype != torch.bool:
            raise ValueError("CO2 keep mask must be boolean with shape [batch]")
        return torch.where(keep.to(like.device)[:, None], embedded, missing)

    def predict_v(
        self,
        noisy,
        calendar,
        history,
        t,
        *,
        co2=None,
        co2_keep=None,
        guidance_scale=1.0,
    ):
        """CFG on v predictions: null + scale * (conditional - null).

        Both branches see identical noisy states, calendar, history and timestep.
        Scale 1 makes one conditional call; scale 0 makes one null call.
        """
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and nonnegative")
        if not self.condition_on_co2:
            if guidance_scale != 1:
                raise ValueError(
                    "CO2 guidance requires a CO2-conditioned diffusion model"
                )
            return self(noisy, calendar, history, t)
        if guidance_scale == 0:
            return self(noisy, calendar, history, t, co2=None)
        if co2 is None:
            raise ValueError("Supply target-month CO2 in ppm for conditional diffusion")
        conditional = self(noisy, calendar, history, t, co2=co2, co2_keep=co2_keep)
        if guidance_scale == 1:
            return conditional
        if co2_keep is not None:
            raise ValueError("Training dropout masks cannot be combined with CFG")
        unconditional = self(noisy, calendar, history, t, co2=None)
        return unconditional + guidance_scale * (conditional - unconditional)

    def prepare_forcings(self, f, batch_size, like):
        if self.n_forcings == 0:
            return like.new_empty(batch_size, 0)
        if f is None:
            raise AssertionError("forcing tensor is required")
        if f.ndim == 1:
            f = f.unsqueeze(0).expand(batch_size, -1)
        if f.shape != (batch_size, self.n_forcings):
            raise AssertionError("forcing tensor has incorrect shape")
        return f.to(device=like.device, dtype=like.dtype)

    def prepare_condition(self, z_condition, batch_size, like):
        if self.condition_channels == 0:
            return like.new_zeros(batch_size, self.n_latent_tokens, 0)
        if z_condition is None:
            raise AssertionError("conditioning latent is required")
        self.check_latent(z_condition, self.condition_channels, "conditioning latent")
        if z_condition.shape[0] != batch_size:
            raise AssertionError("conditioning latent has incorrect batch size")
        return z_condition.to(device=like.device, dtype=like.dtype)

    def prepare_timestep(self, t, batch_size, device):
        if t is None:
            return torch.randint(self.diffusion_steps, (batch_size,), device=device)
        t = torch.as_tensor(t, dtype=torch.long, device=device)
        if t.ndim == 0:
            t = t.expand(batch_size)
        if t.shape != (batch_size,):
            raise AssertionError("diffusion timestep has incorrect shape")
        return t

    def schedule(self, t, like):
        alpha = self.alpha.to(device=like.device, dtype=like.dtype)[t].view(
            like.shape[0], 1, 1
        )
        sigma = self.sigma.to(device=like.device, dtype=like.dtype)[t].view(
            like.shape[0], 1, 1
        )
        return alpha, sigma

    def timestep_embedding(self, t, like):
        half = (self.time_embedding_channels + 1) // 2
        freq = torch.arange(half, device=like.device, dtype=like.dtype)
        freq = torch.exp(-math.log(10_000.0) * freq / max(half - 1, 1))
        phase = t.to(dtype=like.dtype)[:, None] * freq[None, :]
        return torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)[
            :, : self.time_embedding_channels
        ]

    def check_latent(self, z, channels, name):
        if z.shape[-2:] != (self.n_latent_tokens, channels):
            raise AssertionError(f"{name} has incorrect shape")


class SphericalTransformer(nn.Module):
    """Atomic full-attention stack on spherical tokens, without a noise schedule.

    Inputs are already embedded [batch,node,width] tensors and a [batch,width]
    conditioning vector. SphericalDiT uses the same DiTBlock building block.
    """

    def __init__(self, width, depth=4, n_heads=4, ffn_ratio=1.55):
        super().__init__()
        self.blocks = nn.ModuleList(
            DiTBlock(width, n_heads, ffn_ratio, width) for _ in range(depth)
        )

    def forward(self, tokens, context, condition):
        for block in self.blocks:
            tokens = block(tokens, context, condition)
        return tokens


class Coupling(str, Enum):
    JACOBI = "jacobi"  # contemporaneous noisy latents, synchronous denoising
    ATMOSPHERE_FIRST = "atmosphere_first"  # generated clean atmosphere -> ocean


def ar1_noise(like, previous=None, *, tau_months, innovation=None):
    """Stationary unit-variance AR(1); tau=0 is independent white noise."""
    if tau_months < 0:
        raise ValueError("Noise timescale cannot be negative")
    innovation = torch.randn_like(like) if innovation is None else innovation
    if previous is None:
        return innovation
    alpha = math.exp(-1.0 / tau_months) if tau_months else 0.0
    return alpha * previous + math.sqrt(1 - alpha**2) * innovation


TIMESTEPS = (999, 674, 373, 120, 29, 6, 0)


class LatentDiffusion(SphericalDiT):
    def diffuse(self, z, t=None, noise=None):
        t = self.prepare_timestep(t, z.shape[0], z.device)
        noise = torch.randn_like(z) if noise is None else noise
        alpha, sigma = self.schedule(t, z)
        return alpha * z + sigma * noise, alpha * noise - sigma * z, t

    def denoise(
        self,
        calendar,
        history,
        noise,
        *,
        timesteps=TIMESTEPS,
        co2=None,
        co2_keep=None,
        guidance_scale=1.0,
    ):
        """Keep autograd through all denoising steps; callers detach deliberately."""
        if (
            timesteps[0] != self.diffusion_steps - 1
            or timesteps[-1] != 0
            or any(a <= b for a, b in zip(timesteps[:-1], timesteps[1:]))
        ):
            raise ValueError(
                "Descending sampler grid must include both diffusion endpoints"
            )
        grid = torch.tensor(timesteps, device=noise.device)
        z, previous_data, previous_step = noise, None, None
        for i, step in enumerate(grid):
            t = step.expand(z.shape[0])
            v = self.predict_v(
                z,
                calendar,
                history,
                t,
                co2=co2,
                co2_keep=co2_keep,
                guidance_scale=guidance_scale,
            )
            alpha, sigma = self.schedule(t, z)
            data = alpha * z - sigma * v
            if i == len(grid) - 1:
                return data
            z = _advance(
                z,
                data,
                previous_data,
                head=self,
                step=step,
                next_step=grid[i + 1],
                previous_step=previous_step,
                method="dpmpp2m_midpoint",
            )
            previous_data, previous_step = data, step

    @torch.no_grad()
    def sample(
        self,
        calendar,
        history,
        noise,
        *,
        timesteps=TIMESTEPS,
        co2=None,
        guidance_scale=1.0,
    ):
        return self.denoise(
            calendar,
            history,
            noise,
            timesteps=timesteps,
            co2=co2,
            guidance_scale=guidance_scale,
        )


class CoupledLatentDiffusion(nn.Module):
    def __init__(
        self,
        *,
        atmosphere_latent_channels=4,
        ocean_latent_channels=4,
        n_latent_tokens=162,
        width=132,
        depth=4,
        n_heads=4,
        ffn_ratio=1.55,
        time_embedding_channels=64,
        atmosphere_tau=0.5,
        ocean_tau=6.0,
        coupling="jacobi",
        cross_gradients=False,
        condition_on_co2=False,
        co2_dropout=0.0,
        co2_mean=0.0,
        co2_std=1.0,
    ):
        super().__init__()
        self.coupling = Coupling(coupling)
        self.cross_gradients = cross_gradients
        self.condition_on_co2 = bool(condition_on_co2)
        self.co2_dropout = float(co2_dropout)
        if not 0 <= self.co2_dropout < 1:
            raise ValueError("co2_dropout must be in [0, 1)")
        if self.co2_dropout and not self.condition_on_co2:
            raise ValueError("CO2 dropout requires CO2 conditioning")
        self.ffn_ratio = ffn_ratio
        self.atmosphere_latent_channels = atmosphere_latent_channels
        self.ocean_latent_channels = ocean_latent_channels
        self.atmosphere_tau, self.ocean_tau = atmosphere_tau, ocean_tau
        a, o = atmosphere_latent_channels, ocean_latent_channels
        self.atmosphere = LatentDiffusion(
            latent_channels=a,
            condition_channels=2 * a
            + 2 * o
            + (o if self.coupling == Coupling.JACOBI else 0),
            n_latent_tokens=n_latent_tokens,
            width=width,
            depth=depth,
            n_heads=n_heads,
            ffn_ratio=ffn_ratio,
            time_embedding_channels=time_embedding_channels,
            condition_on_co2=condition_on_co2,
            co2_mean=co2_mean,
            co2_std=co2_std,
        )
        self.ocean = LatentDiffusion(
            latent_channels=o,
            condition_channels=2 * o + 3 * a,
            n_latent_tokens=n_latent_tokens,
            width=width,
            depth=depth,
            n_heads=n_heads,
            ffn_ratio=ffn_ratio,
            time_embedding_channels=time_embedding_channels,
            condition_on_co2=condition_on_co2,
            co2_mean=co2_mean,
            co2_std=co2_std,
        )
        self.n_latent_tokens = n_latent_tokens
        self.diffusion_steps = self.atmosphere.diffusion_steps

    def co2_keep_mask(self, batch_size, device):
        """One dropout decision per sequence, shared by both heads and all steps."""
        if not self.condition_on_co2:
            return None
        if self.training and self.co2_dropout:
            return torch.rand(batch_size, device=device) >= self.co2_dropout
        return torch.ones(batch_size, device=device, dtype=torch.bool)

    def _atmosphere_context(self, atmosphere_history, ocean_history, noisy_ocean=None):
        if not self.cross_gradients:
            ocean_history = ocean_history.detach()
            noisy_ocean = None if noisy_ocean is None else noisy_ocean.detach()
        values = (
            (atmosphere_history, ocean_history, noisy_ocean)
            if self.coupling == Coupling.JACOBI
            else (atmosphere_history, ocean_history)
        )
        return torch.cat(values, -1)

    def _ocean_context(self, ocean_history, atmosphere_history, atmosphere):
        if not self.cross_gradients:
            atmosphere_history, atmosphere = (
                atmosphere_history.detach(),
                atmosphere.detach(),
            )
        return torch.cat((ocean_history, atmosphere_history, atmosphere), -1)

    def terminal_noise(self, atmosphere_history, ocean_history, previous=(None, None)):
        atmosphere = ar1_noise(
            atmosphere_history[..., : self.atmosphere_latent_channels],
            previous[0],
            tau_months=self.atmosphere_tau,
        )
        ocean = ar1_noise(
            ocean_history[..., : self.ocean_latent_channels],
            previous[1],
            tau_months=self.ocean_tau,
        )
        return atmosphere, ocean

    def denoise(
        self,
        calendar,
        atmosphere_history,
        ocean_history,
        *,
        previous_noise=(None, None),
        noise=None,
        co2=None,
        co2_keep=None,
        guidance_scale=1.0,
    ):
        """Generate both next-month latents; keep gradients through solver steps."""
        atmosphere_noise, ocean_noise = (
            self.terminal_noise(atmosphere_history, ocean_history, previous_noise)
            if noise is None
            else noise
        )
        if self.coupling == Coupling.JACOBI:
            atmosphere, ocean = denoise_paired_with_timesteps(
                self,
                calendar,
                atmosphere_history,
                ocean_history,
                timesteps=torch.tensor(TIMESTEPS, device=atmosphere_history.device),
                method="dpmpp2m_midpoint",
                atmosphere_noise=atmosphere_noise,
                ocean_noise=ocean_noise,
                co2=co2,
                co2_keep=co2_keep,
                guidance_scale=guidance_scale,
            )
        else:
            atmosphere = self.atmosphere.denoise(
                calendar,
                self._atmosphere_context(atmosphere_history, ocean_history),
                atmosphere_noise,
                co2=co2,
                co2_keep=co2_keep,
                guidance_scale=guidance_scale,
            )
            ocean = self.ocean.denoise(
                calendar,
                self._ocean_context(ocean_history, atmosphere_history, atmosphere),
                ocean_noise,
                co2=co2,
                co2_keep=co2_keep,
                guidance_scale=guidance_scale,
            )
        return atmosphere, ocean, atmosphere_noise, ocean_noise

    @torch.no_grad()
    def sample(
        self,
        calendar,
        atmosphere_history,
        ocean_history,
        *,
        previous_noise=(None, None),
        noise=None,
        co2=None,
        guidance_scale=1.0,
    ):
        return self.denoise(
            calendar,
            atmosphere_history,
            ocean_history,
            previous_noise=previous_noise,
            noise=noise,
            co2=co2,
            guidance_scale=guidance_scale,
        )


SamplerMethod = Literal["ddim", "dpmpp2m_midpoint", "dpmpp2m_heun"]
TimestepSpacing = Literal["t", "theta", "logsnr"]


def _warped_progress(nfe: int, rho: float, *, device: torch.device) -> torch.Tensor:
    if nfe < 2:
        raise ValueError("nfe must be at least two")
    if not math.isfinite(rho) or rho <= 0.0:
        raise ValueError("rho must be positive and finite")
    return torch.linspace(0.0, 1.0, nfe, device=device, dtype=torch.float64).pow(rho)


def make_timesteps(
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    *,
    nfe: int,
    spacing: TimestepSpacing = "t",
    rho: float = 1.0,
) -> torch.Tensor:
    """Return a strictly descending grid from the noisiest to cleanest index.

    ``rho`` warps progress from the noisiest endpoint. Values above one devote
    more evaluations to the high-noise end; values below one devote more to
    the low-noise end. The returned grid always contains both endpoint indices.
    """

    if alpha.ndim != 1 or sigma.ndim != 1 or alpha.shape != sigma.shape:
        raise ValueError("alpha and sigma must be same-length one-dimensional tensors")
    if nfe > alpha.numel():
        raise ValueError("nfe cannot exceed the number of diffusion timesteps")
    # Timestep selection is discrete and outside the learned computation. Do
    # its higher-precision arithmetic on CPU: MPS intentionally has no float64
    # tensor support, and casting before transferring would fail there.
    target_device = alpha.device
    cpu = torch.device("cpu")
    progress = _warped_progress(nfe, rho, device=cpu)
    alpha64 = alpha.detach().cpu().to(dtype=torch.float64)
    sigma64 = sigma.detach().cpu().to(dtype=torch.float64)
    if spacing == "t":
        # Match the prototypes' existing linspace(...).round() definition
        # exactly, including PyTorch's half-to-even behavior.
        steps = ((alpha.numel() - 1) * (1.0 - progress)).round().to(dtype=torch.long)
        steps[0] = alpha.numel() - 1
        steps[-1] = 0
        if not bool(torch.all(steps[:-1] > steps[1:])):
            raise ValueError(
                f"{spacing} spacing with nfe={nfe} and rho={rho:g} produced duplicate "
                "or non-descending timesteps"
            )
        return steps.to(device=target_device)
    if spacing == "theta":
        coordinate = torch.atan2(sigma64, alpha64)
    elif spacing == "logsnr":
        coordinate = torch.log(alpha64.clamp_min(1.0e-30)) - torch.log(
            sigma64.clamp_min(1.0e-30)
        )
    else:
        raise ValueError(f"Unknown timestep spacing: {spacing!r}")

    target = coordinate[-1] + progress * (coordinate[0] - coordinate[-1])
    distance = (coordinate[:, None] - target[None, :]).abs()
    steps = distance.argmin(dim=0).to(dtype=torch.long)
    steps[0] = alpha.numel() - 1
    steps[-1] = 0
    if not bool(torch.all(steps[:-1] > steps[1:])):
        raise ValueError(
            f"{spacing} spacing with nfe={nfe} and rho={rho:g} produced duplicate "
            "or non-descending timesteps"
        )
    return steps.to(device=target_device)


def _schedule_value(
    head, step: torch.Tensor, like: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    timestep = step.expand(like.shape[0])
    return head.schedule(timestep, like)


def _lambda(alpha: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    tiny = torch.finfo(alpha.dtype).tiny
    return torch.log(alpha.clamp_min(tiny)) - torch.log(sigma.clamp_min(tiny))


def _first_order_update(
    sample: torch.Tensor,
    data_prediction: torch.Tensor,
    *,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    alpha_next: torch.Tensor,
    sigma_next: torch.Tensor,
) -> torch.Tensor:
    noise_prediction = (sample - alpha * data_prediction) / sigma.clamp_min(
        torch.finfo(sample.dtype).tiny
    )
    return alpha_next * data_prediction + sigma_next * noise_prediction


def _dpmpp2m_update(
    sample: torch.Tensor,
    data_prediction: torch.Tensor,
    previous_data_prediction: torch.Tensor | None,
    *,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    alpha_next: torch.Tensor,
    sigma_next: torch.Tensor,
    previous_alpha: torch.Tensor | None,
    previous_sigma: torch.Tensor | None,
    solver_type: Literal["midpoint", "heun"],
) -> torch.Tensor:
    """Apply one DPM-Solver++(2M) data-prediction update."""

    if previous_data_prediction is None:
        return _first_order_update(
            sample,
            data_prediction,
            alpha=alpha,
            sigma=sigma,
            alpha_next=alpha_next,
            sigma_next=sigma_next,
        )
    if previous_alpha is None or previous_sigma is None:
        raise ValueError("the previous schedule values are required for a 2M update")
    lambda_current = _lambda(alpha, sigma)
    lambda_next = _lambda(alpha_next, sigma_next)
    lambda_previous = _lambda(previous_alpha, previous_sigma)
    h = lambda_next - lambda_current
    h_previous = lambda_current - lambda_previous
    ratio = h_previous / h
    first_difference = (data_prediction - previous_data_prediction) / ratio
    expm1_negative_h = torch.expm1(-h)
    updated = (
        sigma_next / sigma
    ) * sample - alpha_next * expm1_negative_h * data_prediction
    if solver_type == "midpoint":
        return updated - 0.5 * alpha_next * expm1_negative_h * first_difference
    if solver_type == "heun":
        return updated + alpha_next * (expm1_negative_h / h + 1.0) * first_difference
    raise ValueError(f"Unknown DPM-Solver++ solver type: {solver_type!r}")


def _advance(
    sample: torch.Tensor,
    data_prediction: torch.Tensor,
    previous_data_prediction: torch.Tensor | None,
    *,
    head,
    step: torch.Tensor,
    next_step: torch.Tensor,
    previous_step: torch.Tensor | None,
    method: SamplerMethod,
) -> torch.Tensor:
    alpha, sigma = _schedule_value(head, step, sample)
    alpha_next, sigma_next = _schedule_value(head, next_step, sample)
    if method == "ddim":
        return _first_order_update(
            sample,
            data_prediction,
            alpha=alpha,
            sigma=sigma,
            alpha_next=alpha_next,
            sigma_next=sigma_next,
        )
    previous_alpha = previous_sigma = None
    if previous_step is not None:
        previous_alpha, previous_sigma = _schedule_value(head, previous_step, sample)
    return _dpmpp2m_update(
        sample,
        data_prediction,
        previous_data_prediction,
        alpha=alpha,
        sigma=sigma,
        alpha_next=alpha_next,
        sigma_next=sigma_next,
        previous_alpha=previous_alpha,
        previous_sigma=previous_sigma,
        solver_type="midpoint" if method == "dpmpp2m_midpoint" else "heun",
    )


def denoise_paired_with_timesteps(
    transition,
    forcings: torch.Tensor,
    atmosphere_history: torch.Tensor,
    ocean_history: torch.Tensor,
    *,
    timesteps: torch.Tensor,
    method: SamplerMethod,
    atmosphere_noise: torch.Tensor,
    ocean_noise: torch.Tensor,
    co2=None,
    co2_keep=None,
    guidance_scale=1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Denoise paired heads from caller-constructed terminal-noise states.

    MDV1 rebuilds each head's context at every reverse-time step, so an
    atmospheric score sees the current ocean noisy state and vice versa.
    Terminal-noise construction remains outside this helper and the two AR(1)
    innovations are still independent.
    """

    if timesteps.ndim != 1 or timesteps.numel() < 2:
        raise ValueError(
            "timesteps must be a one-dimensional grid with at least two entries"
        )
    if not bool(torch.all(timesteps[:-1] > timesteps[1:])):
        raise ValueError("timesteps must be strictly descending")
    if int(timesteps[0]) != transition.diffusion_steps - 1 or int(timesteps[-1]) != 0:
        raise ValueError("timesteps must include both diffusion endpoints")
    if method not in {"ddim", "dpmpp2m_midpoint", "dpmpp2m_heun"}:
        raise ValueError(f"Unknown sampler method: {method!r}")

    batch_size = atmosphere_history.shape[0]
    expected_atmosphere_shape = (
        batch_size,
        transition.n_latent_tokens,
        transition.atmosphere_latent_channels,
    )
    expected_ocean_shape = (
        batch_size,
        transition.n_latent_tokens,
        transition.ocean_latent_channels,
    )
    if atmosphere_noise.shape != expected_atmosphere_shape:
        raise ValueError(
            "atmosphere terminal noise has shape "
            f"{tuple(atmosphere_noise.shape)}, expected {expected_atmosphere_shape}"
        )
    if ocean_noise.shape != expected_ocean_shape:
        raise ValueError(
            "ocean terminal noise has shape "
            f"{tuple(ocean_noise.shape)}, expected {expected_ocean_shape}"
        )

    atmosphere, ocean = atmosphere_noise, ocean_noise
    timesteps = timesteps.to(device=atmosphere.device)
    previous_atmosphere_data = previous_ocean_data = None
    previous_step = None

    for index, step in enumerate(timesteps):
        timestep = step.expand(batch_size)
        atmosphere_v = transition.atmosphere.predict_v(
            atmosphere,
            forcings,
            transition._atmosphere_context(atmosphere_history, ocean_history, ocean),
            timestep,
            co2=co2,
            co2_keep=co2_keep,
            guidance_scale=guidance_scale,
        )
        ocean_v = transition.ocean.predict_v(
            ocean,
            forcings,
            transition._ocean_context(ocean_history, atmosphere_history, atmosphere),
            timestep,
            co2=co2,
            co2_keep=co2_keep,
            guidance_scale=guidance_scale,
        )
        atmosphere_alpha, atmosphere_sigma = transition.atmosphere.schedule(
            timestep, atmosphere
        )
        ocean_alpha, ocean_sigma = transition.ocean.schedule(timestep, ocean)
        atmosphere_data = (
            atmosphere_alpha * atmosphere - atmosphere_sigma * atmosphere_v
        )
        ocean_data = ocean_alpha * ocean - ocean_sigma * ocean_v
        if index == timesteps.numel() - 1:
            return atmosphere_data, ocean_data
        next_step = timesteps[index + 1]
        atmosphere = _advance(
            atmosphere,
            atmosphere_data,
            previous_atmosphere_data,
            head=transition.atmosphere,
            step=step,
            next_step=next_step,
            previous_step=previous_step,
            method=method,
        )
        ocean = _advance(
            ocean,
            ocean_data,
            previous_ocean_data,
            head=transition.ocean,
            step=step,
            next_step=next_step,
            previous_step=previous_step,
            method=method,
        )
        previous_atmosphere_data = atmosphere_data
        previous_ocean_data = ocean_data
        previous_step = step
    raise AssertionError("unreachable sampler exit")
