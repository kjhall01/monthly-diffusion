"""MDV1 low-NFE sampler with live cross-component noisy-state conditioning.

The helpers in this module are deliberately independent of a particular
prototype.  They make the reverse-time grid explicit and provide the
first-order DDIM update used by the existing prototypes plus the one-evaluation
per-step DPM-Solver++(2M) multistep update.  All operations remain
differentiable so a selected method can later be used inside rollout training.
"""

from __future__ import annotations

import math
from typing import Literal

import torch


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


def _schedule_value(head, step: torch.Tensor, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        (sigma_next / sigma) * sample
        - alpha_next * expm1_negative_h * data_prediction
    )
    if solver_type == "midpoint":
        return updated - 0.5 * alpha_next * expm1_negative_h * first_difference
    if solver_type == "heun":
        return updated + alpha_next * (
            expm1_negative_h / h + 1.0
        ) * first_difference
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Denoise paired heads from caller-constructed terminal-noise states.

    MDV1 rebuilds each head's context at every reverse-time step, so an
    atmospheric score sees the current ocean noisy state and vice versa.
    Terminal-noise construction remains outside this helper and the two AR(1)
    innovations are still independent.
    """

    if timesteps.ndim != 1 or timesteps.numel() < 2:
        raise ValueError("timesteps must be a one-dimensional grid with at least two entries")
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
        transition.config.atmosphere_latent_channels,
    )
    expected_ocean_shape = (
        batch_size,
        transition.n_latent_tokens,
        transition.config.ocean_latent_channels,
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
        atmosphere_v = transition.atmosphere(
            atmosphere,
            forcings,
            transition._atmosphere_context(atmosphere_history, ocean_history, ocean),
            timestep,
        )
        ocean_v = transition.ocean(
            ocean,
            forcings,
            transition._ocean_context(ocean_history, atmosphere_history, atmosphere),
            timestep,
        )
        atmosphere_alpha, atmosphere_sigma = transition.atmosphere.schedule(
            timestep, atmosphere
        )
        ocean_alpha, ocean_sigma = transition.ocean.schedule(timestep, ocean)
        atmosphere_data = atmosphere_alpha * atmosphere - atmosphere_sigma * atmosphere_v
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
