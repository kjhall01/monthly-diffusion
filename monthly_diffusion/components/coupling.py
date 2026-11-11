"""Two explicit coupling choices, independent of temporal noise persistence."""

from enum import Enum
import math
import torch


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
