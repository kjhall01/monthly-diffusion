"""Factorized spatial/month FiLM and post-SiLU additive CO2 conditioning."""

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


class StaticFiLM(nn.Module):
    """P107's spatial FiLM driven only by collocated LSM and topography."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, width),
            nn.SiLU(),
            nn.Linear(width, 2 * width),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, nodes: torch.Tensor, statics: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(statics).chunk(2, dim=-1)
        return nodes * (1.0 + gamma) + beta


class TimeFiLM(nn.Module):
    """Channel-wise month FiLM plus a separate linear CO2 shift."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.month = nn.Sequential(
            nn.Linear(2, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.co2_shift = nn.Linear(1, width, bias=False)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        nn.init.zeros_(self.co2_shift.weight)

    def affine_terms(
        self,
        month: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the channel-only month scale and shift."""

        gamma, beta = self.net(self.month(month)).chunk(2, dim=-1)
        return gamma, beta

    def forward(
        self,
        nodes: torch.Tensor,
        month: torch.Tensor,
        co2: torch.Tensor,
    ) -> torch.Tensor:
        gamma, beta = self.affine_terms(month)
        return (
            nodes * (1.0 + gamma[:, None, :])
            + beta[:, None, :]
            + self.co2_shift(co2)[:, None, :]
        )


@dataclass(frozen=True)
class ConditioningTerms:
    """FiLM values fixed for one branch, date, and mesh level."""

    gamma_static: torch.Tensor
    beta_static: torch.Tensor
    gamma_time: torch.Tensor
    beta_time: torch.Tensor
    co2_shift: torch.Tensor


class ConditionalPreActivation(nn.Module):
    """Apply factorized context inside SiLU and CO2 after SiLU.

    For hidden state ``h[B,N,H]``, spatial statics produce node-by-channel
    scale and shift, while month produces channel-only scale and shift. The
    standardized raw-CO2 term is a separate channel-only vector broadcast over
    nodes and added after the activation::

        SiLU(
            LayerNorm(h) * (1 + gamma_static + gamma_month)
            + beta_static + beta_month
        )
        + CO2_standardized * w_co2

    Position remains in the ordinary learned position embeddings and does
    not enter this conditioner. Each component and mesh level owns a separate
    conditioner, exactly as in P107.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = int(width)
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.static = StaticFiLM(width)
        self.time = TimeFiLM(width)

    def modulation_terms(
        self,
        statics: torch.Tensor,
        month: torch.Tensor,
        co2: torch.Tensor,
        *,
        include_co2: bool = True,
    ) -> ConditioningTerms:
        """Return static/month affine fields and the optional CO2 shift."""

        if statics.ndim != 3 or statics.shape[-1] != 2:
            raise ValueError(
                "MDV1 conditioner statics must have shape [batch, node, 2], "
                f"got {tuple(statics.shape)}."
            )
        batch = statics.shape[0]
        if month.shape != (batch, 2):
            raise ValueError(
                f"MDV1 conditioner month must have shape [{batch}, 2], got "
                f"{tuple(month.shape)}."
            )
        if co2.shape != (batch, 1):
            raise ValueError(
                f"MDV1 conditioner CO2 must have shape [{batch}, 1], got "
                f"{tuple(co2.shape)}."
            )
        gamma_static, beta_static = self.static.net(statics).chunk(2, dim=-1)
        gamma_time, beta_time = self.time.affine_terms(month)
        co2_shift = (
            self.time.co2_shift(co2)
            if include_co2
            else torch.zeros(batch, self.width, device=co2.device, dtype=co2.dtype)
        )
        return ConditioningTerms(
            gamma_static=gamma_static,
            beta_static=beta_static,
            gamma_time=gamma_time,
            beta_time=beta_time,
            co2_shift=co2_shift,
        )

    def apply_terms(
        self, nodes: torch.Tensor, terms: ConditioningTerms
    ) -> torch.Tensor:
        """Apply cached context values to the evolving hidden state."""

        if nodes.shape != (
            terms.gamma_static.shape[0],
            terms.gamma_static.shape[1],
            self.width,
        ):
            raise ValueError("MDV1 conditioner nodes do not match cached FiLM terms.")
        conditioned = self.norm(nodes)
        conditioned = conditioned * (
            1.0 + terms.gamma_static + terms.gamma_time[:, None, :]
        )
        conditioned = conditioned + terms.beta_static + terms.beta_time[:, None, :]
        return F.silu(conditioned) + terms.co2_shift[:, None, :]

    def forward(
        self,
        nodes: torch.Tensor,
        statics: torch.Tensor,
        month: torch.Tensor,
        co2: torch.Tensor,
    ) -> torch.Tensor:
        if nodes.shape != (statics.shape[0], statics.shape[1], self.width):
            raise ValueError(
                "MDV1 conditioner nodes must have shape [batch, node, width], "
                f"got {tuple(nodes.shape)}."
            )
        return self.apply_terms(nodes, self.modulation_terms(statics, month, co2))


# Names expose the broadcasting geometry, without changing the inherited heads.
NodeChannelFiLM = StaticFiLM
ChannelFiLM = TimeFiLM


class AdditiveCO2(nn.Linear):
    """A learned channel-only CO2 direction, broadcast over all nodes."""

    def __init__(self, width):
        super().__init__(1, width, bias=False)
        nn.init.zeros_(self.weight)

    def forward(self, co2):
        return super().forward(co2)[:, None, :]
