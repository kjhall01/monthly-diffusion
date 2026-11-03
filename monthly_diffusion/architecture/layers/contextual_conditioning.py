"""Joint static/seasonal FiLM followed by a channel-only linear CO2 shift."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .conditioning import AdditiveCO2


@dataclass(frozen=True)
class ContextualConditioningTerms:
    gamma: torch.Tensor
    beta: torch.Tensor
    co2_shift: torch.Tensor


class ContextualCO2PreActivation(nn.Module):
    """Pointwise contextual LayerNorm; neither XYZ nor CO2 enters the MLP.

    [standardized TOPO, LSM, standardized BATHY, sin(month), cos(month)]
    -> Linear(5,H) -> SiLU -> Linear(H,2H) -> gamma, beta [B,N,H].
    The same MLP is evaluated at every node. The static channel count is
    explicit, allowing a two-static-field version as well.

    Output = SiLU(LayerNorm(h) * (1 + gamma) + beta) + C * b.

    C is training-standardized raw CO2 [B,1]; b is a learned constant [H]
    vector. Its broadcast shift is [B,1,H], outside SiLU. Each hierarchy
    level/component owns its own parameters. All modulation starts at zero.
    """

    def __init__(self, width, static_channels=3):
        super().__init__()
        self.width, self.static_channels = int(width), int(static_channels)
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.net = nn.Sequential(
            nn.Linear(self.static_channels + 2, width), nn.SiLU(), nn.Linear(width, 2 * width)
        )
        self.co2_shift = AdditiveCO2(width)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def modulation_terms(self, statics, month, co2, *, include_co2=True):
        if statics.ndim != 3 or statics.shape[-1] != self.static_channels:
            raise ValueError(f"Statics must have shape [batch, node, {self.static_channels}]")
        batch, nodes, _ = statics.shape
        if month.shape != (batch, 2) or co2.shape != (batch, 1):
            raise ValueError("Require month [B,2] and CO2 [B,1]")
        context = torch.cat((statics, month[:, None, :].expand(-1, nodes, -1)), dim=-1)
        gamma, beta = self.net(context).chunk(2, dim=-1)
        shift = self.co2_shift(co2) if include_co2 else co2.new_zeros(batch, 1, self.width)
        return ContextualConditioningTerms(gamma, beta, shift)

    def apply_terms(self, nodes, terms):
        if nodes.shape != terms.gamma.shape or nodes.shape != terms.beta.shape:
            raise ValueError("Hidden nodes and contextual FiLM terms must have identical shapes")
        return F.silu(self.norm(nodes) * (1 + terms.gamma) + terms.beta) + terms.co2_shift

    def forward(self, nodes, statics, month, co2):
        return self.apply_terms(nodes, self.modulation_terms(statics, month, co2))


class XYZContextualCO2PreActivation(ContextualCO2PreActivation):
    """Pointwise XYZ/static/seasonal FiLM with a separate post-SiLU CO2 shift.

    [x, y, z, TOPO_scaled, LSM, BATHY_scaled, sin(month), cos(month)]
    -> Linear(8,H) -> SiLU -> Linear(H,2H) -> gamma, beta [B,N,H].
    XYZ are the Earth-fixed unit-sphere coordinates of this level's nodes;
    they are not pooled grid coordinates or learned per-node parameters.
    CO2 never enters this MLP. Application of the terms is identical to
    ContextualCO2PreActivation, including its channel-only post-SiLU shift.
    """

    def __init__(self, width, static_channels=3):
        # Construct only the requested MLP; no unused five-input initialization.
        nn.Module.__init__(self)
        self.width, self.static_channels = int(width), int(static_channels)
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.net = nn.Sequential(
            nn.Linear(self.static_channels + 5, width), nn.SiLU(), nn.Linear(width, 2 * width)
        )
        self.co2_shift = AdditiveCO2(width)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def modulation_terms(self, statics, month, co2, *, xyz, include_co2=True):
        if statics.ndim != 3 or statics.shape[-1] != self.static_channels:
            raise ValueError(f"Statics must have shape [batch, node, {self.static_channels}]")
        batch, nodes, _ = statics.shape
        if month.shape != (batch, 2) or co2.shape != (batch, 1) or xyz.shape != (nodes, 3):
            raise ValueError("Require month [B,2], CO2 [B,1], and XYZ [N,3]")
        context = torch.cat(
            (
                xyz.to(statics)[None].expand(batch, -1, -1),
                statics,
                month[:, None, :].expand(-1, nodes, -1),
            ),
            dim=-1,
        )
        gamma, beta = self.net(context).chunk(2, dim=-1)
        shift = self.co2_shift(co2) if include_co2 else co2.new_zeros(batch, 1, self.width)
        return ContextualConditioningTerms(gamma, beta, shift)

    def forward(self, nodes, statics, month, co2, *, xyz):
        return self.apply_terms(nodes, self.modulation_terms(statics, month, co2, xyz=xyz))
