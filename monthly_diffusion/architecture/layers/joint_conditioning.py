"""Pointwise joint spatial, seasonal, and CO2 conditional LayerNorm."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class JointConditioningTerms:
    gamma: torch.Tensor
    beta: torch.Tensor


class JointConditionalPreActivation(nn.Module):
    """A shared-across-nodes MLP, not a learned map of independent node weights.

    At every node, [x, y, z, standardized TOPO, LSM, sin(month), cos(month),
    standardized raw CO2] -> MLP(8, H, 2H) -> gamma, beta, each [B, N, H].
    Apply ``SiLU(LayerNorm(h) * (1 + gamma) + beta)``. XYZ are fixed unit-sphere
    coordinates, not a learned embedding. CO2 participates in both affine
    terms inside SiLU. Zero initialization starts with unmodulated LayerNorm.
    Each component and hierarchy level owns its own instance.
    """

    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.net = nn.Sequential(nn.Linear(8, width), nn.SiLU(), nn.Linear(width, 2 * width))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def modulation_terms(self, statics, month, co2, *, xyz, include_co2=True):
        if statics.ndim != 3 or statics.shape[-1] != 2:
            raise ValueError("Statics must have shape [batch, node, 2]")
        batch, nodes, _ = statics.shape
        if month.shape != (batch, 2) or co2.shape != (batch, 1) or xyz.shape != (nodes, 3):
            raise ValueError("Require month [B,2], CO2 [B,1], and XYZ [N,3]")
        if not include_co2:
            co2 = torch.zeros_like(co2)
        context = torch.cat(
            (
                xyz.to(statics)[None].expand(batch, -1, -1),
                statics,
                month[:, None, :].expand(-1, nodes, -1),
                co2[:, None, :].expand(-1, nodes, -1),
            ),
            dim=-1,
        )
        gamma, beta = self.net(context).chunk(2, dim=-1)
        return JointConditioningTerms(gamma, beta)

    def apply_terms(self, nodes, terms):
        if nodes.shape != terms.gamma.shape or nodes.shape != terms.beta.shape:
            raise ValueError("Hidden nodes and joint FiLM terms must have identical shapes")
        return F.silu(self.norm(nodes) * (1.0 + terms.gamma) + terms.beta)

    def forward(self, nodes, statics, month, co2, *, xyz):
        return self.apply_terms(nodes, self.modulation_terms(statics, month, co2, xyz=xyz))
