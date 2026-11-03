"""Learned Cartesian position embedding (distinct from FiLM statics)."""

import torch
from torch import nn


class PositionEmbedding(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))

    def forward(
        self, xyz: torch.Tensor, batch: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        values = self.net(xyz.to(device=device, dtype=dtype))
        return values.unsqueeze(0).expand(batch, -1, -1)
