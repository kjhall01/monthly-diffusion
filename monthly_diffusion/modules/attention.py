"""Full-attention and channel-affine building blocks."""

import torch
import torch.nn as nn


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


class CrossAttentionBlock(nn.Module):
    def __init__(self, width: int, n_heads: int, ffn_ratio: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, n_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = FeedForward(width, ffn_ratio)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.query_norm(query)
        kv = self.context_norm(context)
        attended, _ = self.attention(q, kv, kv, need_weights=False)
        query = query + attended
        return query + self.ffn(self.ffn_norm(query))
