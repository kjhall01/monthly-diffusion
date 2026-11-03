"""Full-attention spherical-token DiT; geometry enters learned token identities.

No geographic edge features or relative-position bias enter these attention blocks.
"""

from dataclasses import dataclass
import math
import torch
from torch import nn
from .layers.attention import FeedForward, FiLM


@dataclass(frozen=True)
class DenoiserConfig:
    latent_channels: int = 4
    n_latent_tokens: int = 162
    condition_channels: int = 20
    width: int = 132
    n_forcings: int = 6
    depth: int = 4
    n_heads: int = 4
    ffn_ratio: float = 1.55
    time_embedding_channels: int = 64

    @property
    def latent_mesh_size(self) -> int:
        return self.n_latent_tokens

    @property
    def hidden_channels(self) -> int:
        return self.width

    @property
    def n_layers(self) -> int:
        return self.depth


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

    def forward(self, x: torch.Tensor, context: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
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
    """Neural denoiser only. Caller supplies diffusion time, noise and targets.

    z_noisy: [B,N,C]; f: [B,F]; z_condition: [B,N,H]; t: scalar or [B].
    Output: [B,N,C]. No particular epsilon/v/x0 parameterization is imposed.
    f is a generic conditioning vector, not necessarily physical forcing."""

    def __init__(self, config: DenoiserConfig):
        super().__init__()
        self.config = config
        self.latent_channels = config.latent_channels
        self.n_latent_tokens = config.n_latent_tokens
        self.latent_mesh_size = config.n_latent_tokens  # compatibility attribute only
        self.condition_channels = config.condition_channels
        self.hidden_channels = config.width
        self.n_forcings = config.n_forcings
        self.n_layers = config.depth
        self.time_embedding_channels = config.time_embedding_channels

        self.input_proj = nn.Linear(config.latent_channels, config.width)
        self.context_proj = nn.Linear(config.condition_channels, config.width)
        self.token_identity = nn.Parameter(torch.randn(config.n_latent_tokens, config.width) * 0.02)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_embedding_channels, config.width),
            nn.SiLU(),
            nn.Linear(config.width, config.width),
        )
        self.forcing_mlp = nn.Sequential(
            nn.Linear(config.n_forcings, config.width),
            nn.SiLU(),
            nn.Linear(config.width, config.width),
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(config.width, config.n_heads, config.ffn_ratio, config.width)
                for _ in range(config.depth)
            ]
        )
        self.output_norm = nn.LayerNorm(config.width)
        self.output_proj = nn.Linear(config.width, config.latent_channels)

    def forward(self, z_noisy, f, z_condition, t):
        batch_size = z_noisy.shape[0]
        self.check_latent(z_noisy, self.latent_channels, "noisy latent")
        f = self.prepare_forcings(f, batch_size, z_noisy)
        context = self.prepare_condition(z_condition, batch_size, z_noisy)
        t = self.prepare_timestep(t, batch_size, z_noisy.device)
        identity = self.token_identity.to(dtype=z_noisy.dtype).unsqueeze(0)
        h = self.input_proj(z_noisy) + identity
        context = self.context_proj(context) + identity
        cond = self.time_mlp(self.timestep_embedding(t, z_noisy)) + self.forcing_mlp(f)
        for block in self.blocks:
            h = block(h, context, cond)
        return self.output_proj(self.output_norm(h))

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
            raise ValueError("Pass diffusion time explicitly; sampling belongs outside architecture")
        t = torch.as_tensor(t, dtype=torch.float32, device=device)
        if t.ndim == 0:
            t = t.expand(batch_size)
        if t.shape != (batch_size,):
            raise AssertionError("diffusion timestep has incorrect shape")
        return t

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
