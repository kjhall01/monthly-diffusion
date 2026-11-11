"""PyTorch operators over ``SphericalMap`` connectivity."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from ._geometry.graph import SphericalMap
from ._geometry.runtime import SphericalMapTensorCache


class MLP(nn.Module):
    """Small feed-forward network used by spherical operators."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0 or hidden_dim <= 0:
            raise ValueError("in_dim, out_dim, and hidden_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        layers: list[nn.Module] = []
        if num_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation())
            layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DynamicSphericalConv(nn.Module):


    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_attr_dim: int,
        hidden_dim: int,
        kernel_channels: Literal["scalar", "per_channel"] = "per_channel",
        aggregation: Literal["sum", "mean"] = "mean",
        residual: bool = True,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()


        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.edge_attr_dim = int(edge_attr_dim)
        self.hidden_dim = int(hidden_dim)
        self.kernel_channels = kernel_channels
        self.aggregation = aggregation
        self.residual = bool(residual)
        self._map_cache = SphericalMapTensorCache()

        kernel_dim = 1 if kernel_channels == "scalar" else self.out_dim
        self.source_proj = nn.Linear(self.in_dim, self.out_dim)
        self.kernel_mlp = MLP(
            in_dim=self.edge_attr_dim,
            out_dim=kernel_dim,
            hidden_dim=self.hidden_dim,
        )
        self.out_proj = nn.Linear(self.out_dim, self.out_dim)


    def forward(
        self,
        source_nodes: torch.Tensor,
        target_nodes: torch.Tensor | None,
        graph_map: SphericalMap,
    ) -> torch.Tensor:

        batch, n_source, source_dim = source_nodes.shape
        if target_nodes is None:
            n_target = graph_map.n_target
        else:
            target_batch, n_target, target_dim = target_nodes.shape

        runtime = self._map_cache.get(
            graph_map,
            device=source_nodes.device,
            dtype=source_nodes.dtype,
        )
        sender = runtime.sender
        receiver = runtime.receiver
        edge_attr = runtime.edge_attr

        projected = self.source_proj(source_nodes)
        source_edge = projected[:, sender, :]
        kernel = self.kernel_mlp(edge_attr)
        messages = source_edge * kernel.unsqueeze(0)

        aggregated = torch.zeros(
            (batch, n_target, self.out_dim),
            dtype=messages.dtype,
            device=messages.device,
        )
        receiver_index = receiver.view(1, -1, 1).expand(batch, -1, self.out_dim)
        aggregated.scatter_add_(dim=1, index=receiver_index, src=messages)

        if self.aggregation == "mean":
            counts = torch.zeros((n_target,), dtype=messages.dtype, device=messages.device)
            counts.scatter_add_(0, receiver, torch.ones_like(receiver, dtype=messages.dtype))
            aggregated = aggregated / counts.clamp_min(1.0).view(1, -1, 1)

        update = self.out_proj(aggregated)
        if self.residual:
            if target_nodes is None:
                raise ValueError("target_nodes is required when residual=True.")
            return target_nodes + update
        return update

    def clear_cache(self) -> None:
        self._map_cache.clear()
