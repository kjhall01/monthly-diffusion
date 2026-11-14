"""PaddedECC: one padded neighbor table for the entire spherical map.

Uses the same learned layers as ScatterECC and BucketedECC. Padding is compact
when destination nodes have similar neighbor counts, as on the icosphere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from monthly_diffusion._geometry.graph import SphericalMap


class _MLP(nn.Module):
    """Map geometric edge features to learned convolution weights."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


@dataclass(frozen=True)
class _PaddedInboundStencil:
    sender: torch.Tensor
    edge_attr: torch.Tensor
    valid: torch.Tensor
    inverse_degree: torch.Tensor


class _PaddedInboundStencilCache:
    """Cache padded inbound map stencils by map, device, and floating dtype."""

    def __init__(self) -> None:
        self._cache: dict[
            tuple[int, torch.device, torch.dtype], _PaddedInboundStencil
        ] = {}

    def get(
        self,
        graph_map: SphericalMap,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> _PaddedInboundStencil:
        key = (id(graph_map), torch.device(device), dtype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        edges = graph_map.edge_set
        sender = edges.sender.detach().cpu().to(torch.long)
        receiver = edges.receiver.detach().cpu().to(torch.long)
        edge_attr = edges.edge_attr.detach().cpu()
        n_target = int(graph_map.n_target)
        counts = torch.bincount(receiver, minlength=n_target)
        if counts.numel() != n_target:
            raise ValueError(
                f"PaddedECC map {graph_map.name!r} has invalid target counts."
            )
        degree = int(counts.max())
        order = torch.argsort(receiver)
        sorted_receiver = receiver[order]
        starts = torch.cumsum(counts, dim=0) - counts
        slots = torch.arange(sender.numel()) - starts[sorted_receiver]

        padded_sender = torch.zeros((n_target, degree), dtype=torch.long)
        padded_attr = torch.zeros(
            (n_target, degree, edge_attr.shape[-1]), dtype=edge_attr.dtype
        )
        valid = torch.zeros((n_target, degree, 1), dtype=edge_attr.dtype)
        padded_sender[sorted_receiver, slots] = sender[order]
        padded_attr[sorted_receiver, slots] = edge_attr[order]
        valid[sorted_receiver, slots] = 1.0
        cached = _PaddedInboundStencil(
            sender=padded_sender.to(device=device),
            edge_attr=padded_attr.to(device=device, dtype=dtype),
            valid=valid.to(device=device, dtype=dtype),
            # The scatter reference returns an all-zero aggregate for a
            # target with no incoming radius-map sources, then divides by one.
            # Preserve that behavior exactly for sparse grid-to-mesh maps.
            inverse_degree=counts.clamp_min(1)
            .to(device=device, dtype=dtype)
            .reciprocal()
            .view(1, n_target, 1),
        )
        self._cache[key] = cached
        return cached

    def clear(self) -> None:
        self._cache.clear()


class PaddedECC(nn.Module):
    """ECC using one cached, padded table of incoming neighbors."""

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
        mask_policy: str = "support",
    ) -> None:
        super().__init__()
        if kernel_channels not in {"scalar", "per_channel"}:
            raise ValueError("kernel_channels must be 'scalar' or 'per_channel'.")
        if aggregation not in {"sum", "mean"}:
            raise ValueError("aggregation must be 'sum' or 'mean'.")
        if mask_policy not in {"support", "legacy"}:
            raise ValueError("mask_policy must be 'support' or 'legacy'")
        self.mask_policy = mask_policy
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.edge_attr_dim = int(edge_attr_dim)
        self.hidden_dim = int(hidden_dim)
        self.kernel_channels = kernel_channels
        self.aggregation = aggregation
        self.residual = bool(residual)
        self._stencil_cache = _PaddedInboundStencilCache()
        kernel_dim = 1 if kernel_channels == "scalar" else self.out_dim
        self.source_proj = nn.Linear(self.in_dim, self.out_dim)
        self.kernel_mlp = _MLP(self.edge_attr_dim, kernel_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.out_dim, self.out_dim)

    def forward(
        self,
        source_nodes: torch.Tensor,
        target_nodes: torch.Tensor | None,
        graph_map: SphericalMap,
        *,
        source_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if source_nodes.ndim != 3:
            raise ValueError("source_nodes must have shape [batch, source, channel].")
        batch, n_source, channels = source_nodes.shape
        if channels != self.in_dim or n_source != graph_map.n_source:
            raise ValueError("PaddedECC source shape does not match its graph map.")
        if graph_map.edge_set.edge_attr_dim != self.edge_attr_dim:
            raise ValueError(
                "PaddedECC edge attribute dimension does not match its graph map."
            )
        n_target = graph_map.n_target if target_nodes is None else target_nodes.shape[1]
        if n_target != graph_map.n_target:
            raise ValueError("PaddedECC target shape does not match its graph map.")
        if self.residual and target_nodes is None:
            raise ValueError("target_nodes is required for a residual PaddedECC.")

        def node_mask(mask, expected):
            if mask is None:
                return None
            if mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask[..., 0]
            if mask.shape != (batch, expected):
                raise ValueError(
                    f"PaddedECC masks must have shape {(batch, expected)}, got {tuple(mask.shape)}"
                )
            return mask.to(device=source_nodes.device, dtype=source_nodes.dtype).clamp(
                0.0, 1.0
            )

        source_mask = node_mask(source_mask, n_source)
        target_mask = node_mask(target_mask, graph_map.n_target)
        if target_mask is not None and self.mask_policy == "support":
            # Fractions weight source averages; targets are simply wet or dry.
            # In particular, a partly wet node keeps its entire skip connection.
            target_mask = (target_mask > 0).to(source_nodes.dtype)

        stencil = self._stencil_cache.get(
            graph_map, device=source_nodes.device, dtype=source_nodes.dtype
        )
        projected = self.source_proj(source_nodes)
        kernels = self.kernel_mlp(stencil.edge_attr)
        gathered = projected[:, stencil.sender, :]
        edge_weight = stencil.valid.unsqueeze(0)
        if source_mask is not None:
            edge_weight = edge_weight * source_mask[:, stencil.sender, None]
        aggregated = (gathered * kernels.unsqueeze(0) * edge_weight).sum(dim=2)
        if self.aggregation == "mean":
            if source_mask is None:
                aggregated = aggregated * stencil.inverse_degree
            else:
                # A weighted mean divides by the actual wet support, even when
                # it totals less than one. Guard only the all-dry denominator.
                total_weight = edge_weight.sum(dim=2)
                if self.mask_policy == "legacy":
                    total_weight = total_weight.clamp_min(1.0)
                else:
                    total_weight = total_weight.masked_fill(total_weight == 0, 1.0)
                aggregated = aggregated / total_weight
        update = self.out_proj(aggregated)
        if target_mask is not None:
            update = update * target_mask[..., None]
        if self.residual:
            if target_mask is None:
                return target_nodes + update
            return target_nodes * target_mask[..., None] + update
        return update

    def clear_cache(self) -> None:
        self._stencil_cache.clear()
