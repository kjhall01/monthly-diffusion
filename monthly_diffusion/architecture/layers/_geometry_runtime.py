"""Runtime helpers for device-local spherical map tensors."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..geometry.graph import SphericalMap


@dataclass(frozen=True)
class RuntimeEdgeTensors:
    """Device-local tensors for a ``SphericalMap`` edge set."""

    sender: torch.Tensor
    receiver: torch.Tensor
    edge_attr: torch.Tensor


@dataclass(frozen=True)
class RuntimeStencilTensors:
    """Device-local fixed-in-degree stencil view of a ``SphericalMap``.

    ``sender[target, k]`` stores the source node index for one stencil slot of a
    target node. ``edge_attr[target, k]`` stores the matching directed geometry.
    This layout supports convolution-style accumulation without materializing
    batched edge messages shaped ``[B, E, C]``.
    """

    sender: torch.Tensor
    edge_attr: torch.Tensor
    degree: int


class SphericalMapTensorCache:
    """Cache edge tensors moved from CPU graph maps to a runtime device/dtype."""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, torch.device, torch.dtype], RuntimeEdgeTensors] = {}

    def get(
        self,
        graph_map: SphericalMap,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> RuntimeEdgeTensors:
        key = (id(graph_map), torch.device(device), dtype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        edge_set = graph_map.edge_set
        runtime = RuntimeEdgeTensors(
            sender=edge_set.sender.to(device=device),
            receiver=edge_set.receiver.to(device=device),
            edge_attr=edge_set.edge_attr.to(device=device, dtype=dtype),
        )
        self._cache[key] = runtime
        return runtime

    def clear(self) -> None:
        self._cache.clear()


class SphericalMapStencilCache:
    """Cache fixed-in-degree stencil tensors for convolution-style operators."""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, torch.device, torch.dtype], RuntimeStencilTensors] = {}

    def get(
        self,
        graph_map: SphericalMap,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> RuntimeStencilTensors:
        key = (id(graph_map), torch.device(device), dtype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        sender_cpu, edge_attr_cpu, degree = _build_fixed_indegree_stencil(graph_map)
        runtime = RuntimeStencilTensors(
            sender=sender_cpu.to(device=device),
            edge_attr=edge_attr_cpu.to(device=device, dtype=dtype),
            degree=degree,
        )
        self._cache[key] = runtime
        return runtime

    def clear(self) -> None:
        self._cache.clear()


def _build_fixed_indegree_stencil(
    graph_map: SphericalMap,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    edge_set = graph_map.edge_set
    sender = edge_set.sender.detach().cpu().to(torch.long)
    receiver = edge_set.receiver.detach().cpu().to(torch.long)
    edge_attr = edge_set.edge_attr.detach().cpu()
    n_target = graph_map.n_target
    n_edges = int(sender.numel())

    if n_target <= 0:
        raise ValueError(f"Map {graph_map.name!r} has no target nodes.")
    if n_edges == 0:
        raise ValueError(f"Map {graph_map.name!r} has no edges.")

    counts = torch.bincount(receiver, minlength=n_target)
    unique_counts = torch.unique(counts)
    if unique_counts.numel() != 1:
        min_count = int(counts.min().item())
        max_count = int(counts.max().item())
        raise ValueError(
            f"Map {graph_map.name!r} is not fixed-in-degree: target degrees range "
            f"from {min_count} to {max_count}. Use a fixed-degree map such as barycentric, or the "
            "edge-message DynamicSphericalConv for irregular maps."
        )

    degree = int(unique_counts.item())
    if degree <= 0:
        raise ValueError(f"Map {graph_map.name!r} has zero incoming edges per target.")

    stencil_sender = torch.empty((n_target, degree), dtype=torch.long)
    stencil_edge_attr = torch.empty((n_target, degree, edge_attr.shape[-1]), dtype=edge_attr.dtype)
    fill = torch.zeros((n_target,), dtype=torch.long)

    for edge_idx in range(n_edges):
        target_idx = int(receiver[edge_idx].item())
        slot = int(fill[target_idx].item())
        stencil_sender[target_idx, slot] = sender[edge_idx]
        stencil_edge_attr[target_idx, slot] = edge_attr[edge_idx]
        fill[target_idx] += 1

    return stencil_sender, stencil_edge_attr, degree
