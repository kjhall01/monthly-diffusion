"""Core spherical graph records.

``SphericalGraph`` is a named set of nodes on the unit sphere.
``SphericalMap`` is directed sender/receiver connectivity between two graphs.
All edge sets use directed records. Geometrically undirected adjacency should
store both directions explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import torch


def _metadata_proxy(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(metadata))


def _cpu_float_tensor(value: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().cpu()
    if not torch.is_floating_point(tensor):
        tensor = tensor.to(torch.float32)
    return tensor.to(torch.float32)


def _cpu_long_tensor(value: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.dtype != torch.long:
        raise ValueError(f"{name} must have dtype torch.long, got {tensor.dtype}.")
    return tensor


@dataclass(frozen=True)
class SphericalGraph:
    """A named set of nodes on the unit sphere."""

    name: str
    xyz: torch.Tensor
    lat: torch.Tensor
    lon: torch.Tensor
    area: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        xyz = _cpu_float_tensor(self.xyz, name="xyz")
        lat = _cpu_float_tensor(self.lat, name="lat")
        lon = _cpu_float_tensor(self.lon, name="lon") % 360.0

        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"SphericalGraph {self.name!r}: xyz must have shape [N, 3].")
        n_nodes = int(xyz.shape[0])
        if lat.shape != (n_nodes,):
            raise ValueError(
                f"SphericalGraph {self.name!r}: lat must have shape [{n_nodes}], got {tuple(lat.shape)}."
            )
        if lon.shape != (n_nodes,):
            raise ValueError(
                f"SphericalGraph {self.name!r}: lon must have shape [{n_nodes}], got {tuple(lon.shape)}."
            )
        if not torch.isfinite(xyz).all():
            raise ValueError(f"SphericalGraph {self.name!r}: xyz contains non-finite values.")

        norm = torch.linalg.norm(xyz, dim=-1, keepdim=True)
        if torch.any(norm <= 1.0e-12):
            raise ValueError(f"SphericalGraph {self.name!r}: xyz contains zero-length vectors.")
        xyz = xyz / norm

        area = None
        if self.area is not None:
            area = _cpu_float_tensor(self.area, name="area")
            if area.shape != (n_nodes,):
                raise ValueError(
                    f"SphericalGraph {self.name!r}: area must have shape [{n_nodes}], got {tuple(area.shape)}."
                )

        mask = None
        if self.mask is not None:
            mask = torch.as_tensor(self.mask).detach().cpu().to(torch.bool)
            if mask.shape != (n_nodes,):
                raise ValueError(
                    f"SphericalGraph {self.name!r}: mask must have shape [{n_nodes}], got {tuple(mask.shape)}."
                )

        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "lat", lat)
        object.__setattr__(self, "lon", lon)
        object.__setattr__(self, "area", area)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "metadata", _metadata_proxy(self.metadata))

    @property
    def n_nodes(self) -> int:
        return int(self.xyz.shape[0])


@dataclass(frozen=True)
class EdgeSet:
    """Directed sender/receiver records and per-edge attributes."""

    name: str
    sender: torch.Tensor
    receiver: torch.Tensor
    edge_attr: torch.Tensor
    directed: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sender = _cpu_long_tensor(self.sender, name="sender")
        receiver = _cpu_long_tensor(self.receiver, name="receiver")
        edge_attr = _cpu_float_tensor(self.edge_attr, name="edge_attr")

        if sender.ndim != 1:
            raise ValueError(f"EdgeSet {self.name!r}: sender must have shape [E].")
        if receiver.ndim != 1:
            raise ValueError(f"EdgeSet {self.name!r}: receiver must have shape [E].")
        if sender.shape != receiver.shape:
            raise ValueError(
                f"EdgeSet {self.name!r}: sender and receiver must have the same shape, "
                f"got {tuple(sender.shape)} and {tuple(receiver.shape)}."
            )
        if edge_attr.ndim != 2:
            raise ValueError(f"EdgeSet {self.name!r}: edge_attr must have shape [E, D].")
        if edge_attr.shape[0] != sender.shape[0]:
            raise ValueError(
                f"EdgeSet {self.name!r}: edge_attr first dimension must equal edge count "
                f"{sender.shape[0]}, got {edge_attr.shape[0]}."
            )

        object.__setattr__(self, "sender", sender)
        object.__setattr__(self, "receiver", receiver)
        object.__setattr__(self, "edge_attr", edge_attr)
        object.__setattr__(self, "metadata", _metadata_proxy(self.metadata))

    @property
    def n_edges(self) -> int:
        return int(self.sender.shape[0])

    @property
    def edge_attr_dim(self) -> int:
        return int(self.edge_attr.shape[1])


@dataclass(frozen=True)
class SphericalMap:
    """Directed connectivity from one ``SphericalGraph`` to another."""

    name: str
    source: SphericalGraph
    target: SphericalGraph
    edge_set: EdgeSet
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.edge_set.n_edges:
            sender_min = int(self.edge_set.sender.min().item())
            sender_max = int(self.edge_set.sender.max().item())
            receiver_min = int(self.edge_set.receiver.min().item())
            receiver_max = int(self.edge_set.receiver.max().item())
            if sender_min < 0 or sender_max >= self.source.n_nodes:
                raise ValueError(
                    f"SphericalMap {self.name!r}: sender indices must be in "
                    f"[0, {self.source.n_nodes}), got min={sender_min}, max={sender_max}."
                )
            if receiver_min < 0 or receiver_max >= self.target.n_nodes:
                raise ValueError(
                    f"SphericalMap {self.name!r}: receiver indices must be in "
                    f"[0, {self.target.n_nodes}), got min={receiver_min}, max={receiver_max}."
                )
        object.__setattr__(self, "metadata", _metadata_proxy(self.metadata))

    @property
    def n_source(self) -> int:
        return self.source.n_nodes

    @property
    def n_target(self) -> int:
        return self.target.n_nodes

    @property
    def n_edges(self) -> int:
        return self.edge_set.n_edges
