"""Geometry utilities for unit-sphere graphs."""

from __future__ import annotations

import math

import torch


def normalize_xyz(xyz: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    """Return unit-normalized Cartesian vectors."""

    tensor = torch.as_tensor(xyz)
    if tensor.shape[-1] != 3:
        raise ValueError(f"xyz must have last dimension 3, got shape {tuple(tensor.shape)}.")
    norm = torch.linalg.norm(tensor, dim=-1, keepdim=True)
    if torch.any(norm <= eps):
        raise ValueError("Cannot normalize zero-length Cartesian vectors.")
    return tensor / norm.clamp_min(eps)


def latlon_to_xyz(lat_deg: torch.Tensor, lon_deg: torch.Tensor) -> torch.Tensor:
    """Convert latitude/longitude degrees to [N, 3] unit Cartesian vectors."""

    lat = torch.as_tensor(lat_deg, dtype=torch.float32)
    lon = torch.as_tensor(lon_deg, dtype=torch.float32)
    if lat.shape != lon.shape:
        raise ValueError(f"lat and lon must have the same shape, got {lat.shape} and {lon.shape}.")

    lat_rad = torch.deg2rad(lat.reshape(-1))
    lon_rad = torch.deg2rad(lon.reshape(-1))
    cos_lat = torch.cos(lat_rad)
    xyz = torch.stack(
        (
            cos_lat * torch.cos(lon_rad),
            cos_lat * torch.sin(lon_rad),
            torch.sin(lat_rad),
        ),
        dim=-1,
    )
    return normalize_xyz(xyz)


def latlon_periodic_features(lat_deg: torch.Tensor, lon_deg: torch.Tensor) -> torch.Tensor:
    """Return periodic latitude/longitude node features.

    Features are ``[sin(lat), cos(lat), sin(lon), cos(lon)]``. Longitude is
    wrapped modulo 360 degrees before conversion, so nodes at 0 and 360 degrees
    longitude have identical features.
    """

    lat = torch.as_tensor(lat_deg, dtype=torch.float32).reshape(-1)
    lon = torch.as_tensor(lon_deg, dtype=torch.float32).reshape(-1) % 360.0
    if lat.shape != lon.shape:
        raise ValueError(f"lat and lon must have the same shape, got {lat.shape} and {lon.shape}.")
    lat_rad = torch.deg2rad(lat)
    lon_rad = torch.deg2rad(lon)
    return torch.stack(
        (
            torch.sin(lat_rad),
            torch.cos(lat_rad),
            torch.sin(lon_rad),
            torch.cos(lon_rad),
        ),
        dim=-1,
    )


def xyz_to_latlon(xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert [N, 3] unit Cartesian vectors to latitude and longitude degrees."""

    unit = normalize_xyz(torch.as_tensor(xyz, dtype=torch.float32))
    flat = unit.reshape(-1, 3)
    lat = torch.rad2deg(torch.asin(flat[:, 2].clamp(-1.0, 1.0)))
    lon = torch.rad2deg(torch.atan2(flat[:, 1], flat[:, 0])) % 360.0
    return lat, lon


def chord_distance(xyz_a: torch.Tensor, xyz_b: torch.Tensor) -> torch.Tensor:
    """Euclidean distance between unit-sphere Cartesian points."""

    return torch.linalg.norm(torch.as_tensor(xyz_a) - torch.as_tensor(xyz_b), dim=-1)


def great_circle_distance(xyz_a: torch.Tensor, xyz_b: torch.Tensor) -> torch.Tensor:
    """Great-circle distance in radians on the unit sphere."""

    a = normalize_xyz(torch.as_tensor(xyz_a, dtype=torch.float32))
    b = normalize_xyz(torch.as_tensor(xyz_b, dtype=torch.float32))
    dot = torch.sum(a * b, dim=-1).clamp(-1.0, 1.0)
    cross_norm = torch.linalg.norm(torch.cross(a, b, dim=-1), dim=-1)
    return torch.atan2(cross_norm, dot).clamp(0.0, math.pi)


def compute_edge_features(
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    sender: torch.Tensor,
    receiver: torch.Tensor,
) -> torch.Tensor:
    """Compute directional edge features.

    The returned attributes are
    ``[rel_east, rel_north, chord, great_circle, source_periodic..., target_periodic...]``.

    ``rel_east`` and ``rel_north`` are target-local tangent-plane coordinates
    of the sender position relative to the receiver position. This makes edge
    direction encode concepts such as "source is west/east/north/south of the
    target" rather than absolute Earth-centered Cartesian x/y/z direction.
    The periodic latitude/longitude features are
    ``[sin(lat), cos(lat), sin(lon), cos(lon)]`` for the source node and target
    node, making kernels aware of absolute location on the rotating sphere
    without longitude discontinuities.
    """

    source = normalize_xyz(torch.as_tensor(source_xyz, dtype=torch.float32))
    target = normalize_xyz(torch.as_tensor(target_xyz, dtype=torch.float32))
    sender_idx = torch.as_tensor(sender, dtype=torch.long)
    receiver_idx = torch.as_tensor(receiver, dtype=torch.long)
    if sender_idx.shape != receiver_idx.shape:
        raise ValueError("sender and receiver must have the same shape.")
    if sender_idx.ndim != 1:
        raise ValueError("sender and receiver must be one-dimensional edge index tensors.")

    if sender_idx.numel() == 0:
        return torch.empty((0, 12), dtype=torch.float32)

    src = source[sender_idx]
    tgt = target[receiver_idx]
    source_lat, source_lon = xyz_to_latlon(source)
    target_lat, target_lon = xyz_to_latlon(target)
    source_periodic = latlon_periodic_features(source_lat, source_lon)[sender_idx]
    target_periodic = latlon_periodic_features(target_lat, target_lon)[receiver_idx]
    rel = src - tgt
    east, north, _ = local_tangent_basis(tgt)
    local_rel = torch.stack(
        (
            torch.sum(rel * east, dim=-1),
            torch.sum(rel * north, dim=-1),
        ),
        dim=-1,
    )
    chord = chord_distance(src, tgt).unsqueeze(-1)
    arc = great_circle_distance(src, tgt).unsqueeze(-1)
    return torch.cat((local_rel, chord, arc, source_periodic, target_periodic), dim=-1).to(
        torch.float32
    )


def local_tangent_basis(xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return east/north/up unit vectors for unit-sphere points.

    The basis is derived from latitude/longitude, so it is tied to the sphere's
    rotation axis. ``east`` points toward increasing longitude, ``north`` points
    toward increasing latitude, and ``up`` is radial.
    """

    up = normalize_xyz(torch.as_tensor(xyz, dtype=torch.float32))
    flat = up.reshape(-1, 3)
    lat, lon = xyz_to_latlon(flat)
    lat_rad = torch.deg2rad(lat)
    lon_rad = torch.deg2rad(lon)
    sin_lat = torch.sin(lat_rad)
    cos_lat = torch.cos(lat_rad)
    sin_lon = torch.sin(lon_rad)
    cos_lon = torch.cos(lon_rad)
    east = torch.stack((-sin_lon, cos_lon, torch.zeros_like(lon_rad)), dim=-1)
    north = torch.stack((-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat), dim=-1)
    return (
        east.reshape_as(up),
        north.reshape_as(up),
        flat.reshape_as(up),
    )
