"""Spherical graph builders."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from .geometry import latlon_to_xyz, normalize_xyz, xyz_to_latlon
from .graph import SphericalGraph


def make_equiangular_grid(
    nlat: int,
    nlon: int,
    *,
    name: str = "equiangular_grid",
    lat_min: float = -90.0,
    lat_max: float = 90.0,
    lon_min: float = 0.0,
    lon_max: float = 360.0,
    centers: bool = True,
) -> SphericalGraph:
    """Build an equiangular latitude-longitude grid graph."""

    if nlat <= 0 or nlon <= 0:
        raise ValueError("nlat and nlon must be positive.")
    if lat_min >= lat_max:
        raise ValueError("lat_min must be smaller than lat_max.")
    if lon_min >= lon_max:
        raise ValueError("lon_min must be smaller than lon_max.")

    if centers:
        dlat = (lat_max - lat_min) / float(nlat)
        dlon = (lon_max - lon_min) / float(nlon)
        lat = torch.linspace(lat_min + 0.5 * dlat, lat_max - 0.5 * dlat, nlat)
        lon = torch.linspace(lon_min + 0.5 * dlon, lon_max - 0.5 * dlon, nlon)
    else:
        lat = torch.linspace(lat_min, lat_max, nlat)
        lon = torch.linspace(lon_min, lon_max, nlon)

    lat2d, lon2d = torch.meshgrid(
        lat.to(torch.float32), lon.to(torch.float32), indexing="ij"
    )
    flat_lat = lat2d.reshape(-1)
    flat_lon = lon2d.reshape(-1) % 360.0
    xyz = latlon_to_xyz(flat_lat, flat_lon)
    n_nodes = int(nlat * nlon)
    metadata = {
        "kind": "equiangular_grid",
        "nlat": int(nlat),
        "nlon": int(nlon),
        "shape": (int(nlat), int(nlon)),
        "flat_index": torch.arange(n_nodes, dtype=torch.long),
        "lat_index": torch.arange(n_nodes, dtype=torch.long) // int(nlon),
        "lon_index": torch.arange(n_nodes, dtype=torch.long) % int(nlon),
        "centers": bool(centers),
    }
    return SphericalGraph(
        name=name, xyz=xyz, lat=flat_lat, lon=flat_lon, metadata=metadata
    )


def make_equiangular_grid_from_xarray(
    lat_coord: Any,
    lon_coord: Any,
    *,
    name: str = "equiangular_grid",
    validate_equiangular: bool = True,
    atol: float = 1.0e-5,
    rtol: float = 1.0e-5,
) -> SphericalGraph:
    """Build an equiangular grid graph from xarray latitude/longitude coordinates.

    ``lat_coord`` and ``lon_coord`` are expected to be one-dimensional coordinate
    arrays, such as ``ds["lat"]`` and ``ds["lon"]`` from an xarray Dataset. The
    input coordinate order is preserved so flattened grid nodes match tensors
    laid out as ``[lat, lon]``. Longitudes are stored on the graph in ``[0, 360)``.
    """

    lat = _coordinate_to_1d_tensor(lat_coord, "lat_coord")
    lon = _coordinate_to_1d_tensor(lon_coord, "lon_coord")
    if validate_equiangular:
        _validate_regular_spacing(lat, "lat_coord", atol=atol, rtol=rtol)
        _validate_regular_spacing(lon, "lon_coord", atol=atol, rtol=rtol)

    lat2d, lon2d = torch.meshgrid(lat, lon, indexing="ij")
    flat_lat = lat2d.reshape(-1)
    flat_lon = lon2d.reshape(-1) % 360.0
    xyz = latlon_to_xyz(flat_lat, flat_lon)
    n_nodes = int(lat.numel() * lon.numel())
    metadata = {
        "kind": "equiangular_grid",
        "source": "xarray_coords",
        "nlat": int(lat.numel()),
        "nlon": int(lon.numel()),
        "shape": (int(lat.numel()), int(lon.numel())),
        "flat_index": torch.arange(n_nodes, dtype=torch.long),
        "lat_index": torch.arange(n_nodes, dtype=torch.long) // int(lon.numel()),
        "lon_index": torch.arange(n_nodes, dtype=torch.long) % int(lon.numel()),
        "centers": None,
        "lat_coord_name": _coordinate_name(lat_coord),
        "lon_coord_name": _coordinate_name(lon_coord),
        "lat_first": float(lat[0].item()),
        "lat_last": float(lat[-1].item()),
        "lon_first_input": float(lon[0].item()),
        "lon_last_input": float(lon[-1].item()),
        "lon_wrapped_to_360": True,
    }
    return SphericalGraph(
        name=name, xyz=xyz, lat=flat_lat, lon=flat_lon, metadata=metadata
    )


def make_icosphere(
    level: int,
    *,
    name: str | None = None,
) -> tuple[SphericalGraph, torch.Tensor]:
    """Build an icosphere by recursively subdividing an icosahedron."""

    if level < 0:
        raise ValueError("level must be non-negative.")

    phi = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = [
        (-1.0, phi, 0.0),
        (1.0, phi, 0.0),
        (-1.0, -phi, 0.0),
        (1.0, -phi, 0.0),
        (0.0, -1.0, phi),
        (0.0, 1.0, phi),
        (0.0, -1.0, -phi),
        (0.0, 1.0, -phi),
        (phi, 0.0, -1.0),
        (phi, 0.0, 1.0),
        (-phi, 0.0, -1.0),
        (-phi, 0.0, 1.0),
    ]
    faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]

    vertex_tensor = normalize_xyz(torch.tensor(vertices, dtype=torch.float32))
    vertex_list = [tuple(row.tolist()) for row in vertex_tensor]

    def midpoint_index(i: int, j: int, cache: dict[tuple[int, int], int]) -> int:
        key = (min(i, j), max(i, j))
        if key in cache:
            return cache[key]
        midpoint = normalize_xyz(
            0.5
            * (
                torch.tensor(vertex_list[i], dtype=torch.float32)
                + torch.tensor(vertex_list[j], dtype=torch.float32)
            )
        )
        vertex_list.append(tuple(midpoint.tolist()))
        cache[key] = len(vertex_list) - 1
        return cache[key]

    for _ in range(level):
        cache: dict[tuple[int, int], int] = {}
        next_faces: list[tuple[int, int, int]] = []
        for a, b, c in faces:
            ab = midpoint_index(a, b, cache)
            bc = midpoint_index(b, c, cache)
            ca = midpoint_index(c, a, cache)
            next_faces.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))
        faces = next_faces

    xyz = normalize_xyz(torch.tensor(vertex_list, dtype=torch.float32))
    lat, lon = xyz_to_latlon(xyz)
    graph_name = name or f"icosphere_l{level}"
    graph = SphericalGraph(
        name=graph_name,
        xyz=xyz,
        lat=lat,
        lon=lon,
        metadata={"kind": "icosphere", "level": int(level)},
    )
    return graph, torch.tensor(faces, dtype=torch.long)


def make_fibonacci_sphere(
    n_nodes: int,
    *,
    name: str = "fibonacci_sphere",
) -> SphericalGraph:
    """Build a simple quasi-uniform Fibonacci sphere graph."""

    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive.")

    idx = torch.arange(n_nodes, dtype=torch.float32)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - 2.0 * (idx + 0.5) / float(n_nodes)
    radius = torch.sqrt((1.0 - z * z).clamp_min(0.0))
    theta = golden_angle * idx
    xyz = torch.stack((radius * torch.cos(theta), radius * torch.sin(theta), z), dim=-1)
    xyz = normalize_xyz(xyz)
    lat, lon = xyz_to_latlon(xyz)
    return SphericalGraph(
        name=name,
        xyz=xyz,
        lat=lat,
        lon=lon,
        metadata={"kind": "fibonacci_sphere", "n_nodes": int(n_nodes)},
    )


def _coordinate_to_1d_tensor(coord: Any, label: str) -> torch.Tensor:
    if hasattr(coord, "to_numpy"):
        array = coord.to_numpy()
    else:
        array = np.asarray(coord)
    array = np.asarray(array)
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional, got shape {array.shape}.")
    if array.size == 0:
        raise ValueError(f"{label} must contain at least one value.")
    try:
        tensor = torch.tensor(array, dtype=torch.float32).detach().cpu()
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{label} must contain numeric latitude/longitude values."
        ) from exc
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{label} must contain only finite values.")
    return tensor


def _validate_regular_spacing(
    coord: torch.Tensor,
    label: str,
    *,
    atol: float,
    rtol: float,
) -> None:
    if coord.numel() <= 2:
        return
    deltas = coord[1:] - coord[:-1]
    expected = deltas[0].expand_as(deltas)
    if not torch.allclose(deltas, expected, atol=atol, rtol=rtol):
        raise ValueError(
            f"{label} must be regularly spaced for an equiangular grid; "
            f"first spacing is {float(deltas[0].item())}."
        )


def _coordinate_name(coord: Any) -> str | None:
    name = getattr(coord, "name", None)
    return str(name) if name is not None else None
