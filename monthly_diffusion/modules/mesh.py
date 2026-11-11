"""Fixed-orientation spherical hierarchy assembly, adapted from P163.

Only the mesh coordinates change. The physical grid stays Earth-fixed. All
levels/components use R = Rz(11 deg) @ Ry(23 deg) @ Rx(17 deg), acting on column
vectors (x at 0N/0E, y at 0N/90E, z at the North Pole).
Edges, geographic features and pooled statics are rebuilt at the new locations.
No random draws, learned rotations, or layer-dependent orientations are used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import torch

from ._geometry.builders import (
    make_equiangular_grid,
    make_equiangular_grid_from_xarray,
    make_icosphere,
    make_fibonacci_sphere,
)
from ._geometry.connectivity import (
    build_barycentric_map,
    build_mesh_adjacency_map,
    build_radius_map,
)
from ._geometry.geometry import xyz_to_latlon
from ._geometry.graph import SphericalGraph


@dataclass
class MeshHierarchy:
    """Plain dictionaries of fixed graphs, maps, and construction metadata."""

    graphs: dict
    maps: dict
    metadata: dict


MESH_ROTATION_DEGREES = (17.0, 23.0, 11.0)
ROTATION_CONVENTION = "Earth-fixed active rotation; column xyz: Rz(z) @ Ry(y) @ Rx(x)"


def rotation_matrix(degrees: Sequence[float] = MESH_ROTATION_DEGREES) -> torch.Tensor:
    """Proper orthogonal matrix, float64 CPU; angles are x, y, z in degrees."""
    if len(degrees) != 3 or not all(math.isfinite(float(v)) for v in degrees):
        raise ValueError(
            "mesh_rotation_degrees must contain three finite x/y/z angles."
        )
    x, y, z = (math.radians(float(v)) for v in degrees)
    cx, sx, cy, sy, cz, sz = (
        math.cos(x),
        math.sin(x),
        math.cos(y),
        math.sin(y),
        math.cos(z),
        math.sin(z),
    )
    rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=torch.float64)
    ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float64)
    rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=torch.float64)
    return rz @ ry @ rx


def make_rotated_icosphere(
    level: int,
    *,
    name: str | None = None,
    rotation_degrees: Sequence[float] = MESH_ROTATION_DEGREES,
) -> tuple[SphericalGraph, torch.Tensor]:
    """Rigidly rotate MDV1's mesh, preserving face indices and nested vertices."""
    original, faces = make_icosphere(level, name=name)
    matrix = rotation_matrix(rotation_degrees)
    xyz = (original.xyz.to(torch.float64) @ matrix.T).to(torch.float32)
    lat, lon = xyz_to_latlon(xyz)
    return SphericalGraph(
        name=original.name,
        xyz=xyz,
        lat=lat,
        lon=lon,
        metadata={
            **original.metadata,
            "rotation_degrees": list(rotation_degrees),
            "rotation_convention": ROTATION_CONVENTION,
            "rotation_matrix": matrix.tolist(),
        },
    ), faces


def build_mesh_hierarchy_context(
    *,
    nlat: int,
    nlon: int,
    mesh_levels: Sequence[int],
    grid_to_mesh_radius: float | None = None,
    grid_to_mesh_radius_meters: float | Literal["auto"] | None = "auto",
    down_radius: float | None = None,
    down_radius_meters: float | Literal["auto"] | None = "auto",
    radius_query_fraction_edge_length: float = 0.6,
    rotation_degrees: Sequence[float] = MESH_ROTATION_DEGREES,
    grid: SphericalGraph | None = None,
) -> MeshHierarchy:
    """MDV1 hierarchy recipe with the same rigid rotation at every mesh level.

    Grid->mesh and mesh-down use the inherited radius rule (0.6 times maximum
    target edge length); mesh-up/grid use the inherited barycentric rule.
    Failure of radius coverage is an error, never silently repaired by adding
    neighbors or changing the neighborhood policy.
    """
    if not mesh_levels:
        raise ValueError("mesh_levels must contain at least one icosphere level.")
    level_values = [int(level) for level in mesh_levels]
    if any(level < 0 for level in level_values):
        raise ValueError("mesh_levels must be non-negative.")
    if len(level_values) > 1 and not all(
        level_values[i] > level_values[i + 1] for i in range(len(level_values) - 1)
    ):
        raise ValueError("mesh_levels must be finest-to-coarsest, e.g. [5, 4, 3].")
    if grid is None:
        grid = make_equiangular_grid(nlat=nlat, nlon=nlon, name="grid")
    meshes = []
    faces_by_level = []
    for index, level in enumerate(level_values):
        mesh, faces = make_rotated_icosphere(
            level,
            name=f"mesh_l{index}",
            rotation_degrees=rotation_degrees,
        )
        meshes.append(mesh)
        faces_by_level.append(faces)

    graphs, maps = {"grid": grid}, {}
    for index, mesh in enumerate(meshes):
        mesh_name = f"mesh_l{index}"
        graphs[mesh_name] = mesh
        maps[f"{mesh_name}_to_{mesh_name}"] = build_mesh_adjacency_map(
            mesh,
            faces_by_level[index],
            name=f"{mesh_name}_to_{mesh_name}",
            bidirectional=True,
        )
    maps["grid_to_mesh_l0"] = build_radius_map(
        grid,
        meshes[0],
        radius=grid_to_mesh_radius,
        radius_meters=grid_to_mesh_radius_meters,
        radius_query_fraction_edge_length=radius_query_fraction_edge_length,
        auto_radius_graph="target",
        name="grid_to_mesh_l0",
    )
    maps["mesh_l0_to_grid"] = build_barycentric_map(
        meshes[0], faces_by_level[0], grid, name="mesh_l0_to_grid"
    )
    for index in range(len(meshes) - 1):
        fine_name, coarse_name = f"mesh_l{index}", f"mesh_l{index + 1}"
        fine, coarse = meshes[index], meshes[index + 1]
        maps[f"{fine_name}_to_{coarse_name}"] = build_radius_map(
            fine,
            coarse,
            radius=down_radius,
            radius_meters=down_radius_meters,
            radius_query_fraction_edge_length=radius_query_fraction_edge_length,
            auto_radius_graph="target",
            name=f"{fine_name}_to_{coarse_name}",
        )
        maps[f"{coarse_name}_to_{fine_name}"] = build_barycentric_map(
            coarse,
            faces_by_level[index + 1],
            fine,
            name=f"{coarse_name}_to_{fine_name}",
        )
    for graph_map in maps.values():
        indegree = torch.bincount(
            graph_map.edge_set.receiver, minlength=graph_map.n_target
        )
        if (indegree == 0).any():
            raise ValueError(
                f"MDV1 rotated map {graph_map.name} has targets without sources."
            )

    metadata = dict(
        mesh_level_names=tuple(f"mesh_l{i}" for i in range(len(meshes))),
        mesh_levels=tuple(level_values),
        grid_to_mesh_method="radius",
        mesh_down_method="radius",
        mesh_up_method="barycentric",
        mesh_to_grid_method="barycentric",
        mesh_rotation_degrees=tuple(float(v) for v in rotation_degrees),
        mesh_rotation_convention=ROTATION_CONVENTION,
        mesh_rotation_matrix=rotation_matrix(rotation_degrees).tolist(),
    )
    return MeshHierarchy(graphs, maps, metadata)


from ._geometry.graph import EdgeSet, SphericalMap
from ._geometry.geometry import latlon_to_xyz, compute_edge_features
from ._geometry.connectivity import build_spherical_delaunay_faces


def spherical_voronoi(graph):
    """Voronoi cells/areas for arbitrary unit-sphere node locations (SciPy)."""
    import numpy as np
    from scipy.spatial import SphericalVoronoi

    xyz = graph.xyz.double().numpy()
    xyz = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    cells = SphericalVoronoi(xyz, radius=1.0, center=np.zeros(3))
    cells.sort_vertices_of_regions()
    return cells
