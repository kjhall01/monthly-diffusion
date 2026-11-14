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

from monthly_diffusion._geometry.builders import (
    make_equiangular_grid,
    make_equiangular_grid_from_xarray,
    make_icosphere,
    make_fibonacci_sphere,
)
from monthly_diffusion._geometry.connectivity import (
    build_barycentric_map,
    build_mesh_adjacency_map,
    build_radius_map,
)
from monthly_diffusion._geometry.geometry import xyz_to_latlon
from monthly_diffusion._geometry.graph import SphericalGraph


@dataclass
class MeshHierarchy:
    """Fixed geometry only; lists run from the physical grid toward the latent mesh.

    down_maps[i] goes to meshes[i]; up_maps[i] reverses that transfer.
    neighbor_maps[i] connects adjacent nodes within meshes[i].
    """

    grid: SphericalGraph
    meshes: list
    down_maps: list
    up_maps: list
    neighbor_maps: list
    mesh_levels: tuple
    rotation: tuple


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


def build_mesh_hierarchy(
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

    neighbor_maps = []
    down_maps = []
    up_maps = []
    for i, (mesh, faces) in enumerate(zip(meshes, faces_by_level)):
        neighbor_maps.append(
            build_mesh_adjacency_map(
                mesh,
                faces,
                name=f"{mesh.name}_to_{mesh.name}",
                bidirectional=True,
            )
        )
        source = grid if i == 0 else meshes[i - 1]
        down_maps.append(
            build_radius_map(
                source,
                mesh,
                radius=grid_to_mesh_radius if i == 0 else down_radius,
                radius_meters=grid_to_mesh_radius_meters
                if i == 0
                else down_radius_meters,
                radius_query_fraction_edge_length=radius_query_fraction_edge_length,
                auto_radius_graph="target",
                name=f"{source.name}_to_{mesh.name}",
            )
        )
        up_maps.append(
            build_barycentric_map(
                mesh,
                faces,
                source,
                name=f"{mesh.name}_to_{source.name}",
            )
        )
    for mapping in down_maps + up_maps + neighbor_maps:
        indegree = torch.bincount(mapping.edge_set.receiver, minlength=mapping.n_target)
        if (indegree == 0).any():
            raise ValueError(f"Map {mapping.name} has targets without sources")
    return MeshHierarchy(
        grid,
        meshes,
        down_maps,
        up_maps,
        neighbor_maps,
        tuple(level_values),
        tuple(rotation_degrees),
    )


from monthly_diffusion._geometry.graph import EdgeSet, SphericalMap
from monthly_diffusion._geometry.geometry import latlon_to_xyz, compute_edge_features
from monthly_diffusion._geometry.connectivity import build_spherical_delaunay_faces


def spherical_voronoi(graph):
    """Voronoi cells/areas for arbitrary unit-sphere node locations (SciPy)."""
    import numpy as np
    from scipy.spatial import SphericalVoronoi

    xyz = graph.xyz.double().numpy()
    xyz = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    cells = SphericalVoronoi(xyz, radius=1.0, center=np.zeros(3))
    cells.sort_vertices_of_regions()
    return cells
