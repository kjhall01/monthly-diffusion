"""Fixed-orientation spherical hierarchy assembly.

Only the mesh coordinates change. The physical grid stays Earth-fixed. All
levels/components use R = Rz(11 deg) @ Ry(23 deg) @ Rx(17 deg), acting on column
vectors (x at 0N/0E, y at 0N/90E, z at the North Pole).
Edges, geographic features and pooled statics are rebuilt at the new locations.
No random draws, learned rotations, or layer-dependent orientations are used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal, Sequence

import torch

from .mesh import (
    make_equiangular_grid,
    make_equiangular_grid_from_xarray as make_equiangular_grid_from_xarray,
    make_icosphere,
    make_fibonacci_sphere as make_fibonacci_sphere,
)
from .connectivity import (
    build_barycentric_map,
    build_mesh_adjacency_map,
    build_radius_map,
    build_spherical_delaunay_faces as build_spherical_delaunay_faces,
)
from .features import (
    xyz_to_latlon,
    latlon_to_xyz as latlon_to_xyz,
    compute_edge_features as compute_edge_features,
)
from .graph import SphericalGraph, EdgeSet as EdgeSet, SphericalMap as SphericalMap


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
        raise ValueError("mesh_rotation_degrees must contain three finite x/y/z angles.")
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

    Grid->mesh and mesh-down use radius queries (0.6 times maximum target
    edge length); mesh-up/grid use barycentric triangle connectivity.
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
            mesh, faces_by_level[index], name=f"{mesh_name}_to_{mesh_name}", bidirectional=True
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
            coarse, faces_by_level[index + 1], fine, name=f"{coarse_name}_to_{fine_name}"
        )
    for graph_map in maps.values():
        indegree = torch.bincount(graph_map.edge_set.receiver, minlength=graph_map.n_target)
        if (indegree == 0).any():
            raise ValueError(f"MDV1 rotated map {graph_map.name} has targets without sources.")

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


def with_static_edge_features(hierarchy, grid_statics, *, names):
    """Append source/target statics to every ECC map, without changing edges.

    ``grid_statics`` is finite, already normalized [grid_node, static_channel]
    data in physical-grid flattening order. Mesh values use the same inbound
    arithmetic means as FiLM's StaticPool (not area weighting or interpolation).
    Attribute order is [existing geometry, source statics, target statics].
    Barycentric weights keep their existing column. Returned maps are new so
    stencil caches cannot reuse attributes from the geometry-only maps.
    """
    if "edge_static_fields" in hierarchy.metadata:
        raise ValueError("Static edge features are already attached")
    names = tuple(names)
    values = torch.as_tensor(grid_statics).detach().cpu().to(torch.float32)
    if not names or len(set(names)) != len(names):
        raise ValueError("Static edge feature names must be nonempty and unique")
    if values.shape != (hierarchy.graphs["grid"].n_nodes, len(names)):
        raise ValueError("grid_statics must have shape [grid_node, static_channel]")
    if not torch.isfinite(values).all():
        raise ValueError("Static edge features must be finite after normalization")
    levels = ["grid", *hierarchy.metadata["mesh_level_names"]]
    fields = {"grid": values}
    for source, target in zip(levels[:-1], levels[1:]):
        graph_map = hierarchy.maps[f"{source}_to_{target}"]
        edges = graph_map.edge_set
        pooled = values.new_zeros(graph_map.n_target, len(names))
        pooled.scatter_add_(0, edges.receiver[:, None].expand(-1, len(names)), values[edges.sender])
        count = torch.bincount(edges.receiver, minlength=graph_map.n_target).clamp_min(1)
        values = pooled / count.to(pooled)[:, None]
        fields[target] = values

    maps = {}
    for name, graph_map in hierarchy.maps.items():
        edges = graph_map.edge_set
        attributes = torch.cat(
            (
                edges.edge_attr,
                fields[graph_map.source.name][edges.sender],
                fields[graph_map.target.name][edges.receiver],
            ),
            dim=-1,
        )
        metadata = dict(
            static_feature_names=names,
            static_feature_start=edges.edge_attr_dim,
            static_feature_order="source_then_target",
        )
        maps[name] = replace(
            graph_map,
            edge_set=replace(edges, edge_attr=attributes, metadata={**edges.metadata, **metadata}),
            metadata={**graph_map.metadata, **metadata},
        )
    return MeshHierarchy(
        hierarchy.graphs,
        maps,
        {**hierarchy.metadata, "edge_static_fields": names, "edge_static_pooling": "inbound_mean"},
    )


def spherical_voronoi(graph):
    """Voronoi cells/areas for arbitrary unit-sphere node locations (SciPy)."""
    import numpy as np
    from scipy.spatial import SphericalVoronoi

    xyz = graph.xyz.double().numpy()
    xyz = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    cells = SphericalVoronoi(xyz, radius=1.0, center=np.zeros(3))
    cells.sort_vertices_of_regions()
    return cells
