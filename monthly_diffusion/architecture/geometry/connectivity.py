"""Connectivity builders for spherical graphs."""

from __future__ import annotations

import math
import logging
from typing import Literal

import torch

from .features import compute_edge_features
from .graph import EdgeSet, SphericalGraph, SphericalMap

try:
    from scipy.spatial import ConvexHull, QhullError, cKDTree
except Exception:  # pragma: no cover - exercised only when scipy is unavailable.
    ConvexHull = None
    QhullError = Exception
    cKDTree = None

DEFAULT_EARTH_RADIUS_METERS = 6_371_000.0
logger = logging.getLogger(__name__)


def build_mesh_adjacency_map(
    mesh: SphericalGraph,
    faces: torch.Tensor,
    *,
    name: str = "mesh_to_mesh",
    bidirectional: bool = True,
) -> SphericalMap:
    """Build directed mesh adjacency from triangular faces."""

    logger.info(
        "Building mesh adjacency map %s: mesh=%s nodes=%d faces=%d bidirectional=%s",
        name,
        mesh.name,
        mesh.n_nodes,
        int(torch.as_tensor(faces).shape[0]),
        bidirectional,
    )
    face_tensor = torch.as_tensor(faces, dtype=torch.long).detach().cpu()
    if face_tensor.ndim != 2 or face_tensor.shape[1] != 3:
        raise ValueError(f"faces must have shape [F, 3], got {tuple(face_tensor.shape)}.")
    if face_tensor.numel():
        min_index = int(face_tensor.min().item())
        max_index = int(face_tensor.max().item())
        if min_index < 0 or max_index >= mesh.n_nodes:
            raise ValueError(
                f"faces contain vertex indices outside [0, {mesh.n_nodes}): "
                f"min={min_index}, max={max_index}."
            )

    undirected_edges: set[tuple[int, int]] = set()
    for a, b, c in face_tensor.tolist():
        for u, v in ((a, b), (b, c), (c, a)):
            edge = (min(int(u), int(v)), max(int(u), int(v)))
            if edge[0] != edge[1]:
                undirected_edges.add(edge)

    directed_edges: list[tuple[int, int]] = []
    for u, v in sorted(undirected_edges):
        directed_edges.append((u, v))
        if bidirectional:
            directed_edges.append((v, u))

    if directed_edges:
        edge_tensor = torch.tensor(directed_edges, dtype=torch.long)
        sender = edge_tensor[:, 0]
        receiver = edge_tensor[:, 1]
    else:
        sender = torch.empty((0,), dtype=torch.long)
        receiver = torch.empty((0,), dtype=torch.long)

    edge_attr = compute_edge_features(mesh.xyz, mesh.xyz, sender, receiver)
    edge_set = EdgeSet(
        name=name,
        sender=sender,
        receiver=receiver,
        edge_attr=edge_attr,
        directed=True,
        metadata={"kind": "mesh_adjacency", "bidirectional": bool(bidirectional)},
    )
    graph_map = SphericalMap(
        name=name,
        source=mesh,
        target=mesh,
        edge_set=edge_set,
        metadata={"kind": "mesh_adjacency", "bidirectional": bool(bidirectional)},
    )
    logger.info("Built mesh adjacency map %s: edges=%d", name, graph_map.n_edges)
    return graph_map


def build_radius_map(
    source: SphericalGraph,
    target: SphericalGraph,
    *,
    radius: float | None = None,
    radius_meters: float | Literal["auto"] | None = "auto",
    earth_radius_meters: float = DEFAULT_EARTH_RADIUS_METERS,
    radius_query_fraction_edge_length: float = 0.6,
    auto_radius_graph: Literal["source", "target"] = "target",
    auto_qhull_options: str | None = None,
    name: str,
    max_neighbors: int | None = None,
) -> SphericalMap:
    """Connect source nodes within a radius of each target.

    ``SphericalGraph`` coordinates stay on the unit sphere. By default,
    ``radius_meters="auto"`` follows GraphCast's encoder heuristic:
    ``radius = 0.6 * longest_edge_length`` of the target graph's spherical
    Delaunay triangulation. This is appropriate for grid -> mesh encoder maps.

    Pass ``radius`` to specify a unit-sphere chord distance directly, or pass a
    numeric ``radius_meters`` to specify a physical great-circle radius.
    """

    logger.info(
        "Building radius map %s: source=%s(%d) target=%s(%d) radius=%s radius_meters=%s",
        name,
        source.name,
        source.n_nodes,
        target.name,
        target.n_nodes,
        radius,
        radius_meters,
    )
    chord_radius, resolved_radius_meters, radius_metadata = _resolve_chord_radius(
        source=source,
        target=target,
        radius=radius,
        radius_meters=radius_meters,
        earth_radius_meters=earth_radius_meters,
        radius_query_fraction_edge_length=radius_query_fraction_edge_length,
        auto_radius_graph=auto_radius_graph,
        auto_qhull_options=auto_qhull_options,
    )
    if max_neighbors is not None and max_neighbors <= 0:
        raise ValueError("max_neighbors must be positive when provided.")

    pairs: list[tuple[int, int]] = []
    if cKDTree is not None:
        tree = cKDTree(source.xyz.numpy())
        neighbor_lists = tree.query_ball_point(target.xyz.numpy(), r=float(chord_radius))
        for target_idx, neighbors in enumerate(neighbor_lists):
            if max_neighbors is not None and len(neighbors) > max_neighbors:
                src = source.xyz[torch.as_tensor(neighbors, dtype=torch.long)]
                dist = torch.linalg.norm(src - target.xyz[target_idx], dim=-1)
                order = torch.argsort(dist)[:max_neighbors].tolist()
                neighbors = [neighbors[i] for i in order]
            for source_idx in neighbors:
                pairs.append((int(source_idx), int(target_idx)))
    else:
        dist = torch.cdist(target.xyz, source.xyz)
        for target_idx in range(target.n_nodes):
            neighbors = torch.nonzero(dist[target_idx] <= chord_radius, as_tuple=False).flatten()
            if max_neighbors is not None and neighbors.numel() > max_neighbors:
                order = torch.argsort(dist[target_idx, neighbors])[:max_neighbors]
                neighbors = neighbors[order]
            for source_idx in neighbors.tolist():
                pairs.append((int(source_idx), int(target_idx)))

    if pairs:
        edge_tensor = torch.tensor(pairs, dtype=torch.long)
        sender = edge_tensor[:, 0]
        receiver = edge_tensor[:, 1]
    else:
        sender = torch.empty((0,), dtype=torch.long)
        receiver = torch.empty((0,), dtype=torch.long)

    edge_attr = compute_edge_features(source.xyz, target.xyz, sender, receiver)
    metadata = {
        "kind": "radius",
        "radius": float(chord_radius),
        "radius_units": "unit_sphere_chord",
        "radius_meters": resolved_radius_meters,
        "earth_radius_meters": None
        if radius_metadata["mode"] == "unit_sphere_chord"
        else float(earth_radius_meters),
        "max_neighbors": max_neighbors,
        **radius_metadata,
    }
    edge_set = EdgeSet(
        name=name,
        sender=sender,
        receiver=receiver,
        edge_attr=edge_attr,
        directed=True,
        metadata=metadata,
    )
    graph_map = SphericalMap(
        name=name,
        source=source,
        target=target,
        edge_set=edge_set,
        metadata=metadata,
    )
    logger.info(
        "Built radius map %s: edges=%d chord_radius=%.6e radius_meters=%s",
        name,
        graph_map.n_edges,
        float(chord_radius),
        resolved_radius_meters,
    )
    return graph_map


def build_spherical_delaunay_faces(
    graph: SphericalGraph,
    *,
    qhull_options: str | None = None,
    orient_outward: bool = True,
) -> torch.Tensor:
    """Triangulate spherical graph nodes with a convex-hull Delaunay construction.

    For points on the unit sphere in general position, the triangular facets of
    their 3D convex hull are the spherical Delaunay triangulation. The returned
    ``[F, 3]`` long tensor can be passed directly to ``build_barycentric_map``.

    If Qhull reports a degeneracy, try ``qhull_options="QJ"`` to joggle nearly
    degenerate inputs.
    """

    if graph.n_nodes < 4:
        raise ValueError("At least four non-coplanar spherical nodes are required to build faces.")
    if ConvexHull is None:
        raise ImportError("build_spherical_delaunay_faces requires scipy.spatial.ConvexHull.")

    logger.info(
        "Building spherical Delaunay faces: graph=%s nodes=%d qhull_options=%s",
        graph.name,
        graph.n_nodes,
        qhull_options,
    )
    unique_xyz = torch.unique(graph.xyz, dim=0)
    if unique_xyz.shape[0] != graph.n_nodes:
        raise ValueError("Spherical Delaunay triangulation requires unique graph.xyz points.")

    try:
        hull = ConvexHull(graph.xyz.to(torch.float64).numpy(), qhull_options=qhull_options)
    except QhullError as exc:
        raise ValueError(
            "Could not build spherical Delaunay faces from graph.xyz. "
            "The points may be degenerate; try qhull_options='QJ'."
        ) from exc

    faces = torch.as_tensor(hull.simplices, dtype=torch.long)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"ConvexHull did not return triangular faces, got shape {tuple(faces.shape)}."
        )
    if faces.numel():
        min_index = int(faces.min().item())
        max_index = int(faces.max().item())
        if min_index < 0 or max_index >= graph.n_nodes:
            raise ValueError(
                f"ConvexHull returned vertex indices outside [0, {graph.n_nodes}): "
                f"min={min_index}, max={max_index}."
            )

    if orient_outward:
        faces = _orient_faces_outward(graph.xyz, faces)
    logger.info(
        "Built spherical Delaunay faces: graph=%s faces=%d",
        graph.name,
        int(faces.shape[0]),
    )
    return faces


def build_barycentric_faces(
    graph: SphericalGraph,
    *,
    qhull_options: str | None = None,
    orient_outward: bool = True,
) -> torch.Tensor:
    """Build source faces for barycentric maps from arbitrary spherical nodes."""

    return build_spherical_delaunay_faces(
        graph,
        qhull_options=qhull_options,
        orient_outward=orient_outward,
    )


def build_barycentric_map(
    source_mesh: SphericalGraph,
    source_faces: torch.Tensor,
    target: SphericalGraph,
    *,
    name: str = "mesh_to_target_barycentric",
    candidate_faces_k: int = 16,
    candidate_vertices_k: int = 8,
    target_chunk_size: int = 8192,
) -> SphericalMap:
    """Build GraphCast-faithful triangle connectivity from mesh to target.

    Each target node receives exactly three directed edges from the vertices of
    a containing, or best nearby, source triangle. The final edge-attribute
    column stores fixed geometric barycentric weights for diagnostics and
    optional fixed interpolation. Learned GraphCast decoders should still use a
    message passing operator over this map.

    This v0 method searches nearby face centers and nearby source vertices,
    tests candidate source faces in chunks, and uses planar barycentric
    coordinates in each triangular face's 3D plane. It is a local approximation
    to spherical barycentric interpolation, not exact spherical computational
    geometry.
    """

    logger.info(
        "Building barycentric map %s: source=%s(%d) target=%s(%d) faces=%d",
        name,
        source_mesh.name,
        source_mesh.n_nodes,
        target.name,
        target.n_nodes,
        int(torch.as_tensor(source_faces).shape[0]),
    )
    faces = torch.as_tensor(source_faces, dtype=torch.long).detach().cpu()
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"source_faces must have shape [F, 3], got {tuple(faces.shape)}.")
    if faces.shape[0] == 0:
        raise ValueError("source_faces must contain at least one triangular face.")
    if faces.numel():
        min_index = int(faces.min().item())
        max_index = int(faces.max().item())
        if min_index < 0 or max_index >= source_mesh.n_nodes:
            raise ValueError(
                f"source_faces contain vertex indices outside [0, {source_mesh.n_nodes}): "
                f"min={min_index}, max={max_index}."
            )

    if candidate_faces_k <= 0:
        raise ValueError("candidate_faces_k must be positive.")
    if candidate_vertices_k < 0:
        raise ValueError("candidate_vertices_k must be non-negative.")
    if target_chunk_size <= 0:
        raise ValueError("target_chunk_size must be positive.")

    selected_vertices: torch.Tensor
    selected_weights: torch.Tensor
    fallback_count = 0
    fast_path = None

    self_selection = _try_build_self_barycentric_selection(source_mesh, faces, target)
    if self_selection is not None:
        selected_vertices, selected_weights = self_selection
        fast_path = "matching_source_target_vertices"
    else:
        selected_vertices, selected_weights, fallback_count = _select_barycentric_vertices_chunked(
            source_xyz=source_mesh.xyz,
            faces=faces,
            target_xyz=target.xyz,
            candidate_faces_k=min(int(candidate_faces_k), int(faces.shape[0])),
            candidate_vertices_k=min(int(candidate_vertices_k), source_mesh.n_nodes),
            target_chunk_size=int(target_chunk_size),
        )

    sender = selected_vertices.reshape(-1).to(torch.long)
    receiver = torch.arange(target.n_nodes, dtype=torch.long).repeat_interleave(3)
    weight_tensor = selected_weights.reshape(-1, 1).to(torch.float32)
    geometry_attr = compute_edge_features(source_mesh.xyz, target.xyz, sender, receiver)
    edge_attr = torch.cat(
        (
            geometry_attr,
            weight_tensor,
        ),
        dim=-1,
    )
    metadata = {
        "kind": "barycentric",
        "weight_column": int(geometry_attr.shape[1]),
        "source_faces": int(faces.shape[0]),
        "edges_per_target": 3,
        "barycentric_type": "local_planar_3d",
        "fallback_count": int(fallback_count),
        "candidate_faces_k": int(min(candidate_faces_k, int(faces.shape[0]))),
        "candidate_vertices_k": int(min(candidate_vertices_k, source_mesh.n_nodes)),
        "target_chunk_size": int(target_chunk_size),
        "fast_path": fast_path,
        "approximation": (
            "local planar 3D barycentric coordinates on nearby face-center and vertex-adjacent "
            "candidate mesh faces; "
            "falls back to inverse-distance weights over nearest source vertices when needed"
        ),
    }
    edge_set = EdgeSet(
        name=name,
        sender=sender,
        receiver=receiver,
        edge_attr=edge_attr,
        directed=True,
        metadata=metadata,
    )
    graph_map = SphericalMap(
        name=name,
        source=source_mesh,
        target=target,
        edge_set=edge_set,
        metadata=metadata,
    )
    logger.info(
        "Built barycentric map %s: edges=%d fallback_count=%d fast_path=%s",
        name,
        graph_map.n_edges,
        int(fallback_count),
        fast_path,
    )
    return graph_map


def _resolve_chord_radius(
    *,
    source: SphericalGraph,
    target: SphericalGraph,
    radius: float | None,
    radius_meters: float | Literal["auto"] | None,
    earth_radius_meters: float,
    radius_query_fraction_edge_length: float,
    auto_radius_graph: Literal["source", "target"],
    auto_qhull_options: str | None,
) -> tuple[float, float | None, dict[str, object]]:
    if radius is not None:
        if radius_meters not in (None, "auto"):
            raise ValueError("Specify radius or numeric radius_meters, not both.")
        if radius <= 0.0:
            raise ValueError("radius must be positive.")
        return float(radius), None, {"mode": "unit_sphere_chord"}

    if radius_meters == "auto":
        if radius_query_fraction_edge_length <= 0.0:
            raise ValueError("radius_query_fraction_edge_length must be positive.")
        if earth_radius_meters <= 0.0:
            raise ValueError("earth_radius_meters must be positive.")
        if auto_radius_graph == "source":
            reference_graph = source
        elif auto_radius_graph == "target":
            reference_graph = target
        else:
            raise ValueError("auto_radius_graph must be 'source' or 'target'.")

        max_edge_chord = _max_spherical_delaunay_edge_chord(
            reference_graph,
            qhull_options=auto_qhull_options,
        )
        chord_radius = float(radius_query_fraction_edge_length) * max_edge_chord
        meters = _chord_to_meters(chord_radius, earth_radius_meters)
        return (
            chord_radius,
            meters,
            {
                "mode": "graphcast_auto",
                "auto_radius_graph": auto_radius_graph,
                "auto_reference_graph_name": reference_graph.name,
                "auto_reference_max_edge_chord": max_edge_chord,
                "radius_query_fraction_edge_length": float(radius_query_fraction_edge_length),
            },
        )

    if radius_meters is None:
        raise ValueError("Specify radius, numeric radius_meters, or radius_meters='auto'.")
    if radius_meters <= 0.0:
        raise ValueError("radius_meters must be positive.")
    if earth_radius_meters <= 0.0:
        raise ValueError("earth_radius_meters must be positive.")
    chord_radius = _meters_to_chord(float(radius_meters), earth_radius_meters)
    return chord_radius, float(radius_meters), {"mode": "physical_meters"}


def _meters_to_chord(radius_meters: float, earth_radius_meters: float) -> float:
    angular_radius = float(radius_meters) / float(earth_radius_meters)
    return float(2.0 * math.sin(0.5 * angular_radius))


def _chord_to_meters(chord_radius: float, earth_radius_meters: float) -> float:
    clipped = max(-2.0, min(2.0, float(chord_radius)))
    angular_radius = 2.0 * math.asin(0.5 * clipped)
    return float(angular_radius * float(earth_radius_meters))


def _max_spherical_delaunay_edge_chord(
    graph: SphericalGraph,
    *,
    qhull_options: str | None,
) -> float:
    faces = build_spherical_delaunay_faces(
        graph,
        qhull_options=qhull_options,
        orient_outward=False,
    )
    undirected_edges: set[tuple[int, int]] = set()
    for a, b, c in faces.tolist():
        for u, v in ((a, b), (b, c), (c, a)):
            if u != v:
                undirected_edges.add((min(int(u), int(v)), max(int(u), int(v))))
    if not undirected_edges:
        raise ValueError("Could not infer a radius from a graph with no Delaunay edges.")
    edge_tensor = torch.tensor(sorted(undirected_edges), dtype=torch.long)
    distances = torch.linalg.norm(
        graph.xyz[edge_tensor[:, 0]] - graph.xyz[edge_tensor[:, 1]],
        dim=-1,
    )
    return float(distances.max().item())


def _orient_faces_outward(xyz: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    if faces.numel() == 0:
        return faces
    out = faces.clone()
    tri = xyz[out].to(torch.float32)
    normals = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    centers = tri.mean(dim=1)
    inward = torch.sum(normals * centers, dim=-1) < 0.0
    if bool(inward.any()):
        flipped = out[inward].clone()
        out[inward, 1] = flipped[:, 2]
        out[inward, 2] = flipped[:, 1]
    return out


def _build_vertex_to_face_adjacency(
    n_vertices: int,
    faces: torch.Tensor,
) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(n_vertices)]
    for face_idx, face in enumerate(faces.tolist()):
        for vertex_idx in face:
            adjacency[int(vertex_idx)].append(int(face_idx))
    return adjacency


def _build_padded_vertex_to_face_adjacency(
    n_vertices: int,
    faces: torch.Tensor,
) -> torch.Tensor:
    adjacency = _build_vertex_to_face_adjacency(n_vertices, faces)
    max_degree = max((len(face_indices) for face_indices in adjacency), default=0)
    padded = torch.full((n_vertices, max(1, max_degree)), -1, dtype=torch.long)
    for vertex_idx, face_indices in enumerate(adjacency):
        if face_indices:
            padded[vertex_idx, : len(face_indices)] = torch.tensor(face_indices, dtype=torch.long)
    return padded


def _try_build_self_barycentric_selection(
    source_mesh: SphericalGraph,
    faces: torch.Tensor,
    target: SphericalGraph,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if source_mesh.n_nodes != target.n_nodes:
        return None
    if not torch.allclose(source_mesh.xyz, target.xyz, atol=1.0e-6):
        return None

    selected_vertices = torch.full((source_mesh.n_nodes, 3), -1, dtype=torch.long)
    selected_weights = torch.zeros((source_mesh.n_nodes, 3), dtype=torch.float32)
    filled = torch.zeros((source_mesh.n_nodes,), dtype=torch.bool)
    for face in faces.tolist():
        for local_idx, vertex_idx in enumerate(face):
            vertex = int(vertex_idx)
            if not bool(filled[vertex]):
                selected_vertices[vertex] = torch.tensor(face, dtype=torch.long)
                selected_weights[vertex, local_idx] = 1.0
                filled[vertex] = True

    if not bool(filled.all()):
        return None
    return selected_vertices, selected_weights


def _select_barycentric_vertices_chunked(
    *,
    source_xyz: torch.Tensor,
    faces: torch.Tensor,
    target_xyz: torch.Tensor,
    candidate_faces_k: int,
    candidate_vertices_k: int,
    target_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if target_xyz.shape[0] == 0:
        empty_vertices = torch.empty((0, 3), dtype=torch.long)
        empty_weights = torch.empty((0, 3), dtype=torch.float32)
        return empty_vertices, empty_weights, 0

    face_centers = _normalized_face_centers(source_xyz, faces)
    nearest_faces = _nearest_point_indices(
        face_centers,
        target_xyz,
        k=candidate_faces_k,
        chunk_size=target_chunk_size,
    )
    if candidate_vertices_k > 0:
        vertex_to_faces = _build_padded_vertex_to_face_adjacency(source_xyz.shape[0], faces)
        nearest_vertices = _nearest_point_indices(
            source_xyz,
            target_xyz,
            k=candidate_vertices_k,
            chunk_size=target_chunk_size,
        )
    else:
        vertex_to_faces = torch.empty((source_xyz.shape[0], 0), dtype=torch.long)
        nearest_vertices = torch.empty((target_xyz.shape[0], 0), dtype=torch.long)

    vertex_chunks: list[torch.Tensor] = []
    weight_chunks: list[torch.Tensor] = []
    fallback_count = 0
    for start in range(0, target_xyz.shape[0], target_chunk_size):
        stop = min(start + target_chunk_size, target_xyz.shape[0])
        candidate_chunks = [nearest_faces[start:stop]]
        if candidate_vertices_k > 0:
            incident_faces = vertex_to_faces[nearest_vertices[start:stop]].reshape(stop - start, -1)
            candidate_chunks.append(incident_faces)
        candidate_face_indices = torch.cat(candidate_chunks, dim=1)
        vertices, weights, fallback_mask = _choose_barycentric_vertices_for_points(
            points=target_xyz[start:stop],
            source_xyz=source_xyz,
            faces=faces,
            candidate_face_indices=candidate_face_indices,
        )
        vertex_chunks.append(vertices)
        weight_chunks.append(weights)
        fallback_count += int(fallback_mask.sum().item())

    return torch.cat(vertex_chunks, dim=0), torch.cat(weight_chunks, dim=0), fallback_count


def _normalized_face_centers(source_xyz: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    centers = source_xyz[faces].to(torch.float32).mean(dim=1)
    norm = torch.linalg.norm(centers, dim=-1, keepdim=True)
    if bool(torch.any(norm <= 1.0e-12)):
        raise ValueError("source_faces contain a degenerate face with near-zero center.")
    return centers / norm.clamp_min(1.0e-12)


def _nearest_point_indices(
    points_xyz: torch.Tensor,
    query_xyz: torch.Tensor,
    *,
    k: int,
    chunk_size: int,
) -> torch.Tensor:
    if k <= 0:
        return torch.empty((query_xyz.shape[0], 0), dtype=torch.long)
    if k > points_xyz.shape[0]:
        raise ValueError(f"k={k} exceeds point count {points_xyz.shape[0]}.")
    if query_xyz.shape[0] == 0:
        return torch.empty((0, k), dtype=torch.long)

    if cKDTree is not None:
        tree = cKDTree(points_xyz.numpy())
        _, indices = tree.query(query_xyz.numpy(), k=k)
        out = torch.as_tensor(indices, dtype=torch.long)
        if out.ndim == 1:
            out = out[:, None]
        return out

    chunks: list[torch.Tensor] = []
    points = points_xyz.to(torch.float32)
    for start in range(0, query_xyz.shape[0], chunk_size):
        stop = min(start + chunk_size, query_xyz.shape[0])
        distances = torch.cdist(query_xyz[start:stop].to(torch.float32), points)
        chunks.append(torch.topk(distances, k=k, dim=1, largest=False).indices.to(torch.long))
    return torch.cat(chunks, dim=0)


def _planar_barycentric_3d_batch(
    points: torch.Tensor,
    tri_xyz: torch.Tensor,
    *,
    eps: float = 1.0e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a = tri_xyz[:, :, 0, :]
    b = tri_xyz[:, :, 1, :]
    c = tri_xyz[:, :, 2, :]
    v0 = b - a
    v1 = c - a
    v2 = points[:, None, :] - a

    d00 = torch.sum(v0 * v0, dim=-1)
    d01 = torch.sum(v0 * v1, dim=-1)
    d11 = torch.sum(v1 * v1, dim=-1)
    d20 = torch.sum(v2 * v0, dim=-1)
    d21 = torch.sum(v2 * v1, dim=-1)
    denom = d00 * d11 - d01 * d01
    valid = torch.abs(denom) > eps
    safe_denom = torch.where(valid, denom, torch.ones_like(denom))

    v = (d11 * d20 - d01 * d21) / safe_denom
    w = (d00 * d21 - d01 * d20) / safe_denom
    u = 1.0 - v - w
    weights = torch.stack((u, v, w), dim=-1).to(torch.float32)
    reconstruction = torch.sum(weights[..., None] * tri_xyz.to(torch.float32), dim=-2)
    error = torch.linalg.norm(reconstruction - points[:, None, :].to(torch.float32), dim=-1)
    finite = torch.isfinite(weights).all(dim=-1) & torch.isfinite(error)
    return weights, error.to(torch.float32), valid & finite


def _choose_barycentric_vertices_for_points(
    *,
    points: torch.Tensor,
    source_xyz: torch.Tensor,
    faces: torch.Tensor,
    candidate_face_indices: torch.Tensor,
    inside_tol: float = 1.0e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if candidate_face_indices.ndim != 2:
        raise ValueError("candidate_face_indices must have shape [N, K].")
    if candidate_face_indices.shape[0] != points.shape[0]:
        raise ValueError("candidate_face_indices first dimension must match points.")
    if candidate_face_indices.shape[1] == 0:
        raise ValueError("candidate_face_indices must contain at least one candidate per point.")

    valid_candidates = candidate_face_indices >= 0
    safe_face_indices = candidate_face_indices.clamp_min(0)
    face_vertices = faces[safe_face_indices]
    tri_xyz = source_xyz[face_vertices]
    raw_weights, raw_error, valid_barycentric = _planar_barycentric_3d_batch(points, tri_xyz)

    inside = torch.all(raw_weights >= -inside_tol, dim=-1)
    negative_penalty = torch.relu(-raw_weights).sum(dim=-1)
    outside_score = 1.0e9 + negative_penalty * 1.0e6 + raw_error
    score = torch.where(inside, raw_error, outside_score)
    score = torch.where(
        valid_candidates & valid_barycentric,
        score,
        torch.full_like(score, float("inf")),
    )

    best_score, best_position = torch.min(score, dim=1)
    row = torch.arange(points.shape[0], dtype=torch.long)
    selected_vertices = face_vertices[row, best_position].to(torch.long)
    selected_raw_weights = raw_weights[row, best_position]
    clamped = selected_raw_weights.clamp_min(0.0)
    weight_sum = clamped.sum(dim=-1, keepdim=True)
    selected_weights = clamped / weight_sum.clamp_min(1.0e-12)
    fallback_mask = (
        ~torch.isfinite(best_score)
        | ~torch.isfinite(selected_weights).all(dim=-1)
        | (weight_sum.squeeze(-1) <= 1.0e-12)
    )

    if bool(fallback_mask.any()):
        fallback_vertices, fallback_weights = _inverse_distance_fallback_batch(
            points=points[fallback_mask],
            source_xyz=source_xyz,
        )
        selected_vertices[fallback_mask] = fallback_vertices
        selected_weights[fallback_mask] = fallback_weights

    return selected_vertices, selected_weights.to(torch.float32), fallback_mask


def _inverse_distance_fallback_batch(
    *,
    points: torch.Tensor,
    source_xyz: torch.Tensor,
    eps: float = 1.0e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    k = min(3, source_xyz.shape[0])
    distances = torch.cdist(points.to(torch.float32), source_xyz.to(torch.float32))
    nearest = torch.topk(distances, k=k, largest=False, dim=1).indices.to(torch.long)
    if k < 3:
        repeat_count = (3 + k - 1) // k
        nearest = nearest.repeat(1, repeat_count)[:, :3]
    nearest_distances = distances.gather(1, nearest).clamp_min(eps)
    inverse = 1.0 / nearest_distances
    weights = inverse / inverse.sum(dim=1, keepdim=True).clamp_min(eps)
    exact = nearest_distances[:, 0] <= eps
    if bool(exact.any()):
        weights[exact] = 0.0
        weights[exact, 0] = 1.0
    return nearest, weights.to(torch.float32)


def _nearest_vertex_indices(
    source_mesh: SphericalGraph,
    target: SphericalGraph,
    k: int,
) -> torch.Tensor:
    if cKDTree is not None:
        tree = cKDTree(source_mesh.xyz.numpy())
        _, indices = tree.query(target.xyz.numpy(), k=k)
        out = torch.as_tensor(indices, dtype=torch.long)
        if out.ndim == 1:
            out = out[:, None]
        return out
    dist = torch.cdist(target.xyz, source_mesh.xyz)
    return torch.topk(dist, k=k, dim=1, largest=False).indices.to(torch.long)


def _planar_barycentric_3d(
    point: torch.Tensor,
    tri_xyz: torch.Tensor,
    *,
    eps: float = 1.0e-12,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return planar 3D barycentric weights and reconstruction error."""

    a = tri_xyz[0]
    b = tri_xyz[1]
    c = tri_xyz[2]
    v0 = b - a
    v1 = c - a
    v2 = point - a
    d00 = torch.dot(v0, v0)
    d01 = torch.dot(v0, v1)
    d11 = torch.dot(v1, v1)
    d20 = torch.dot(v2, v0)
    d21 = torch.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if torch.abs(denom) <= eps:
        return None
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    weights = torch.stack((u, v, w)).to(torch.float32)
    if not torch.isfinite(weights).all():
        return None
    reconstruction = torch.sum(weights[:, None] * tri_xyz, dim=0)
    error = torch.linalg.norm(reconstruction - point).to(torch.float32)
    if not torch.isfinite(error):
        return None
    return weights, error


def _choose_barycentric_vertices_for_point(
    *,
    point: torch.Tensor,
    source_xyz: torch.Tensor,
    faces: torch.Tensor,
    candidate_face_indices: list[int],
    inside_tol: float = 1.0e-5,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if not candidate_face_indices:
        raise ValueError("Cannot choose barycentric face from an empty candidate set.")

    best: tuple[tuple[int, float, float], torch.Tensor, torch.Tensor] | None = None
    for face_idx in candidate_face_indices:
        face = faces[face_idx]
        tri = source_xyz[face]
        barycentric = _planar_barycentric_3d(point, tri)
        if barycentric is None:
            continue
        raw_weights, raw_error = barycentric
        clamped = raw_weights.clamp_min(0.0)
        weight_sum = clamped.sum()
        if weight_sum <= 1.0e-12:
            continue
        weights = clamped / weight_sum
        reconstruction_error = float(raw_error.item())
        penalty = float(torch.relu(-raw_weights).sum().item())
        outside = 0 if bool(torch.all(raw_weights >= -inside_tol)) else 1
        score = (outside, penalty, reconstruction_error)
        if best is None or score < best[0]:
            best = (score, face.to(torch.long), weights)

    if best is None:
        return _inverse_distance_fallback(point=point, source_xyz=source_xyz)
    return best[1], best[2], False


def _inverse_distance_fallback(
    *,
    point: torch.Tensor,
    source_xyz: torch.Tensor,
    eps: float = 1.0e-12,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    k = min(3, source_xyz.shape[0])
    distances = torch.linalg.norm(source_xyz - point, dim=-1)
    nearest = torch.topk(distances, k=k, largest=False).indices.to(torch.long)
    if k < 3:
        nearest = nearest.repeat((3 + k - 1) // k)[:3]
    nearest_distances = distances[nearest].clamp_min(eps)
    if float(nearest_distances[0].item()) <= eps:
        weights = torch.zeros((3,), dtype=torch.float32)
        weights[0] = 1.0
    else:
        inverse = 1.0 / nearest_distances
        weights = (inverse / inverse.sum()).to(torch.float32)
    return nearest.to(torch.long), weights, True
