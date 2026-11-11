"""Audit whether the ocean branch has land-crossing geometric pathways.

This is deliberately a geometry/data audit, not a model diagnostic.  It uses
SST validity as a common ocean-support mask, propagates that support through
the actual grid-to-mesh and mesh-down maps, and reports mixed/land-supported
same-mesh edges and example Pacific--Atlantic shortest paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from monthly_diffusion.modules.mesh import build_mesh_hierarchy_context


def _target_support(graph_map, source_support):
    receiver = graph_map.edge_set.receiver
    sender = graph_map.edge_set.sender
    values = torch.zeros(graph_map.n_target, dtype=torch.float64)
    counts = torch.zeros(graph_map.n_target, dtype=torch.float64)
    values.scatter_add_(0, receiver, source_support[sender])
    counts.scatter_add_(0, receiver, torch.ones_like(receiver, dtype=torch.float64))
    return values / counts.clamp_min(1.0)


def _edge_summary(graph_map, support, threshold=0.5):
    source = support[graph_map.edge_set.sender] >= threshold
    target = support[graph_map.edge_set.receiver] >= threshold
    return {
        "edges": int(source.numel()),
        "ocean_to_ocean": int((source & target).sum()),
        "ocean_to_land": int((source & ~target).sum()),
        "land_to_ocean": int((~source & target).sum()),
        "land_to_land": int((~source & ~target).sum()),
        "cross_boundary_fraction": float((source != target).double().mean()),
    }


def _shortest_path(graph, start, goal):
    """Unweighted undirected shortest path over a mesh adjacency map."""
    sender = graph.edge_set.sender.numpy()
    receiver = graph.edge_set.receiver.numpy()
    adjacency = [[] for _ in range(graph.n_target)]
    for a, b in zip(sender, receiver):
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))
    previous = np.full(graph.n_target, -1, dtype=np.int64)
    previous[start] = start
    queue = [start]
    for node in queue:
        if node == goal:
            break
        for neighbor in adjacency[node]:
            if previous[neighbor] < 0:
                previous[neighbor] = node
                queue.append(neighbor)
    if previous[goal] < 0:
        return []
    path = [goal]
    while path[-1] != start:
        path.append(int(previous[path[-1]]))
    return path[::-1]


def _region_indices(graph, lat_range, lon_ranges):
    lat = graph.lat.numpy()
    lon = graph.lon.numpy() % 360.0
    lon_mask = np.zeros(lon.shape, dtype=bool)
    for lo, hi in lon_ranges:
        lon_mask |= (lon >= lo) & (lon <= hi)
    return np.flatnonzero((lat >= lat_range[0]) & (lat <= lat_range[1]) & lon_mask)


def audit(label, source_path, mesh_levels, nlat, nlon, rotation):
    ds = xr.open_zarr(source_path, consolidated=True)
    valid = torch.from_numpy(np.isfinite(ds["sst"].isel(time=0).values).ravel()).double()
    geometry = build_mesh_hierarchy_context(
        nlat=nlat, nlon=nlon, mesh_levels=mesh_levels,
        rotation_degrees=rotation,
    )
    supports = {"grid": valid}
    maps = {}
    first = geometry.maps["grid_to_mesh_l0"]
    supports["mesh_l0"] = _target_support(first, valid)
    maps[first.name] = {
        "targets": first.n_target,
        "mean_ocean_support": float(supports["mesh_l0"].mean()),
        "p10_ocean_support": float(torch.quantile(supports["mesh_l0"], 0.10)),
        "p50_ocean_support": float(torch.quantile(supports["mesh_l0"], 0.50)),
        "p90_ocean_support": float(torch.quantile(supports["mesh_l0"], 0.90)),
        "zero_support_targets": int((supports["mesh_l0"] == 0).sum()),
    }
    for i in range(len(mesh_levels) - 1):
        name = f"mesh_l{i}_to_mesh_l{i+1}"
        graph_map = geometry.maps[name]
        supports[f"mesh_l{i+1}"] = _target_support(graph_map, supports[f"mesh_l{i}"])
        maps[name] = {
            "targets": graph_map.n_target,
            "mean_ocean_support": float(supports[f"mesh_l{i+1}"].mean()),
            "p10_ocean_support": float(torch.quantile(supports[f"mesh_l{i+1}"], 0.10)),
            "p50_ocean_support": float(torch.quantile(supports[f"mesh_l{i+1}"], 0.50)),
            "p90_ocean_support": float(torch.quantile(supports[f"mesh_l{i+1}"], 0.90)),
            "zero_support_targets": int((supports[f"mesh_l{i+1}"] == 0).sum()),
        }

    edges = {}
    for i, level in enumerate(mesh_levels):
        name = f"mesh_l{i}_to_mesh_l{i}"
        edges[name] = _edge_summary(geometry.maps[name], supports[f"mesh_l{i}"])

    finest = geometry.graphs["mesh_l0"]
    pacific = _region_indices(finest, (-20, 20), [(145, 250)])
    atlantic = _region_indices(finest, (-20, 20), [(285, 360), (0, 25)])
    routes = []
    if len(pacific) and len(atlantic):
        # Use the closest pair in mesh hops, then report support along it.
        best = None
        for start in pacific:
            for goal in atlantic:
                path = _shortest_path(geometry.maps["mesh_l0_to_mesh_l0"], int(start), int(goal))
                if path and (best is None or len(path) < len(best)):
                    best = path
        if best:
            support = supports["mesh_l0"][best]
            coords = finest.lat.numpy()[best], finest.lon.numpy()[best] % 360.0
            routes.append({
                "mesh": "mesh_l0",
                "hops": len(best) - 1,
                "path_node_count": len(best),
                "minimum_ocean_support": float(support.min()),
                "mean_ocean_support": float(support.mean()),
                "land_supported_nodes": int((support < 0.5).sum()),
                "coordinates_lat_lon": [[float(a), float(b)] for a, b in zip(*coords)],
            })
    return {
        "label": label,
        "source": str(Path(source_path).resolve()),
        "grid_sst_ocean_fraction": float(valid.mean()),
        "mesh_levels": list(mesh_levels),
        "maps": maps,
        "same_mesh_edges": edges,
        "example_pacific_atlantic_routes": routes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--one-degree", type=Path, required=True)
    parser.add_argument("--five-degree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [
        audit("1deg", args.one_degree, (4, 3, 2), 180, 360, (17, 23, 11)),
        audit("5deg", args.five_degree, (3, 2), 36, 72, (17, 23, 11)),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
