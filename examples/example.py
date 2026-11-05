import numpy as np
import torch

from monthly_diffusion.architecture.geometry.mesh import (
    make_equiangular_grid_from_xarray,
    make_icosphere,
)
from monthly_diffusion.architecture.geometry.connectivity import build_radius_map


# 1. Latitude/longitude coordinates in degrees.
# Replace these with ds.lat.values and ds.lon.values.
lat = np.arange(-89.5, 90.0, 1.0)   # [H=180]
lon = np.arange(0.5, 360.0, 1.0)    # [W=360]


# 2. Construct the grid's spherical point representation.
# This preserves the supplied latitude/longitude ordering.
grid = make_equiangular_grid_from_xarray(lat, lon, name="grid")

# grid.xyz: [H*W, 3], Cartesian coordinates on the unit sphere.
# Flattening convention: node_index = latitude_index * W + longitude_index.
# No grid-to-grid connectivity is constructed here.


# 3. Construct an icosphere, returned as a SphericalGraph plus its faces.
mesh, faces = make_icosphere(level=3, name="mesh")

# mesh.xyz: [642, 3], Cartesian coordinates on the unit sphere.
# faces:    [1280, 3], three vertex indices per triangular face.
# Level L has 10 * 4**L + 2 vertices.


# 4. For each mesh node, find all grid points within 500 km.
# This radius is an example choice, not a tuned model parameter.
grid_to_mesh = build_radius_map(
    source=grid,
    target=mesh,
    radius_meters=500_000.0,
    name="grid_to_mesh",
)

# Each edge carries information FROM a grid point TO a mesh node.
sender = grid_to_mesh.edge_set.sender      # [E], indices into grid.xyz
receiver = grid_to_mesh.edge_set.receiver  # [E], indices into mesh.xyz

edge_index = torch.stack([sender, receiver])  # [2, E], COO connectivity


# 5. Inspect the incoming neighborhood of mesh node 0.
neighbors_of_node_0 = sender[receiver == 0]

print("Grid points:", grid.xyz.shape)
print("Mesh points:", mesh.xyz.shape)
print("Mesh faces:", faces.shape)
print("Grid → mesh edges:", edge_index.shape)
print("Grid neighbors of mesh node 0:", neighbors_of_node_0)

# Number of incoming grid points per mesh node.
neighbor_counts = torch.bincount(receiver, minlength=mesh.n_nodes)
print("Neighbors per mesh node:", neighbor_counts.min().item(), "to", neighbor_counts.max().item())