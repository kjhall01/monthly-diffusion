###### build  mesh topology
lat = train_ds["lat"].values
lon = train_ds["lon"].values

grid = mdag.make_equiangular_grid_from_xarray(lat, lon, name="grid")
m4, m4_faces = mdag.make_icosphere(level=4, name="m4")      # most nodes
m3, m3_faces = mdag.make_icosphere(level=3, name="m3")      # 642 nodes
m2, m2_faces = mdag.make_icosphere(level=2, name="m2") # 162 nodes

# Radius-based neighborhoods for encoder.
grid_to_m4 = mdag.build_radius_map(
    grid, m4,
    name="grid_to_m4",
    radius_query_fraction_edge_length=1.0,
)

m4_to_m3 = mdag.build_radius_map(
    m4, m3,
    name="m4_to_m3",
    radius_query_fraction_edge_length=1.0,
)

m3_to_m2 = mdag.build_radius_map(
    m3, m2,
    name="m3_to_m2",
    radius_query_fraction_edge_length=1.0,
)

# Barycentric neighborhoods for decoder
m2_to_m3 = mdag.build_barycentric_map(
    m2, m2_faces, m3, name="m2_to_m3",
)
m3_to_m4 = mdag.build_barycentric_map(
    m3, m3_faces, m4, name="m3_to_m4",
)
m4_to_grid = mdag.build_barycentric_map(
    m4, m4_faces, grid, name="m4_to_grid",
)

# same-mesh neighborhoods

m2_to_m2 = mdag.build_mesh_adjacency_map(
    m2, m2_faces, bidirectional=True
)
m3_to_m3 = mdag.build_mesh_adjacency_map(
    m3, m3_faces, bidirectional=True
)
m4_to_m4 = mdag.build_mesh_adjacency_map(
    m4, m4_faces, bidirectional=True
)

# Check that every target receives information.
for mapping in (
    grid_to_m4, m4_to_m3, m3_to_m2, 
    m2_to_m3, m3_to_m4, m4_to_grid,
    m2_to_m2, m3_to_m3, m4_to_m4
):
    counts = torch.bincount(
        mapping.edge_set.receiver, minlength=mapping.n_target
    )
    assert torch.all(counts > 0), f"Unconnected targets: {mapping.name}"

up_maps = [m2_to_m3, m3_to_m4, m4_to_grid]
