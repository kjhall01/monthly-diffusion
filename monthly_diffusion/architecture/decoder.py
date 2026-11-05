import torch 
from torch import nn
import torch.nn.functional as F
import xarray as xr 
import numpy as np 
from dataclasses import dataclass

from .layers.ecc import EdgeConditionedConv
from .geometry import SphericalGraph, make_icosphere, build_barycentric_map, build_radius_map, build_mesh_adjacency_map, make_equiangular_grid_from_xarray
from .preprocessor import Preprocessor



class RefinedIcosahedralMeshDecoder(nn.Module):

    def __init__(
        self,
        hierarchy: list,
        preprocessor: Preprocessor, 
        channels: int = 16,
        latent_channels: int = 256,
        grid_kernel_hidden_dim: int = 128,
        time: str = 'time',
        lat: str = "lat", 
        lon: str = "lon", 
        variable: str = "variable",
        n_statics: int = 2,
    ):
        super().__init__()

        self.n_meshes = len(hierarchy)
        self.latent_channels = latent_channels 
        self.channels = channels 
        self.preprocessor = preprocessor
        self.grid = self.preprocessor.grid 
        self.meshes = []
        self.faces = [] 
        for stage in hierarchy:
            mesh, faces = make_icosphere(level=stage.refinement, name=stage.name)     
            self.meshes.append(mesh)
            self.faces.append(faces)

        mesh_to_grid = build_barycentric_map(
            self.meshes[-1], self.faces[-1], self.grid,
            name = f"{hierarchy[0].name}_to_grid"
        )

        self.up_maps = [] 
        for _, stage in enumerate(hierarchy): 
            if _ < len(hierarchy)-1:
                if stage.map_to_next_stage == "radius":
                    mapping = build_radius_map(
                        self.meshes[_], self.meshes[_+1],
                        name = f"{stage.name}_to_{hierarchy[_+1].name}"
                    )
                elif stage.map_to_next_stage == "barycentric":
                    mapping = build_barycentric_map(
                        self.meshes[_], self.faces[_], self.meshes[_+1],
                        name = f"{stage.name}_to_{hierarchy[_+1].name}"
                    )
                else:
                    raise ValueError(f"invalid mesh-to-mesh transfer policy: {stage.map_to_next_stage}")
                self.up_maps.append(mapping)
        self.up_maps.append(mesh_to_grid)
        

        self.refiner_maps = [
            build_mesh_adjacency_map(
                mesh, faces,
                name=f"{stage.name}_to_{stage.name}",
                bidirectional=True,
            )
            for stage, mesh, faces in zip(
                hierarchy, self.meshes, self.faces
            )
        ]
       
        self.up_layers = [] 
        self.refiner_layers = []
        self.activations = []

        for _1, map in enumerate(self.up_maps):
            if _1 < len(self.up_maps)-1: 
                down_ecc = EdgeConditionedConv(
                    in_dim=hierarchy[_1].width,
                    out_dim=hierarchy[_1+1].width,
                    edge_attr_dim=map.edge_set.edge_attr_dim,
                    hidden_dim=hierarchy[-1].kernel_mlp_hidden_dim,
                    aggregation="mean",
                    residual=False,
                )
            else: 
                down_ecc = EdgeConditionedConv(
                    in_dim=hierarchy[-1].width,
                    out_dim=hierarchy[-1].width,
                    edge_attr_dim=map.edge_set.edge_attr_dim,
                    hidden_dim=grid_kernel_hidden_dim,
                    aggregation="mean",
                    residual=False,
                )
            self.up_layers.append(down_ecc)

            refiners_1 = nn.ModuleList([
                EdgeConditionedConv(
                    in_dim=hierarchy[_1].width,
                    out_dim=hierarchy[_1].width,
                    edge_attr_dim=self.refiner_maps[_1].edge_set.edge_attr_dim,
                    hidden_dim=hierarchy[_1].kernel_mlp_hidden_dim,
                    aggregation="mean",
                    residual=True,
                )
                for _ in range(hierarchy[_1].refiners)
            ])
            self.refiner_layers.append(refiners_1)

            self.activations.append(nn.SiLU())
        self.refiner_layers = nn.ModuleList(self.refiner_layers)
        self.up_layers = nn.ModuleList(self.up_layers)

        self.output_head = nn.Sequential(
            nn.Linear(hierarchy[-1].width + 3 + n_statics, 128),
            nn.SiLU(),
            nn.Linear(128, self.channels),
        )

        self.latent_expand = nn.Linear(self.latent_channels, hierarchy[0].width)


    def forward(self, latent, statics):
        h = self.latent_expand(latent)                  # [B, m2_nodes, m2_width]

        for i, up in enumerate(self.up_layers):
            for refine in self.refiner_layers[i]:
                h = refine(F.silu(h), h, self.refiner_maps[i])

            h = up(h, None, self.up_maps[i])

            if i < len(self.up_layers) - 1:
                h = F.silu(h)
        xyz = self.grid.xyz[None].expand(h.shape[0], -1, -1).to(h.device)
        statics = statics[None].expand(h.shape[0], -1, -1)

        reconstruction = self.output_head(
            torch.cat([h, xyz, statics], dim=-1)
        )
        return reconstruction                                       # [B, H*W, variables]

