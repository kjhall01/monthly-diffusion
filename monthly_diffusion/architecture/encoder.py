import torch 
from torch import nn
import torch.nn.functional as F
import xarray as xr 
import numpy as np 
from dataclasses import dataclass

from .geometry import SphericalGraph, make_icosphere, build_barycentric_map, build_radius_map, build_mesh_adjacency_map, make_equiangular_grid_from_xarray
from .layers.ecc import EdgeConditionedConv
from .preprocessor import Preprocessor

@dataclass(frozen=True)
class MeshStage:
    name: str
    refinement: int
    width: int
    refiners: int = 4
    activation: str = "silu"
    kernel_mlp_hidden_dim: int = 128
    map_to_next_stage: str = "radius"

class RefinedIcosahedralMeshEncoder(nn.Module):

    def __init__(
        self,
        hierarchy: list,
        preprocessor: Preprocessor,
        latent_channels: int = 256,
        channels: int = 16,
        grid_kernel_hidden_dim: int = 128,
        time: str = 'time',
        lat: str = "lat", 
        lon: str = "lon", 
        variable: str = "variable"
    ):
        super().__init__()

        self.channels = channels 
        self.latent_channels = latent_channels
        self.preprocessor = preprocessor
        self.grid = self.preprocessor.grid 
        self.n_meshes = len(hierarchy)

        self.meshes = []
        self.faces = [] 

        for stage in hierarchy:
            mesh, faces = make_icosphere(level=stage.refinement, name=stage.name)     
            self.meshes.append(mesh)
            self.faces.append(faces)

        grid_to_mesh = build_radius_map(
            self.grid, self.meshes[0],
            name = f"grid_to_{hierarchy[0].name}"
        )

        self.down_maps = [grid_to_mesh] 
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
                self.down_maps.append(mapping)
        

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
       
        self.down_layers = [] 
        self.refiner_layers = []
        self.activations = []

        for _1, map in enumerate(self.down_maps):
            if _1 == 0: 
                down_ecc = EdgeConditionedConv(
                    in_dim=self.channels,
                    out_dim=hierarchy[0].width,
                    edge_attr_dim=map.edge_set.edge_attr_dim,
                    hidden_dim=grid_kernel_hidden_dim,
                    aggregation="mean",
                    residual=False,
                )
            else: 
                down_ecc = EdgeConditionedConv(
                    in_dim=hierarchy[_1-1].width,
                    out_dim=hierarchy[_1].width,
                    edge_attr_dim=map.edge_set.edge_attr_dim,
                    hidden_dim=hierarchy[_1].kernel_mlp_hidden_dim,
                    aggregation="mean",
                    residual=False,
                )
            self.down_layers.append(down_ecc)

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
        self.down_layers = nn.ModuleList(self.down_layers)
        self.latent_projection = nn.Linear(
            hierarchy[-1].width, latent_channels
        )



    def forward(self, state):
        # state: [B, H*W, C]
        h = state

        # h is already normalized and filled.
        # A grid node can send a message if it has at least one valid variable.
        grid_mask = self.preprocessor.valid_mask_torch.any(dim=-1).expand(h.shape[0], -1)

        for i, down in enumerate(self.down_layers):
            source_mask = grid_mask if i == 0 else None
            h = down(h, None, self.down_maps[i], source_mask=source_mask)
            h = F.silu(h)

            for refine in self.refiner_layers[i]:
                h = F.silu(refine(h, h, self.refiner_maps[i]))

        return self.latent_projection(h)
