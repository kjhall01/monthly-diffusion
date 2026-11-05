import torch 
from torch import nn
import torch.nn.functional as F
import xarray as xr 
import numpy as np 
from dataclasses import dataclass

from .geometry import make_icosphere, build_barycentric_map, build_radius_map, build_mesh_adjacency_map, make_equiangular_grid_from_xarray
from .layers.ecc import EdgeConditionedConv


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
        train_da: xr.DataArray, 
        latent_channels: int = 62,
        grid_kernel_hidden_dim: int = 128,
        time: str = 'time',
        lat: str = "lat", 
        lon: str = "lon", 
        variable: str = "variable"
    ):
        super().__init__()

        self.fit_preprocessing(train_da, time, lat, lon, variable)
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


    def fit_preprocessing(self, train_da, time='time', lat="lat", lon="lon", variable="variable"):

        self.coords = {
            dim: train_da.coords[dim].copy(deep=True)
            for dim in (lat, lon, variable)
        }

        lati = train_da[lat].values
        longi = train_da[lon].values
        self.grid = make_equiangular_grid_from_xarray(lati, longi, name="grid")
        self.variables = train_da[variable].values 
        self.channels = train_da.sizes[variable]

        # data: xr.DataArray with dims (time, lat, lon, variable)
        self.latitude_weights_xr = np.cos(np.deg2rad(train_da[lat])).broadcast_like(
            train_da.isel(**{time: 0, variable:0}, drop=True)
        ).transpose("lat", "lon")
        self.latitude_weights_xr = self.latitude_weights_xr / self.latitude_weights_xr.sum() 
        
        self.register_buffer('latitude_weights_torch', torch.tensor(
                self.latitude_weights_xr.values,
                dtype=torch.float32,
            ).reshape(1, -1, 1)
        )
        # [1, H*W, 1]

        self.valid_mask_xr = (
            np.isfinite(train_da)
            .all(dim=time)
            .transpose(lat, lon, variable)
        )
        # [H, W, C]

        self.register_buffer('valid_mask_torch', torch.tensor(
                self.valid_mask_xr.values,
                dtype=torch.bool,
            ).reshape(1, -1, train_da.sizes[variable])
        )
        # [1, H*W, C]

        loss_weights = (
            self.latitude_weights_torch * self.valid_mask_torch
        )
        # [1, H*W, C]

        # normalize so variables with invalid points are weighted the same
        self.register_buffer('loss_weights', loss_weights / (
                loss_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
            )
        )

        # Exclude invalid locations before calculating statistics.
        valid_train = train_da.where(self.valid_mask_xr)
        weighted = valid_train.weighted(self.latitude_weights_xr)

        self.mean_xr = weighted.mean(dim=(time, lat, lon))
        self.std_xr = weighted.std(dim=(time, lat, lon))
        # Both: [variable]

        self.register_buffer('mean_torch', torch.tensor(
                self.mean_xr.values, dtype=torch.float32
            ).reshape(1, 1, -1)
        )

        self.register_buffer('std_torch', torch.tensor(
                self.std_xr.values, dtype=torch.float32
            ).reshape(1, 1, -1) 
        )


    def xr_to_torch(self, da, *, device=None, time='time', lat="lat", lon="lon", variable="variable"):
        """[time, lat, lon, variable] → float32 [B, H*W, C].

        Converts layout only; does not normalize or fill missing values.
        """
        da = da.transpose(time, lat, lon, variable)

        # Ensure values correspond to the geometry and channel ordering.
        for dim in (lat, lon, variable):
            if not da.get_index(dim).equals(self.coords[dim].to_index()):
                raise ValueError(f"{dim} coordinates do not match training data")

        values = np.array(da.values, dtype=np.float32, order="C", copy=True)

        tensor = torch.from_numpy(values).reshape(
            da.sizes[time],
            da.sizes[lat] * da.sizes[lon],
            da.sizes[variable],
        )
        return tensor.to(device=device) if device is not None else tensor


    def torch_to_xr(self, tensor, *, times, name=None, time='time', lat="lat", lon="lon", variable="variable"):
        """[B, H*W, C] → xr.DataArray [time, lat, lon, variable].

        Converts layout only; detaches gradients and transfers to CPU.
        """
        nlat = self.coords[lat].size
        nlon = self.coords[lon].size
        nvar = self.coords[variable].size

        if tensor.ndim != 3 or tuple(tensor.shape[1:]) != (nlat * nlon, nvar):
            raise ValueError(
                f"Expected [B, {nlat * nlon}, {nvar}], got {tuple(tensor.shape)}"
            )

        time_values = np.asarray(times)
        if time_values.ndim != 1 or len(time_values) != tensor.shape[0]:
            raise ValueError("Supply one timestamp per batch sample")

        values = (
            tensor.detach()
            .to(device="cpu", dtype=torch.float32)
            .reshape(tensor.shape[0], nlat, nlon, nvar)
            .numpy()
            .copy()
        )

        return xr.DataArray(
            values,
            dims=(time, lat, lon, variable),
            coords={time: time_values, **self.coords},
            name=name,
        )

    def forward(self, state):
        # state: [B, H*W, C]
        h = state

        for i, down in enumerate(self.down_layers):
            h = F.silu(down(h, None, self.down_maps[i]))

            for refine in self.refiner_layers[i]:
                h = F.silu(refine(h, h, self.refiner_maps[i]))

        return self.latent_projection(h)

    def preprocess_torch(self, x):
        return (x - self.mean_torch) / self.std_torch

    def postprocess_torch(self, x):
        return x * self.std_torch + self.mean_torch
