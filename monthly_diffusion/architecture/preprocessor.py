import torch 
from torch import nn
import xarray as xr 
import numpy as np 
from .geometry import make_icosphere, build_barycentric_map, build_radius_map, build_mesh_adjacency_map, make_equiangular_grid_from_xarray


class Preprocessor(nn.Module):

    def __init__(
        self,
        train_da: xr.DataArray, 
        time: str = 'time',
        lat: str = "lat", 
        lon: str = "lon", 
        variable: str = "variable"
    ):
        super().__init__()
       
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


        # Fixed background: average over training time and longitude.
        # Retains latitude dependence; no month dependence.
        zonal_mean = valid_train.mean(dim=(time, lon))
        # [lat, variable]

        # Broadcast that background onto every longitude.
        self.zonal_mean_xr = (
            zonal_mean
            .broadcast_like(valid_train.isel({time: 0}, drop=True))
            .transpose(lat, lon, variable)
        )
        # [lat, lon, variable]

        # A latitude band may have no valid values for a variable.
        # Those locations are excluded by the validity mask anyway.
        self.zonal_mean_xr = self.zonal_mean_xr.fillna(0.0)



        weighted = (valid_train - self.zonal_mean_xr).weighted(self.latitude_weights_xr)

        self.mean_xr = weighted.mean(dim=(time, lat, lon))
        self.std_xr = weighted.std(dim=(time, lat, lon))
        # Both: [variable]

        self.register_buffer(
            "zonal_mean_torch",
            torch.tensor(
                self.zonal_mean_xr.values,
                dtype=torch.float32,
            ).reshape(1, -1, self.channels),
        )
        # [1, H*W, C]

        self.register_buffer('mean_torch', torch.tensor(
                self.mean_xr.values, dtype=torch.float32
            ).reshape(1, 1, -1)
        )

        self.register_buffer('std_torch', torch.tensor(
                self.std_xr.values, dtype=torch.float32
            ).reshape(1, 1, -1) 
        )

        # [time, lat, lon, variable], with invalid locations excluded.
        valid_train = train_da.where(self.valid_mask_xr)



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

    def preprocess_torch(self, x):
        x = x - self.zonal_mean_torch
        filled = torch.where(self.valid_mask_torch, x, self.mean_torch)
        ret = (filled - self.mean_torch) / self.std_torch
        return ret

    def postprocess_torch(self, prediction):
        physical = prediction * self.std_torch + self.mean_torch
        physical = physical.masked_fill(~self.valid_mask_torch, float("nan"))
        return physical + self.zonal_mean_torch
