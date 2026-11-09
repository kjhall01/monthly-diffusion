import torch 
from torch import nn
import xarray as xr 
import numpy as np 
from ..architecture.geometry import SphericalGraph, make_equiangular_grid_from_xarray


class Preprocessor(nn.Module):
    """Fit training statistics; compact=True keeps points ever observed in training.

    Compact callers should request the per-sample mask and use masked_mse().
    """

    @classmethod
    def from_state_dict(cls, state_dict, coords):
        """Restore fitted statistics without reading or fitting evaluation fields.

        Coordinates must come from the original training grid, in channel order.
        Older checkpoints save these buffers but not coordinates, so callers can
        read coordinates alone from the original training NetCDF.
        """
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self.coords = {dim: coords[dim].copy(deep=True) for dim in ("lat", "lon", "variable")}
        self.variables = self.coords["variable"].values
        self.channels = len(self.variables)
        nlat, nlon = len(self.coords["lat"]), len(self.coords["lon"])
        self.grid = make_equiangular_grid_from_xarray(
            self.coords["lat"].values, self.coords["lon"].values, name="grid"
        )
        self._set_grid_points(state_dict.get("point_indices"))
        npoints = self.grid.n_nodes
        shapes = {
            "latitude_weights_torch": (1, npoints, 1),
            "valid_mask_torch": (1, npoints, self.channels),
            "loss_weights": (1, npoints, self.channels),
            "zonal_mean_torch": (1, npoints, self.channels),
            "mean_torch": (1, 1, self.channels),
            "std_torch": (1, 1, self.channels),
        }
        for name, shape in shapes.items():
            value = state_dict[name].detach().cpu().clone()
            if tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid saved preprocessing buffer: {name}")
            self.register_buffer(name, value)
        self.load_state_dict(state_dict, strict=True)
        if not (self.std_torch > 0).all():
            raise ValueError("Saved training standard deviations must be positive")
        if not torch.allclose(self.loss_weights.sum(1), torch.ones(1, self.channels)):
            raise ValueError("Saved spatial loss weights must sum to one per variable")
        self.latitude_weights_xr = xr.DataArray(
            self._full_grid(self.latitude_weights_torch, 0).numpy().reshape(nlat, nlon),
            dims=("lat", "lon"),
            coords={dim: self.coords[dim] for dim in ("lat", "lon")},
        )
        for attr, buffer in (("valid_mask_xr", "valid_mask_torch"),
                             ("zonal_mean_xr", "zonal_mean_torch")):
            setattr(self, attr, xr.DataArray(
                self._full_grid(getattr(self, buffer), 0).numpy().reshape(nlat, nlon, self.channels),
                dims=("lat", "lon", "variable"), coords=self.coords,
            ))
        for attr, buffer in (("mean_xr", "mean_torch"), ("std_xr", "std_torch")):
            setattr(self, attr, xr.DataArray(
                getattr(self, buffer).numpy().reshape(self.channels),
                dims="variable", coords={"variable": self.coords["variable"]},
            ))
        return self

    def __init__(
        self,
        train_da: xr.DataArray, 
        time: str = 'time',
        lat: str = "lat", 
        lon: str = "lon", 
        variable: str = "variable",
        *,
        compact: bool = False,
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

        finite = np.isfinite(train_da)
        if compact:
            # Keep a point if any variable is observed at any training time.
            support = finite.any(dim=(time, variable)).transpose(lat, lon)
            self.valid_mask_xr = support.broadcast_like(
                train_da.isel({time: 0}, drop=True)
            ).transpose(lat, lon, variable)
            self._set_grid_points(torch.from_numpy(np.flatnonzero(support.values)))
        else:
            # Preserve the existing atmosphere/checkpoint preprocessing policy.
            self.valid_mask_xr = finite.all(dim=time).transpose(lat, lon, variable)
            self._set_grid_points(None)

        # data: xr.DataArray with dims (time, lat, lon, variable)
        self.latitude_weights_xr = np.cos(np.deg2rad(train_da[lat])).broadcast_like(
            train_da.isel(**{time: 0, variable:0}, drop=True)
        ).transpose("lat", "lon")
        self.latitude_weights_xr = self.latitude_weights_xr / self.latitude_weights_xr.sum() 
        
        self.register_buffer('latitude_weights_torch', self.select_points(torch.tensor(
                self.latitude_weights_xr.values,
                dtype=torch.float32,
            ).reshape(1, -1, 1))
        )
        # [1, retained points, 1]

        self.register_buffer('valid_mask_torch', self.select_points(torch.tensor(
                self.valid_mask_xr.values,
                dtype=torch.bool,
            ).reshape(1, -1, train_da.sizes[variable]))
        )
        # [1, retained points, C]

        loss_weights = (
            self.latitude_weights_torch * self.valid_mask_torch
        )
        # [1, retained points, C]

        # normalize so variables with invalid points are weighted the same
        self.register_buffer('loss_weights', loss_weights / (
                loss_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
            )
        )

        # Exclude invalid locations before calculating statistics.
        valid_train = train_da.where(self.valid_mask_xr & finite)


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

        # A latitude band may have no observations for a variable.
        # Use a zero background there; missing targets are excluded from loss.
        self.zonal_mean_xr = self.zonal_mean_xr.fillna(0.0)



        weighted = (valid_train - self.zonal_mean_xr).weighted(self.latitude_weights_xr)

        self.mean_xr = weighted.mean(dim=(time, lat, lon))
        self.std_xr = weighted.std(dim=(time, lat, lon))
        # Both: [variable]

        self.register_buffer(
            "zonal_mean_torch",
            self.select_points(torch.tensor(
                self.zonal_mean_xr.values,
                dtype=torch.float32,
            ).reshape(1, -1, self.channels)),
        )
        # [1, retained points, C]

        self.register_buffer('mean_torch', torch.tensor(
                self.mean_xr.values, dtype=torch.float32
            ).reshape(1, 1, -1)
        )

        self.register_buffer('std_torch', torch.tensor(
                self.std_xr.values, dtype=torch.float32
            ).reshape(1, 1, -1) 
        )

    def _set_grid_points(self, indices):
        """Bind saved full-grid indices; None keeps legacy full-grid checkpoints."""
        if indices is not None:
            indices = indices.detach().cpu().clone()
            if (indices.dtype != torch.long or indices.ndim != 1 or not indices.numel()
                    or indices[0] < 0 or indices[-1] >= self.grid.n_nodes
                    or not (indices[1:] > indices[:-1]).all()):
                raise ValueError("Grid point indices must be nonempty, sorted and unique")
            grid = self.grid
            self.grid = SphericalGraph(
                name=grid.name, xyz=grid.xyz[indices], lat=grid.lat[indices], lon=grid.lon[indices],
                metadata={**grid.metadata, "kind": "masked_equiangular_grid",
                          "flat_index": indices,
                          "lat_index": grid.metadata["lat_index"][indices],
                          "lon_index": grid.metadata["lon_index"][indices]},
            )
        self.register_buffer("point_indices", indices)

    def select_points(self, tensor, dim=1):
        """Gather full-grid state, statics or weights in the model's node order."""
        if self.point_indices is None:
            return tensor
        return tensor.index_select(dim, self.point_indices.to(tensor.device))

    def _full_grid(self, tensor, fill_value=float("nan")):
        if self.point_indices is None:
            return tensor
        npoints = len(self.coords["lat"]) * len(self.coords["lon"])
        full = tensor.new_full((tensor.shape[0], npoints, tensor.shape[2]), fill_value)
        return full.index_copy(1, self.point_indices.to(tensor.device), tensor)

    def xr_to_torch(self, da, *, device=None, time='time', lat="lat", lon="lon", variable="variable"):
        """[time, lat, lon, variable] → float32 [B, retained points, C].

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
        tensor = self.select_points(tensor)
        return tensor.to(device=device) if device is not None else tensor


    def torch_to_xr(self, tensor, *, times, name=None, time='time', lat="lat", lon="lon", variable="variable"):
        """[B, retained points, C] → full grid, with NaNs at dropped points.

        Converts layout only; detaches gradients and transfers to CPU.
        """
        nlat = self.coords[lat].size
        nlon = self.coords[lon].size
        nvar = self.coords[variable].size

        if tensor.ndim != 3 or tuple(tensor.shape[1:]) != (self.grid.n_nodes, nvar):
            raise ValueError(
                f"Expected [B, {self.grid.n_nodes}, {nvar}], got {tuple(tensor.shape)}"
            )

        time_values = np.asarray(times)
        if time_values.ndim != 1 or len(time_values) != tensor.shape[0]:
            raise ValueError("Supply one timestamp per batch sample")

        values = (
            self._full_grid(tensor.detach().to(device="cpu", dtype=torch.float32))
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

    def preprocess_torch(self, x, *, return_mask=False):
        """Normalize and zero-fill; optionally preserve availability for the loss."""
        observed = self.valid_mask_torch & torch.isfinite(x)
        x = x - self.zonal_mean_torch
        filled = torch.where(observed, x, self.mean_torch)
        ret = (filled - self.mean_torch) / self.std_torch
        return (ret, observed) if return_mask else ret

    def masked_mse(self, prediction, target, observed):
        """Per-sample/variable area-weighted MSE and a nonempty-target mask."""
        error = torch.where(observed, prediction - target, 0.0)
        weights = self.latitude_weights_torch * observed
        weight_sum = weights.sum(dim=1)
        mse = (error.square() * weights).sum(dim=1) / weight_sum.clamp_min(1e-12)
        return mse, weight_sum > 0

    def postprocess_torch(self, prediction):
        physical = prediction * self.std_torch + self.mean_torch
        physical = physical.masked_fill(~self.valid_mask_torch, float("nan"))
        return physical + self.zonal_mean_torch
