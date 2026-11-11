"""Train-only physical transforms, annual zonal centers, and fixed latent units."""

import numpy as np
import torch
import xarray as xr
from scipy.special import expit


def transform(x, spec):
    if spec.transform == "identity":
        return x
    if spec.transform == "softplus_inverse":
        y = (
            (x / spec.scale).clip(min=spec.eps)
            if isinstance(x, xr.DataArray)
            else np.maximum(x / spec.scale, spec.eps)
        )
        return y + np.log(-np.expm1(-y))
    if spec.transform == "logit":
        y = (
            x.clip(min=spec.eps, max=1 - spec.eps)
            if isinstance(x, xr.DataArray)
            else np.clip(x, spec.eps, 1 - spec.eps)
        )
        return np.log(y) - np.log1p(-y)
    raise ValueError(f"Unknown transform {spec.transform}")


def inverse_transform(y, spec):
    if spec.transform == "identity":
        return y
    if spec.transform == "softplus_inverse":
        # NaNs encode unavailable cells, not numerical failures of softplus.
        with np.errstate(invalid="ignore"):
            return spec.scale * np.logaddexp(0.0, y)
    if spec.transform == "logit":
        return (
            xr.apply_ufunc(expit, y, dask="allowed")
            if isinstance(y, xr.DataArray)
            else expit(y)
        )
    raise ValueError(f"Unknown transform {spec.transform}")


class Preprocessor:
    def __init__(
        self, variables, centering="annual_zonal", ocean_support="per_variable"
    ):
        self.variables, self.centering, self.stats = variables, centering, {}
        if ocean_support not in ("per_variable", "sst"):
            raise ValueError("ocean_support must be per_variable or sst")
        self.ocean_support = ocean_support
        self.ocean_support_mask = None

    def mask_ocean(self, ds):
        """Apply fixed TRAINING SST support; preserve all additional NaNs.

        Physical zeros are never manufactured. Missing normalized values are
        zero-filled only at the encoder boundary, and excluded from the loss.
        This also accepts raw evaluation fields on the checkpoint's grid.
        """
        # Older serialized preprocessors have neither attribute.
        if getattr(self, "ocean_support", "per_variable") != "sst":
            return ds
        if self.ocean_support_mask is None:
            raise RuntimeError("Fit training SST support before masking ocean data")
        for coord in ("lat", "lon"):
            if not np.array_equal(ds[coord], self.ocean_support_mask[coord]):
                raise ValueError("Ocean support grid differs from input data")
        return ds.where(self.ocean_support_mask)

    def fit(self, atmosphere, ocean, statics, co2):
        """Inputs MUST already be restricted to training times; no split inference."""
        if self.stats:
            raise RuntimeError(
                "Preprocessing is already fitted; construct a fresh instance to refit"
            )
        if getattr(self, "ocean_support", "per_variable") == "sst":
            if "sst" not in ocean:
                raise ValueError("SST-based ocean support requires sst")
            self.ocean_support_mask = np.isfinite(ocean.sst).any("time").compute()
            if not bool(self.ocean_support_mask.any()):
                raise ValueError("No valid training SST cells")
            self.ocean_support_mask.attrs = dict(
                definition="finite SST in at least one training month; fixed for all ocean variables",
                fitted_start=str(ocean.time.values[0]),
                fitted_end=str(ocean.time.values[-1]),
            )
            ocean = self.mask_ocean(ocean)
        for spec in self.variables:
            ds = atmosphere if spec.component == "atmosphere" else ocean
            x = transform(ds[spec.name].astype("float64"), spec).where(
                np.isfinite(ds[spec.name])
            )
            if self.centering == "annual_zonal":
                center = x.mean(("time", "lon"), skipna=True).fillna(0.0).compute()
            elif self.centering == "global":
                center = x.mean(skipna=True).compute()
            else:
                raise ValueError("centering must be annual_zonal or global")
            scale = float((x - center).std(skipna=True, ddof=0).compute())
            if not np.isfinite(scale):
                raise ValueError(f"No finite training values for {spec.name}")
            self.stats[spec.name] = dict(
                center=center,
                std=max(scale, 1.0e-6),
                mask=np.isfinite(ds[spec.name]).any("time").compute(),
                attrs=dict(ds[spec.name].attrs),
            )
        self.co2_mean, self.co2_std = float(co2.mean()), max(float(co2.std()), 1.0e-6)
        self.topo_mean, self.topo_std = (
            float(statics.phi_s.mean()),
            max(float(statics.phi_s.std()), 1.0e-6),
        )
        self.training_times = atmosphere.time.values.copy()
        self.training_co2 = co2.copy(deep=True)
        self.source_attrs = dict(
            atmosphere=dict(atmosphere.attrs), ocean=dict(ocean.attrs)
        )

    def encode(self, ds, component):
        if component == "ocean":
            ds = self.mask_ocean(ds)
        out = {}
        for spec in self.variables:
            if spec.component == component:
                s = self.stats[spec.name]
                out[spec.name] = (
                    (transform(ds[spec.name], spec) - s["center"]) / s["std"]
                ).astype("float32")
        return xr.Dataset(out)

    def decode(self, ds, component, *, mask=True):
        out = {}
        for spec in self.variables:
            if spec.component == component:
                s = self.stats[spec.name]
                field = inverse_transform(ds[spec.name] * s["std"] + s["center"], spec)
                out[spec.name] = field.where(s["mask"]) if mask else field
                out[spec.name].attrs = s["attrs"]
                out[spec.name].attrs.setdefault("units", spec.units)
        result = xr.Dataset(out)
        return self.mask_ocean(result) if mask and component == "ocean" else result

    def static_tensor(self, ds):
        # Original data specification orders topography first, LSM second.
        array = np.stack(
            [((ds.phi_s - self.topo_mean) / self.topo_std).values, ds.lsm.values]
        )
        if not np.isfinite(array).all():
            raise ValueError("Static fields must be finite")
        return torch.from_numpy(array.astype("float32"))


class LatentNormalizer:
    """One fixed population mean/std per channel, fitted after AE selection."""

    def fit(self, values):
        if hasattr(self, "mean"):
            raise RuntimeError("Latent statistics are immutable once fitted")
        if not torch.isfinite(values).all():
            raise FloatingPointError(
                "Cannot fit latent statistics on nonfinite encoder outputs"
            )
        self.mean = values.double().mean((0, 1)).float()
        self.std = values.double().std((0, 1), correction=0).float().clamp_min(1.0e-4)
        return self

    def normalize(self, values):
        return (values - self.mean.to(values)) / self.std.to(values)

    def denormalize(self, values):
        return values * self.std.to(values) + self.mean.to(values)
