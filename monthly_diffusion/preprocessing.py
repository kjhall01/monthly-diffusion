"""Physical transforms, variable specifications and fixed latent normalization."""

import logging
from dataclasses import dataclass
import numpy as np
import torch
import xarray as xr
from scipy.special import expit
from .rollout import validate_months

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Variable:
    name: str
    component: str
    units: str = ""
    transform: str = "identity"
    scale: float = 1.0
    eps: float = 1.0e-6
    weight: float = 1.0  # Historical paired training only; component scripts ignore it.


def select_variables(names, component):
    """Keep the requested order; use known transforms or identity for a custom field.

    Pass a Variable instead of a name to override units or transform.
    """
    defaults = {variable.name: variable for variable in mdv1_variables()}
    selected = [
        (
            name
            if isinstance(name, Variable)
            else defaults.get(name, Variable(name, component))
        )
        for name in names
    ]
    if not selected or len({variable.name for variable in selected}) != len(selected):
        raise ValueError("Select at least one variable, with no duplicate names")
    if any(variable.component != component for variable in selected):
        raise ValueError(f"All selected variables must belong to {component}")
    return selected


def mdv1_variables():
    """Canonical names are model-facing; source adapters must rename explicitly."""
    atmosphere = [
        ("sfcWind", "m s^-1"),
        ("t2m", "K"),
        ("tcwv", "kg m^-2"),
        ("mtpr", "kg m^-2 s^-1"),
        ("mslp", "Pa"),
        ("olr", "W m^-2"),
        ("z500", "m^2 s^-2"),
        ("tau300_700", "m^2 s^-2"),
        ("t850", "K"),
        ("t500", "K"),
        ("u200", "m s^-1"),
        ("v200", "m s^-1"),
        ("u50", "m s^-1"),
        ("ssrd", "W m^-2"),
        ("tau_x", "N m^-2"),
        ("tau_y", "N m^-2"),
    ]
    ocean = [
        ("sst", "degrees C"),
        ("ssh", "m"),
        ("sss", "psu"),
        ("mld", "m"),
        ("sic", "dimensionless"),
        ("sivol", "m"),
        ("d20", "m"),
        ("ohc300", "J m^-2"),
        ("ohc700_minus_ohc300", "J m^-2"),
    ]
    scales = dict(
        sfcWind=5.0, tcwv=10.0, mtpr=1.0e-5, ssrd=100.0, mld=5.0, d20=50.0, sivol=0.1
    )
    result = []
    for component, fields in (("atmosphere", atmosphere), ("ocean", ocean)):
        for name, units in fields:
            weight = (
                0.5 / 14
                if component == "atmosphere" and name not in ("tau_x", "tau_y")
                else 0.05
            )
            if name in ("sic", "sivol"):
                weight /= 2
            result.append(
                Variable(
                    name,
                    component,
                    units,
                    (
                        "logit"
                        if name == "sic"
                        else "softplus_inverse" if name in scales else "identity"
                    ),
                    scales.get(name, 1.0),
                    1.0e-4 if name == "sic" else 1.0e-6,
                    weight,
                )
            )
    return result


def transform(x, spec):
    if spec.transform == "identity":
        return x
    if spec.transform == "log1p":
        return np.log1p(x / spec.scale)
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
    if spec.transform == "log1p":
        return spec.scale * np.expm1(y)
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
    """Fixed training statistics in transformed coordinates.

    ``gridpoint`` fits a time mean/std at each cell. ``annual_zonal`` and
    ``global`` keep the historical single standard deviation per variable.
    With ``mask_constant_cells``, gridpoint training excludes exact constants,
    except fully ice-covered SIC cells. Decoding restores excluded constants
    to their fitted physical values; genuinely missing cells remain missing.
    """

    def __init__(
        self,
        variables,
        centering="annual_zonal",
        ocean_support="per_variable",
        *,
        gridpoint_std_floor=0.05,
        mask_constant_cells=False,
    ):
        self.variables, self.centering, self.stats = variables, centering, {}
        if centering not in ("annual_zonal", "global", "gridpoint"):
            raise ValueError("centering must be annual_zonal, global or gridpoint")
        if not np.isfinite(gridpoint_std_floor) or not 0 < gridpoint_std_floor <= 1:
            raise ValueError("gridpoint_std_floor must be a fraction in (0, 1]")
        self.gridpoint_std_floor = gridpoint_std_floor
        if mask_constant_cells and centering != "gridpoint":
            raise ValueError("mask_constant_cells requires gridpoint normalization")
        self.mask_constant_cells = mask_constant_cells
        if ocean_support not in ("per_variable", "sst"):
            raise ValueError("ocean_support must be per_variable or sst")
        self.ocean_support = ocean_support
        self.ocean_support_mask = None

    def for_component(self, component):
        """Take one component's fitted transforms from an older combined model."""
        from copy import deepcopy

        result = deepcopy(self)
        result.variables = [
            variable for variable in self.variables if variable.component == component
        ]
        result.stats = {
            variable.name: result.stats[variable.name] for variable in result.variables
        }
        if result.stats:
            # Combined exports may contain different normalization for each AE.
            first = next(iter(result.stats.values()))
            result.centering = first.get(
                "normalization",
                "annual_zonal" if np.ndim(first["center"]) else "global",
            )
            if result.centering == "gridpoint":
                result.gridpoint_std_floor = first["std_floor_fraction"]
            result.mask_constant_cells = "constant_mask" in first
        return result

    def save(self, path):
        from .checkpoints import atomic_save

        atomic_save(self, path)

    @classmethod
    def load(cls, path):
        from .checkpoints import load_checkpoint

        result = load_checkpoint(path)
        if not isinstance(result, cls):
            raise TypeError("This file does not contain fitted preprocessing")
        return result

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
        self._fit({"atmosphere": atmosphere, "ocean": ocean}, statics, co2)

    def fit_component(self, fields, component, statics, co2):
        """Fit one AE's statistics without reading or preparing the other component."""
        if component not in ("atmosphere", "ocean") or any(
            spec.component != component for spec in self.variables
        ):
            raise ValueError("All variables must belong to the supplied component")
        self._fit({component: fields}, statics, co2)

    def _fit(self, datasets, statics, co2):
        if self.stats:
            raise RuntimeError(
                "Preprocessing is already fitted; construct a fresh instance to refit"
            )
        if (
            "ocean" in datasets
            and getattr(self, "ocean_support", "per_variable") == "sst"
        ):
            ocean = datasets["ocean"]
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
            datasets = dict(datasets, ocean=self.mask_ocean(ocean))
        for spec in self.variables:
            ds = datasets[spec.component]
            raw = ds[spec.name].astype("float64").where(np.isfinite(ds[spec.name]))
            x = transform(raw, spec)
            if self.centering == "gridpoint":
                x = x.transpose("time", "lat", "lon")
                count = x.count("time")
                # Explicit sums avoid nanstd warnings at always-missing cells.
                # Zero placeholders below are never used as observed targets.
                center = x.sum("time", skipna=True) / count.clip(min=1)
                variance = ((x - center) ** 2).sum("time", skipna=True) / count.clip(
                    min=1
                )
                fitted = xr.Dataset(dict(center=center, variance=variance, count=count))
                if getattr(self, "mask_constant_cells", False):
                    # Equal extrema mean exactly zero std, without roundoff from
                    # subtracting a nonzero mean. Use raw values before any floor.
                    fitted["minimum"] = raw.min("time", skipna=True)
                    fitted["maximum"] = raw.max("time", skipna=True)
                fitted = fitted.compute()
                mask = fitted["count"] >= 2
                if not bool(mask.any()):
                    raise ValueError(
                        f"No cells with two finite training values for {spec.name}"
                    )
                # A relative floor: a fraction of this variable's
                # area-weighted RMS local temporal std, plus the legacy tiny floor.
                area = np.cos(np.deg2rad(x.lat)).clip(min=0)
                typical_std = (
                    float(fitted.variance.where(mask).weighted(area).mean()) ** 0.5
                )
                if not np.isfinite(typical_std):
                    raise ValueError(f"Nonfinite training variance for {spec.name}")
                floor = max(self.gridpoint_std_floor * typical_std, 1.0e-6)
                self.stats[spec.name] = dict(
                    center=fitted.center,
                    std=np.sqrt(fitted.variance).clip(min=floor),
                    mask=mask,
                    attrs=dict(ds[spec.name].attrs),
                    normalization="gridpoint",
                    std_floor=floor,
                    std_floor_fraction=self.gridpoint_std_floor,
                )
                if getattr(self, "mask_constant_cells", False):
                    constant = mask & (fitted.minimum == fitted.maximum)
                    # A permanent concentration of 1 is still a learning target.
                    excluded = constant
                    if spec.name == "sic":
                        excluded = constant & (fitted.minimum != 1)
                    active = mask & ~excluded
                    if not bool(active.any()):
                        raise ValueError(
                            f"No nonconstant training cells for {spec.name}"
                        )
                    self.stats[spec.name].update(
                        mask=active,
                        data_mask=mask,
                        constant_mask=excluded,
                        constant_value=fitted.minimum.where(excluded, 0.0),
                    )
                continue
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
        self.training_times = next(iter(datasets.values())).time.values.copy()
        self.training_co2 = co2.copy(deep=True)
        self.source_attrs = {name: dict(ds.attrs) for name, ds in datasets.items()}

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
                if s.get("normalization") == "gridpoint":
                    # Also mask validation/future cells with insufficient TRAINING
                    # support. Store this per variable for mixed-component exports.
                    out[spec.name] = out[spec.name].where(s["mask"])
        return xr.Dataset(out)

    def decode(self, ds, component, *, mask=True):
        out = {}
        for spec in self.variables:
            if spec.component == component:
                s = self.stats[spec.name]
                field = inverse_transform(ds[spec.name] * s["std"] + s["center"], spec)
                if mask:
                    if "constant_mask" in s:
                        field = xr.where(s["constant_mask"], s["constant_value"], field)
                    field = field.where(s.get("data_mask", s["mask"]))
                out[spec.name] = field.transpose(*ds[spec.name].dims)
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

    def decode_tensor(self, values, component, *, mask=True, time=None):
        """Undo normalization and nonlinear transforms on [batch, channel, lat, lon].

        This uses Torch on the input device, including MPS. Physical RMSE must
        compare this result with the ORIGINAL targets, before transform clipping.
        """
        variables = [
            variable for variable in self.variables if variable.component == component
        ]
        if values.ndim != 4 or values.shape[1] != len(variables):
            raise ValueError("Expected [batch, selected variable, latitude, longitude]")
        fields = []
        for index, variable in enumerate(variables):
            stats = self.stats[variable.name]
            center = torch.as_tensor(
                np.asarray(stats["center"]), device=values.device, dtype=values.dtype
            )
            if center.ndim == 1:
                center = center[:, None]
            scale = torch.as_tensor(
                np.asarray(stats["std"]), device=values.device, dtype=values.dtype
            )
            field = values[:, index] * scale + center
            if variable.transform == "log1p":
                field = variable.scale * torch.expm1(field)
            elif variable.transform == "softplus_inverse":
                field = variable.scale * torch.logaddexp(field, torch.zeros_like(field))
            elif variable.transform == "logit":
                field = torch.sigmoid(field)
            elif variable.transform != "identity":
                raise ValueError(f"Unknown transform {variable.transform}")
            if mask:
                if "constant_mask" in stats:
                    constant = torch.as_tensor(
                        np.asarray(stats["constant_mask"]),
                        device=values.device,
                        dtype=torch.bool,
                    )
                    physical_value = torch.as_tensor(
                        np.asarray(stats["constant_value"]),
                        device=values.device,
                        dtype=values.dtype,
                    )
                    field = torch.where(constant, physical_value, field)
                support = stats.get("data_mask", stats["mask"])
                if (
                    component == "ocean"
                    and getattr(self, "ocean_support", "per_variable") == "sst"
                ):
                    support = support & self.ocean_support_mask
                support = torch.as_tensor(
                    np.asarray(support), device=values.device, dtype=torch.bool
                )
                field = field.masked_fill(~support, float("nan"))
            fields.append(field)
        return torch.stack(fields, dim=1)


class LatentNormalizer:
    """One fixed population mean/std per channel, fitted after AE selection."""

    @property
    def fitted(self):
        return hasattr(self, "mean") and hasattr(self, "std")

    def fit(self, values):
        if hasattr(self, "mean"):
            raise RuntimeError("Latent statistics are immutable once fitted")
        if not torch.isfinite(values).all():
            raise FloatingPointError(
                "Cannot fit latent statistics on nonfinite encoder outputs"
            )
        # Fit once in CPU float64; MPS does not support float64 tensors.
        statistics = values.detach().to(device="cpu", dtype=torch.float64)
        self.mean = statistics.mean((0, 1)).float()
        self.std = statistics.std((0, 1), correction=0).float().clamp_min(1.0e-4)
        return self

    def normalize(self, values):
        if not self.fitted:
            raise RuntimeError(
                "Fit the latent normalizer on selected training latents first"
            )
        return (values - self.mean.to(values)) / self.std.to(values)

    def denormalize(self, values):
        if not self.fitted:
            raise RuntimeError(
                "Fit the latent normalizer on selected training latents first"
            )
        return values * self.std.to(values) + self.mean.to(values)


def validate_fields(ds, names, grid=None, *, cadence="monthly"):
    if cadence == "monthly":
        validate_months(ds.time)
    elif cadence == "6h":
        from .rollout import validate_six_hourly

        validate_six_hourly(ds.time)
    else:
        raise ValueError("cadence must be monthly or 6h")
    for name in names:
        if set(ds[name].dims) != {"time", "lat", "lon"}:
            raise ValueError(
                f"{name} must have time, lat, lon dimensions; select a single training member"
            )
    for coord in ("lat", "lon"):
        if ds[coord].ndim != 1:
            raise ValueError(
                "The public physical-data API currently requires a regular lat/lon grid"
            )
        if grid is not None and not np.array_equal(
            ds[coord].values, grid[coord].values
        ):
            raise ValueError(f"{coord} does not match the prepared grid")


def normalized_unit_name(units):
    """Normalize accepted spellings only; do not convert physical values."""
    aliases = {
        "degrees C": "degC",
        "degree_Celsius": "degC",
        "degrees_Celsius": "degC",
        "dimensionless": "1",
        "fraction": "1",
    }
    return aliases.get(units, units).replace("^", "")


def validate_units(ds, specifications):
    """Accept notation aliases, never convert numerical values silently."""
    for spec in specifications:
        if not spec.units:
            continue
        actual = ds[spec.name].attrs.get("units", "")
        expected = normalized_unit_name(spec.units)
        normalized = normalized_unit_name(actual)
        if normalized != expected:
            raise ValueError(
                f"{spec.name} units {actual!r} differ from expected {spec.units!r}; supply canonical units before training/inference"
            )


def fit_initialization_moments(model, atmos, ocean):
    """Fit in normalized field coordinates; never change weights or normalization."""
    if not model.preprocessor.stats:
        raise RuntimeError(
            "Prepare or load a model before fitting initialization moments"
        )
    dates = model.preprocessor.training_times
    atmos, ocean = atmos.sel(time=dates), ocean.sel(time=dates)
    model._validate_pair(atmos, ocean)
    for c, ds in (("atmosphere", atmos), ("ocean", ocean)):
        if dict(ds.attrs) != model.preprocessor.source_attrs[c]:
            raise ValueError(
                "Initialization source metadata differs from checkpoint training data"
            )
    moments = {}
    for component, fields in (("atmosphere", atmos), ("ocean", ocean)):
        moments.update(component_moments(model.preprocessor, fields, component))
    model.initialization_moments = moments
    return model


def component_moments(preprocessor, fields, component):
    """Training-month/gridpoint field moments for one component's initializer."""
    dates = preprocessor.training_times
    fields = fields.sel(time=dates)
    moments = {}
    normalized = preprocessor.encode(fields, component)
    for name in normalized.data_vars:
        x = normalized[name].transpose("time", "lat", "lon").compute().astype("float64")
        monthly = []
        for month, samples in x.where(np.isfinite(x)).groupby("time.month"):
            count = samples.count("time")
            supported = count.values >= 2
            mean = xr.full_like(count, np.nan, dtype="float32")
            std = xr.full_like(count, np.nan, dtype="float32")
            # Land and other unsupported cells stay NaN. Reduce only cells with
            # enough observations, so nanstd never sees an empty time series.
            if supported.any():
                values = samples.values[:, supported]
                mean.values[supported] = np.nanmean(values, axis=0)
                std.values[supported] = np.nanstd(values, axis=0, ddof=0)
            monthly.append(
                xr.Dataset(
                    dict(mean=mean, std=std, count=count.astype("int16"))
                ).expand_dims(month=[month])
            )
        moments[name] = xr.concat(monthly, dim="month").reindex(month=np.arange(1, 13))
        moments[name].attrs.update(
            component=component,
            coordinate_system="checkpoint normalized field",
            training_start=str(dates[0]),
            training_end=str(dates[-1]),
            training_months=len(dates),
            sd_ddof=0,
        )
        log.info("Fitted stochastic-initialization moments: %s", name)
    return moments
