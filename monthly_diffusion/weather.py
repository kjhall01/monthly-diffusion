"""Weather AE preprocessing: supplied day/hour climatology and gridpoint scaling.

Only fitted statistics are saved. The climatology remains a lazy xarray Dataset
backed by the public Zarr store; at most two 30-day blocks live in RAM. Loading a
weather AE therefore needs access to that store when reconstructing physical data.
"""

from collections import OrderedDict
import logging

import numpy as np
import torch
import xarray as xr

from .preprocessing import Preprocessor

log = logging.getLogger(__name__)


class WeatherPreprocessor(Preprocessor):
    """Identity physical anomalies, with fixed training-only local mean/std."""

    def __init__(self, variables, climatology, *, gridpoint_std_floor=0.05):
        if any(v.transform != "identity" for v in variables):
            raise ValueError(
                "Weather anomalies use identity transforms, including precipitation"
            )
        super().__init__(
            variables,
            centering="gridpoint",
            gridpoint_std_floor=gridpoint_std_floor,
            mask_constant_cells=True,
        )
        self.climatology = climatology
        self._climatology_blocks = OrderedDict()
        if not np.array_equal(climatology.dayofyear, np.arange(1, 367)):
            raise ValueError("Expected WeatherBench dayofyear coordinates 1..366")
        if not np.array_equal(climatology.hour, [0, 6, 12, 18]):
            raise ValueError("Expected WeatherBench climatology hours 0, 6, 12, 18")

    def __getstate__(self):
        state = dict(self.__dict__)
        # Never embed downloaded climate blocks in a checkpoint.
        state["_climatology_blocks"] = OrderedDict()
        return state

    def baseline(self, time):
        """Use WB2's actual dayofyear convention, including day 366 in leap years."""
        if time is None:
            raise ValueError("Supply timestamps to restore the weather climatology")
        time = xr.DataArray(
            np.atleast_1d(time.values),
            dims="time",
            coords={"time": np.atleast_1d(time.values)},
        )
        if (
            (time.dt.hour % 6 != 0).any()
            or (time.dt.minute != 0).any()
            or (time.dt.second != 0).any()
        ):
            raise ValueError("Weather climatology requires exact six-hourly UTC times")
        groups = (time.dt.dayofyear.values - 1) // 30
        pieces = []
        for group in np.unique(groups):
            key = int(group)
            if key not in self._climatology_blocks:
                # Evict BEFORE fetching another block to bound peak memory.
                if len(self._climatology_blocks) == 2:
                    self._climatology_blocks.popitem(last=False)
                self._climatology_blocks[key] = self.climatology.isel(
                    dayofyear=slice(key * 30, min((key + 1) * 30, 366))
                ).compute(scheduler="threads", num_workers=4)
            self._climatology_blocks.move_to_end(key)
            dates = time.isel(time=np.flatnonzero(groups == group))
            pieces.append(
                self._climatology_blocks[key]
                .sel(dayofyear=dates.dt.dayofyear, hour=dates.dt.hour)
                .drop_vars(["dayofyear", "hour"])
            )
        return (
            xr.concat(pieces, dim="time").sel(time=time).transpose("time", "lat", "lon")
        )

    def fit_batches(self, batches, statics, training_time, source_attrs):
        """One streaming pass with stable, mergeable population moments.

        batches yields loaded physical Datasets. Neither fields nor normalized
        states are accumulated. Exact raw extrema identify training constants.
        """
        if self.stats:
            raise RuntimeError("Preprocessing is already fitted")
        moments = {}
        samples = 0
        for batch_number, physical in enumerate(batches, 1):
            baseline = self.baseline(physical.time)
            for spec in self.variables:
                raw = (
                    physical[spec.name]
                    .transpose("time", "lat", "lon")
                    .values.astype("float64")
                )
                valid = np.isfinite(raw)
                climate = baseline[spec.name].values
                if np.any(valid & ~np.isfinite(climate)):
                    raise ValueError(
                        f"Climatology is missing at observed {spec.name} cells"
                    )
                x = np.where(valid, raw - climate, 0)
                count = valid.sum(0)
                mean = x.sum(0) / count.clip(min=1)
                m2 = np.where(valid, (x - mean) ** 2, 0).sum(0)
                low = np.where(valid, raw, np.inf).min(0)
                high = np.where(valid, raw, -np.inf).max(0)
                if spec.name not in moments:
                    moments[spec.name] = [
                        count,
                        mean,
                        m2,
                        low,
                        high,
                        dict(physical[spec.name].attrs),
                    ]
                else:
                    old_count, old_mean, old_m2, old_low, old_high, attrs = moments[
                        spec.name
                    ]
                    total = old_count + count
                    delta = mean - old_mean
                    merged_mean = old_mean + delta * count / total.clip(min=1)
                    merged_m2 = (
                        old_m2 + m2 + delta**2 * old_count * count / total.clip(min=1)
                    )
                    moments[spec.name] = [
                        total,
                        merged_mean,
                        merged_m2,
                        np.minimum(old_low, low),
                        np.maximum(old_high, high),
                        attrs,
                    ]
            samples += physical.sizes["time"]
            if batch_number % 100 == 0:
                log.info(
                    "Fitting weather statistics: %d / %d timestamps",
                    samples,
                    len(training_time),
                )
        if samples != len(training_time):
            raise ValueError("Statistics stream did not cover all training timestamps")
        coords = dict(lat=statics.lat, lon=statics.lon)
        area = np.cos(np.deg2rad(statics.lat.values)).clip(min=0)[:, None]
        for spec in self.variables:
            count, mean, m2, low, high, attrs = moments[spec.name]
            data_mask = count >= 2
            variance = m2 / count.clip(min=1)
            typical_std = np.sqrt(
                (area * np.where(data_mask, variance, 0)).sum()
                / (area * data_mask).sum()
            )
            # Specific humidity aloft can vary by much less than 1e-6 kg/kg.
            # Keep the floor relative to each channel's residual variability.
            floor = max(self.gridpoint_std_floor * typical_std, 1e-12)
            constant = data_mask & (low == high)
            if spec.name == "sic":
                constant &= low != 1
            active = data_mask & ~constant
            if not active.any() or not np.isfinite(floor):
                raise ValueError(f"No usable training variability for {spec.name}")
            self.stats[spec.name] = dict(
                center=xr.DataArray(mean, dims=("lat", "lon"), coords=coords),
                std=xr.DataArray(
                    np.sqrt(variance).clip(min=floor),
                    dims=("lat", "lon"),
                    coords=coords,
                ),
                mask=xr.DataArray(active, dims=("lat", "lon"), coords=coords),
                data_mask=xr.DataArray(data_mask, dims=("lat", "lon"), coords=coords),
                constant_mask=xr.DataArray(
                    constant, dims=("lat", "lon"), coords=coords
                ),
                constant_value=xr.DataArray(
                    np.where(constant, low, 0), dims=("lat", "lon"), coords=coords
                ),
                attrs=attrs,
                normalization="gridpoint",
                std_floor=floor,
                std_floor_fraction=self.gridpoint_std_floor,
            )
        self.topo_mean, self.topo_std = float(statics.phi_s.mean()), max(
            float(statics.phi_s.std()), 1e-6
        )
        self.co2_mean, self.co2_std = (
            0.0,
            1.0,
        )  # weather model does not condition on CO2
        self.training_times = training_time.values.copy()
        self.source_attrs = {"atmosphere": dict(source_attrs)}

    def encode(self, fields, component):
        anomalies = fields - self.baseline(fields.time)
        return super().encode(anomalies, component)

    def decode(self, fields, component, *, mask=True):
        decoded = super().decode(fields, component, mask=False) + self.baseline(
            fields.time
        )
        for spec in self.variables:
            stats = self.stats[spec.name]
            if mask:
                decoded[spec.name] = xr.where(
                    stats["constant_mask"], stats["constant_value"], decoded[spec.name]
                ).where(stats["data_mask"])
            decoded[spec.name] = decoded[spec.name].transpose(*fields[spec.name].dims)
            decoded[spec.name].attrs = dict(stats["attrs"])
        return decoded

    def decode_tensor(self, values, component, *, mask=True, time=None):
        anomalies = super().decode_tensor(values, component, mask=False)
        baseline = self.baseline(time)
        climate = torch.as_tensor(
            np.stack([baseline[v.name].values for v in self.variables], axis=1),
            dtype=values.dtype,
            device=values.device,
        )
        if climate.shape != values.shape:
            raise ValueError("Supply one timestamp per weather prediction")
        result = anomalies + climate
        if mask:
            constants = torch.as_tensor(
                np.stack(
                    [self.stats[v.name]["constant_mask"].values for v in self.variables]
                ),
                device=values.device,
            )
            constant_values = torch.as_tensor(
                np.stack(
                    [
                        self.stats[v.name]["constant_value"].values
                        for v in self.variables
                    ]
                ),
                dtype=values.dtype,
                device=values.device,
            )
            support = torch.as_tensor(
                np.stack(
                    [self.stats[v.name]["data_mask"].values for v in self.variables]
                ),
                device=values.device,
            )
            result = torch.where(constants, constant_values, result).masked_fill(
                ~support, float("nan")
            )
        return result
