"""Xarray boundary checks and small explicit training datasets. No data downloads."""

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset


def month_features(time):
    angle = 2 * np.pi * (time.dt.month.values - 1) / 12
    return torch.tensor(
        np.stack((np.sin(angle), np.cos(angle)), -1), dtype=torch.float32
    )


def validate_months(time):
    serial = time.dt.year.values * 12 + time.dt.month.values
    if time.ndim != 1 or len(serial) == 0 or (np.diff(serial) != 1).any():
        raise ValueError("Time must contain unique, contiguous, increasing months")


def next_month(time):
    """Month-start valid time, retaining a cftime calendar when present."""
    if hasattr(time, "calendar"):
        year, month = time.year + (time.month == 12), time.month % 12 + 1
        return type(time)(year, month, 1)
    return (pd.Timestamp(time).to_period("M") + 1).to_timestamp().to_datetime64()


def co2_values(co2, time):
    """Scalar means constant; a trajectory must cover every requested month."""
    if np.isscalar(co2):
        values = np.full(len(time), float(co2))
    elif isinstance(co2, xr.DataArray) and co2.dims == ("time",):
        if co2.attrs.get("units", "ppm") not in ("ppm", "ppmv", "1e-6", "1.e-6"):
            raise ValueError(
                "CO2 must already be in ppm; no unit conversion is inferred"
            )
        # Match monthly labels, preserving the physical data's calendar.
        source = co2.time.dt.year.values * 12 + co2.time.dt.month.values
        target = time.dt.year.values * 12 + time.dt.month.values
        if len(np.unique(source)) != len(source):
            raise ValueError("CO2 contains duplicate months")
        lookup = dict(zip(source.tolist(), np.asarray(co2.values).tolist()))
        missing = [int(t) for t in target if int(t) not in lookup]
        if missing:
            raise ValueError(
                f"CO2 trajectory lacks {len(missing)} requested months; no extrapolation is performed"
            )
        values = np.array([lookup[int(t)] for t in target])
    else:
        raise TypeError("co2 must be a positive scalar or monthly xarray DataArray")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("CO2 must be finite positive ppm")
    return xr.DataArray(
        values, dims="time", coords={"time": time}, attrs={"units": "ppm"}
    )


def validate_fields(ds, names, grid=None):
    validate_months(ds.time)
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


def validate_units(ds, specifications):
    """Accept notation aliases, never convert numerical values silently."""
    aliases = {
        "degrees C": "degC",
        "degree_Celsius": "degC",
        "degrees_Celsius": "degC",
        "dimensionless": "1",
        "fraction": "1",
    }
    for spec in specifications:
        if not spec.units:
            continue
        actual = ds[spec.name].attrs.get("units", "")
        expected = aliases.get(spec.units, spec.units).replace("^", "")
        normalized = aliases.get(actual, actual).replace("^", "")
        if normalized != expected:
            raise ValueError(
                f"{spec.name} units {actual!r} differ from expected {spec.units!r}; supply canonical units before training/inference"
            )


class PhysicalStates(Dataset):
    """Lazy physical fields, transformed one sample at a time (no eager duplicate)."""

    def __init__(self, atmosphere, ocean, preprocessor, co2):
        self.datasets = {"atmosphere": atmosphere, "ocean": ocean}
        self.preprocessor = preprocessor
        self.month = month_features(atmosphere.time)
        self.co2 = torch.tensor(
            ((co2 - preprocessor.co2_mean) / preprocessor.co2_std).values,
            dtype=torch.float32,
        )[:, None]

    def __len__(self):
        return len(self.month)

    def __getitem__(self, index):
        result = {"month": self.month[index], "co2": self.co2[index]}
        for component, ds in self.datasets.items():
            fields = self.preprocessor.encode(ds.isel(time=index), component).compute()
            names = [
                v.name for v in self.preprocessor.variables if v.component == component
            ]
            result[component] = torch.from_numpy(
                np.stack([fields[n].transpose("lat", "lon").values for n in names])
            )
        return result


class LatentSequences(Dataset):
    """Chronological windows, with two initial states and H future targets."""

    def __init__(self, atmosphere, ocean, time, steps=3):
        validate_months(time)
        if steps < 1 or len(time) < steps + 2:
            raise ValueError("Not enough adjacent states for the rollout horizon")
        self.atmosphere, self.ocean = atmosphere, ocean
        self.month, self.steps = month_features(time), steps

    def __len__(self):
        return len(self.month) - self.steps - 1

    def __getitem__(self, i):
        s = slice(i, i + self.steps + 2)
        return {
            "atmosphere": self.atmosphere[s],
            "ocean": self.ocean[s],
            "month": self.month[s],
        }
