"""Xarray boundary checks and small explicit training datasets. No data downloads."""

import numpy as np
import torch
from torch.utils.data import Dataset


from monthly_diffusion.rollout import (
    month_features,
    validate_months,
    next_month as next_month,
    co2_values,
)


from monthly_diffusion.preprocessing import (
    validate_fields as validate_fields,
    normalized_unit_name as normalized_unit_name,
    validate_units as validate_units,
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


class ComponentStates(Dataset):
    """One physical component; the other component is never loaded per batch."""

    def __init__(self, fields, component, preprocessor, co2):
        if component not in ("atmosphere", "ocean"):
            raise ValueError("component must be atmosphere or ocean")
        self.fields = fields
        self.component = component
        self.preprocessor = preprocessor
        self.names = [
            v.name for v in preprocessor.variables if v.component == component
        ]
        self.month = month_features(fields.time)
        forcing = co2_values(co2, fields.time)
        self.co2 = torch.tensor(
            ((forcing - preprocessor.co2_mean) / preprocessor.co2_std).values,
            dtype=torch.float32,
        )[:, None]

    def __len__(self):
        return len(self.month)

    def __getitem__(self, index):
        # Load once; retain original physical targets before guarded transforms.
        physical = self.fields.isel(time=index).compute()
        normalized = self.preprocessor.encode(physical, self.component)
        state = np.stack(
            [normalized[name].transpose("lat", "lon").values for name in self.names]
        )
        if self.component == "ocean":
            physical = self.preprocessor.mask_ocean(physical)
        target = np.stack(
            [
                physical[name]
                # Physical metrics include restored constants, even if a cell
                # that was constant in training changes during validation.
                .where(
                    self.preprocessor.stats[name].get(
                        "data_mask", self.preprocessor.stats[name]["mask"]
                    )
                )
                .transpose("lat", "lon")
                .values
                for name in self.names
            ]
        ).astype("float32")
        return (
            torch.from_numpy(state),
            torch.from_numpy(target),
            self.month[index],
            self.co2[index],
        )


class LatentSequences(Dataset):
    """Chronological windows, with two initial states and H future targets."""

    def __init__(self, atmosphere, ocean, time, steps=3, *, co2=None):
        validate_months(time)
        if steps < 1 or len(time) < steps + 2:
            raise ValueError("Not enough adjacent states for the rollout horizon")
        self.atmosphere, self.ocean = atmosphere, ocean
        self.month, self.steps = month_features(time), steps
        self.co2 = (
            None
            if co2 is None
            else torch.as_tensor(np.asarray(co2), dtype=torch.float32)
        )
        if self.co2 is not None and self.co2.shape != (len(time),):
            raise ValueError("Supply one aligned CO2 ppm value per latent month")

    def __len__(self):
        return len(self.month) - self.steps - 1

    def __getitem__(self, i):
        s = slice(i, i + self.steps + 2)
        values = self.atmosphere[s], self.ocean[s], self.month[s]
        return values if self.co2 is None else (*values, self.co2[s])
