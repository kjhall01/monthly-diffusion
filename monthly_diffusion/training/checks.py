"""Small checks shared by the editable training scripts."""

import numpy as np
import xarray as xr
from ..data.dataloader import co2_values, validate_fields, validate_units


def check_component_data(
    training, validation, statics, specifications, co2, saved=None
):
    if not specifications:
        raise ValueError("Select at least one variable")
    names = [spec.name for spec in specifications]
    for fields in (training, validation):
        validate_fields(fields, names, statics)
        validate_units(fields, specifications)
        co2_values(co2, fields.time)
    if np.intersect1d(training.time, validation.time).size:
        raise ValueError("Training and validation dates overlap")
    if saved is not None:
        preprocessor = saved["preprocessor"]
        if not np.array_equal(
            training.time, preprocessor.training_times
        ) or not np.array_equal(validation.time, saved["validation_times"]):
            raise ValueError("Training or validation dates differ from this run")
        if specifications != preprocessor.variables:
            raise ValueError("Variable definitions changed; choose a new run")
        xr.testing.assert_identical(statics, saved["statics"])
        xr.testing.assert_identical(
            co2_values(co2, training.time), preprocessor.training_co2
        )
        xr.testing.assert_identical(
            co2_values(co2, validation.time), saved["validation_co2"]
        )
        component = specifications[0].component
        if (
            dict(training.attrs) != preprocessor.source_attrs[component]
            or dict(validation.attrs) != saved["validation_attrs"]
        ):
            raise ValueError("Source metadata changed; choose a new run")


def check_resume_settings(previous, current):
    """Epoch limit/patience may change on resume; optimizer and objective may not."""
    for name, value in current.items():
        if (
            name not in ("epochs", "patience", "min_delta")
            and previous.get(name) != value
        ):
            raise ValueError(f"Resume changes {name}; choose a new run")
