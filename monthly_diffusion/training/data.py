"""Check that resumed training uses the same physical data and forcing."""

import numpy as np
from ..data.dataloader import co2_values


def validate_training_split(model, atmos, ocean, validation, co2):
    """Reject changed training data and overlap with validation before fitting."""
    model._validate_pair(atmos, ocean)
    model._validate_pair(*validation)
    if not np.array_equal(atmos.time, model.preprocessor.training_times):
        raise ValueError("Training times differ from fitted preprocessing")
    if not np.array_equal(co2_values(co2, atmos.time), model.preprocessor.training_co2):
        raise ValueError("Training CO2 differs from the saved preprocessing fit")
    for name, ds in (("atmosphere", atmos), ("ocean", ocean)):
        if dict(ds.attrs) != model.preprocessor.source_attrs[name]:
            raise ValueError(
                "Training source metadata changed; use a new run directory"
            )
    if np.intersect1d(atmos.time.values, validation[0].time.values).size:
        raise ValueError("Training and validation times overlap")
