"""Training-only local monthly moments for stochastic field initialization."""

import logging
import numpy as np
import xarray as xr

log = logging.getLogger(__name__)


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
    for c, ds in (("atmosphere", atmos), ("ocean", ocean)):
        normalized = model.preprocessor.encode(ds, c)
        for name in model.names[c]:
            x = (
                normalized[name]
                .transpose("time", "lat", "lon")
                .compute()
                .astype("float64")
            )
            grouped = x.where(np.isfinite(x)).groupby("time.month")
            count = grouped.count("time")
            moments[name] = xr.Dataset(
                dict(
                    mean=grouped.mean("time", skipna=True)
                    .where(count >= 2)
                    .astype("float32"),
                    std=grouped.std("time", skipna=True, ddof=0)
                    .where(count >= 2)
                    .astype("float32"),
                    count=count.astype("int16"),
                )
            ).reindex(month=np.arange(1, 13))
            moments[name].attrs.update(
                component=c,
                coordinate_system="checkpoint normalized field",
                training_start=str(dates[0]),
                training_end=str(dates[-1]),
                training_months=len(dates),
                sd_ddof=0,
            )
            log.info("Fitted stochastic-initialization moments: %s", name)
    model.initialization_moments = moments
    return model
