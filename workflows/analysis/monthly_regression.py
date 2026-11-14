"""Streaming monthly teleconnections with per-trajectory nuisance removal."""
import logging
import numpy as np
import xarray as xr

from workflows.analysis.long_rollout import RegressionMoments


def regress_on_monthly_residual(archive, index, variable, block_months=12):
    """Regress on an index already residualized against monthly means + trend.

    Each member's index must be orthogonal to 12 calendar-month intercepts and
    a linear trend on EXACTLY the analysis months. With complete spatial support,
    sum(I_residual * Y_raw) == sum(I_residual * Y_residual), so the native-unit
    slope equals explicit nuisance removal from BOTH index and field. This
    partial-regression identity avoids materializing full anomaly fields or
    estimating hundreds of thousands of unnecessary local climatologies.

    Pool member × month with equal weight. The predictor remains in degrees C,
    not standardized. Require every member/month at a retained cell; any missing
    field sample excludes that cell, rather than changing the regression sample
    and invalidating the residualization identity. Return counts for inspection.
    """
    index = index.transpose("member", "time")
    if not np.isfinite(index).all() or not np.array_equal(index.member.values, archive.members):
        raise ValueError("Index must be finite and match archive members in order")
    keys = index.time.dt.year.values*12 + index.time.dt.month.values
    if np.any(np.diff(keys) != 1):
        raise ValueError("Index months must be consecutive")
    selection = np.flatnonzero(np.isin(archive.keys, keys))
    if not np.array_equal(archive.keys[selection], keys):
        raise ValueError("Index months do not match archive coverage")
    month = index.time.dt.month.values-1
    t = (keys-keys[0])/12
    design = np.column_stack((np.eye(12)[month], t))
    projection = np.linalg.lstsq(design, index.values.T, rcond=None)[0]
    if not np.allclose(design@projection, 0., atol=1.e-8):
        raise ValueError("Index must have its per-member monthly means and linear trend removed")
    moments = RegressionMoments(1, (len(archive.lat), len(archive.lon)))
    for k, block in enumerate(archive.blocks(selection, block_months)):
        field = archive.read(variable, block)
        predictor = index.sel(time=field.time).values[None]
        moments.update(field.values, predictor)
        if k % 20 == 0:
            logging.info("%s regression through %s", variable, str(field.time.values[-1])[:7])
    counts = moments.values[0, 0]
    complete = counts == index.size
    slope = np.where(complete, moments.result()[0], np.nan)
    result = xr.Dataset(dict(slope=(("lat", "lon"), slope),
        sample_count=(("lat", "lon"), counts.astype(np.int32)),
        complete_support=(("lat", "lon"), complete)), coords=dict(lat=archive.lat, lon=archive.lon),
        attrs=dict(regression="pooled partial OLS; per-member monthly-intercept and linear-trend removal from field and index",
            implementation="residual-predictor orthogonality; same result as explicitly residualizing fields",
            start=str(index.time.values[0]), end=str(index.time.values[-1]),
            members=index.sizes["member"], total_samples=index.size,
            incomplete_cells=int((~complete).sum())))
    result.slope.attrs["units"] = archive.units[variable]+" per degC"
    return result
