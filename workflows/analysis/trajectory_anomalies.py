"""Training-period monthly-intercept + linear-trend anomalies, one trajectory at a time.

Reanalysis is not an ensemble. Fit x(t,r) = a_month(r) + b(r)*t + residual
separately for its trajectory and for each emulator member. All coefficients
use only the requested training years. This removes an estimated linear trend,
not a known forced response. NaNs remain masked; no zero-filled ocean values.
"""
from functools import lru_cache
import logging
import numpy as np
import xarray as xr
from workflows.analysis.long_rollout import save_dataset

log = logging.getLogger(__name__)


class TrainingTrajectoryAnomalies:
    def __init__(self, archive, training, output, block_months=12):
        self.archive, self.training = archive, training
        self.output, self.block_months = output, block_months
        self.origin = training[0]*12 + 1

    @lru_cache(maxsize=3)
    def coefficients(self, variable):
        path = self.output/f"{variable}.nc"
        if path.exists():
            return xr.load_dataset(path)
        archive = self.archive
        selection = np.flatnonzero((archive.time.dt.year >= self.training[0]) &
                                   (archive.time.dt.year <= self.training[1]))
        expected = np.arange(self.origin, (self.training[1]+1)*12+1)
        if not np.array_equal(archive.keys[selection], expected):
            raise ValueError("Anomaly fit requires complete training years")
        # count, sum(t), sum(y), sum(t²), sum(t*y), by calendar month.
        moments = np.zeros((5, 12, len(archive.members), len(archive.lat), len(archive.lon)))
        for block in archive.blocks(selection, self.block_months):
            field = archive.read(variable, block)
            years = (archive.keys[block]-self.origin)/12.
            months = field.time.dt.month.values
            for month in np.unique(months):
                y = field.values[:, months == month]
                t = years[months == month][None, :, None, None]
                valid = np.isfinite(y)
                x, y = np.where(valid, t, 0.), np.where(valid, y, 0.)
                moments[:, month-1] += np.stack((valid.sum(1), x.sum(1), y.sum(1),
                                                 (x*x).sum(1), (x*y).sum(1)))
        n, st, sy, stt, sty = moments
        with np.errstate(divide="ignore", invalid="ignore"):
            denominator = (stt-st*st/n).sum(0)
            slope = (sty-st*sy/n).sum(0)/denominator
            slope = np.where((n >= 2).all(0) & (denominator > 0), slope, np.nan)
            intercept = (sy-slope[None]*st)/n
        coefficients = xr.Dataset(
            dict(monthly_intercept=(("month", "member", "lat", "lon"), intercept),
                 trend_per_year=(("member", "lat", "lon"), slope)),
            coords=dict(month=range(1, 13), member=archive.members, lat=archive.lat, lon=archive.lon),
            attrs=dict(training_start=self.training[0], training_end=self.training[1],
                       definition="per-trajectory joint OLS of 12 monthly intercepts and one linear trend",
                       time_origin=f"January {self.training[0]}", variable=variable))
        save_dataset(coefficients, path)
        log.info("Fitted training-only anomaly coefficients: %s", path)
        return coefficients

    def transform(self, field):
        coefficients = self.coefficients(field.name).sel(lat=field.lat, lon=field.lon, member=field.member)
        years = (field.time.dt.year*12+field.time.dt.month-self.origin)/12.
        expected = coefficients.monthly_intercept.sel(month=field.time.dt.month) + coefficients.trend_per_year*years
        residual = (field-expected).transpose("member", "time", "lat", "lon")
        return residual.assign_attrs(field.attrs).rename(field.name)
