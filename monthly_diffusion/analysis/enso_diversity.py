"""Small SST-box-only extension of the ENSO diversity diagnostics.

Fixed spatial weights make regional averaging commute with the joint fit of
12 monthly intercepts and a linear trend used by enso_dynamics. We check fixed
finite support explicitly before exploiting this equivalence. No inference,
lagged maps, EOFs, standardization, or ensemble averaging is performed here.
"""
import logging
import numpy as np
import pandas as pd
import xarray as xr

from .enso_dynamics import BOXES, box_ec
from .indices import regional_mean


def regional_trajectory_anomalies(raw):
    """Fit each member/box over its available analysis months, in physical °C.

    y(t) = monthly_intercept[month(t)] + trend * years_since_January_start + e(t).
    Joint OLS, not sequential detrending; no division by any standard deviation.
    Fits need complete, finite monthly regional means, but not whole edge years.
    """
    dates = pd.DatetimeIndex(raw.time.values)
    keys = dates.year*12 + dates.month
    if np.any(np.diff(keys) != 1):
        raise ValueError("Regional indices must have consecutive monthly dates")
    month = dates.month.to_numpy()-1
    t = (keys.to_numpy() - (dates[0].year*12+1))/12
    design = np.column_stack((np.eye(12)[month], t))
    if np.linalg.matrix_rank(design) != 13:
        raise ValueError("Insufficient months for monthly intercepts plus trend")
    anomalies, coefficients = {}, {}
    for name in BOXES:
        value = raw[name].transpose("time", "member").values
        if not np.isfinite(value).all():
            raise ValueError(f"Missing regional SST values: {name}")
        beta = np.linalg.lstsq(design, value, rcond=None)[0]
        anomalies[name] = (("member", "time"), (value-design@beta).T)
        coefficients[name+"_monthly_intercept"] = (("month", "member"), beta[:12])
        coefficients[name+"_trend"] = (("member",), beta[12])
    out = xr.Dataset(anomalies, coords=dict(member=raw.member, time=raw.time))
    e, c = box_ec(out.nino4, out.nino12)
    out = out.assign(E=e, C=c)
    for name in out:
        out[name].attrs["units"] = "degC"
    fit = xr.Dataset(coefficients, coords=dict(member=raw.member, month=np.arange(1, 13)))
    for name in fit:
        fit[name].attrs["units"] = "degC per year" if name.endswith("trend") else "degC"
    fit.attrs.update(time_origin=f"{dates[0].year}-01", fit_start=str(dates[0].date()),
        fit_end=str(dates[-1].date()), description=regional_trajectory_anomalies.__doc__)
    out.attrs.update(anomaly_definition="joint monthly-intercept + linear-time-trend removal, per trajectory",
        fit_start=str(dates[0].date()), fit_end=str(dates[-1].date()),
        ec_definition="E = Niño1+2 - 0.5 Niño4; C = 1.7 Niño4 - 0.1 Niño1+2; no standardization")
    return out, fit


def extract_boxes(archive, support_archive, years, support_years=(1966, 2006)):
    """Read only the tropical SST region, stream time, keep all source members.

    Support is finite in every training month/member of support_archive. The
    emulator uses its training member's mask; reference uses all its members.
    Fail rather than silently permit time-dependent spatial weighting.
    """
    if not (np.array_equal(archive.lat, support_archive.lat) and
            np.array_equal(archive.lon, support_archive.lon)):
        raise ValueError("Support and source grids differ")
    ilat = np.flatnonzero((archive.lat >= -10) & (archive.lat <= 5))
    ilon = np.flatnonzero((archive.lon >= 160) & (archive.lon <= 280))
    lat, lon = slice(ilat[0], ilat[-1]+1), slice(ilon[0], ilon[-1]+1)
    training = np.flatnonzero((support_archive.time.dt.year >= support_years[0]) &
                             (support_archive.time.dt.year <= support_years[1]))
    if len(training) != 12*(support_years[1]-support_years[0]+1):
        raise ValueError("Incomplete training interval for fixed SST support")
    support = xr.DataArray(np.ones((len(ilat), len(ilon)), dtype=bool), dims=("lat", "lon"),
        coords=dict(lat=archive.lat[lat], lon=archive.lon[lon]), name="sst_support")
    for block in support_archive.blocks(training, 12):
        support &= np.isfinite(support_archive.read("sst", block, lat=lat, lon=lon)).all(("member", "time"))
    selection = np.flatnonzero((archive.time.dt.year >= years[0]) & (archive.time.dt.year <= years[1]))
    if not len(selection):
        raise ValueError("No data in requested historical interval")
    pieces = []
    for k, block in enumerate(archive.blocks(selection, 12)):
        field = archive.read("sst", block, lat=lat, lon=lon)
        if bool((support & ~np.isfinite(field)).any()):
            raise ValueError("Nonfinite SST within fixed support; box-first detrending is not equivalent")
        field = field.where(support)
        pieces.append(xr.Dataset({name: regional_mean(field, lat_bounds=a, lon_bounds=b)
                                  for name, (a, b) in BOXES.items()}))
        if k % 20 == 0:
            logging.info("%s: SST boxes through %s", archive.paths[0].name,
                         str(field.time.values[-1])[:7])
    raw = xr.concat(pieces, "time")
    for name in raw:
        raw[name].attrs["units"] = "degC"
    raw.attrs.update(description="cosine-latitude weighted SST box means on fixed finite support",
                     support_years=str(support_years), fixed_finite_support_verified=1)
    support.attrs.update(training_years=str(support_years), scope="tropical ROI only, not a global mask")
    return raw, support


def event_summary(events, indices):
    """Pooled events, equally weighted; rates use actual member-month exposure.

    Standard deviations are population spreads (ddof=0), not uncertainty on
    the mean. Counts/rates include no minimum duration but omit censored events.
    """
    used = events.loc[events.used.astype(bool)]
    member_years = indices.sizes["member"]*indices.sizes["time"]/12
    rows = []
    for phase in ("El Niño", "La Niña"):
        part = used.loc[used.phase == phase]
        rows.append(dict(phase=phase, events=len(part), members=indices.sizes["member"],
            member_years=member_years, events_per_decade_per_member=10*len(part)/member_years,
            mean_E=part.E.mean(), std_E=part.E.std(ddof=0), mean_C=part.C.mean(), std_C=part.C.std(ddof=0),
            median_duration_months=part.duration_months.median(),
            mean_abs_peak_nino34=part.peak_nino34.abs().mean(), max_E=part.E.max(),
            E_above_1p5_degC=int((part.E > 1.5).sum()),
            ec_correlation=part[["E", "C"]].corr().iloc[0, 1] if len(part) > 2 else np.nan))
    return pd.DataFrame(rows)
