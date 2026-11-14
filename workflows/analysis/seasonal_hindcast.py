"""Fixed-historical-climatology, same-member DJF probabilistic verification.

Scoring functions retained from umd/analysis/djf_hindcast.py; native Gregorian
seasonal weighting added for reanalysis.

E3SM targets are continuous CMIP trajectories, not initialized E3SM forecasts;
reanalysis is a single historical trajectory. Skill includes forced change. No test outcomes
enter either set of tercile thresholds. Arrays use leading case/sample axes
and trailing latitude/longitude axes; probabilities have category axis 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def winter_dates(year: int):
    """September/October history, November--February forecasts; January label."""
    dates = pd.date_range(f"{year - 1}-09-01", periods=6, freq="MS")
    return dates[:2], dates[2:]


def reference_winters(start=1966, end=2006):
    """All complete DJFs inside the interval, with Sept/Oct history inside it."""
    return list(range(start + 1, end + 1))


def co2_increment_climatology(co2, start="1966-01", end="2006-12"):
    """Mean *adjacent monthly* ppm increments, indexed by destination month.

    BOTH endpoints must lie in the training interval. No regression, seasonal
    CO2 residual, annual smoothing, or future increments are introduced.
    """
    train = co2.sel(time=slice(start, end))
    periods = pd.DatetimeIndex(train.time.values).to_period("M")
    if not np.all(np.diff(periods.asi8) == 1):
        raise ValueError("Training CO2 is not contiguous monthly data")
    delta = np.diff(train.values.astype(np.float64))
    months = periods.month[1:]
    return (
        np.array([delta[months == m].mean() for m in range(1, 13)]),
        np.array([(months == m).sum() for m in range(1, 13)]),
    )


def forecast_co2(initial, valid_dates, increments):
    return float(initial) + np.cumsum(
        increments[pd.DatetimeIndex(valid_dates).month - 1]
    )


def djf_mean(monthly, *, axis=0, calendar="noleap", year=None):
    """Complete native-calendar DJF means, before categorization.

    E3SM uses 31/31/28 even though its archive labels are Gregorian.
    Reanalysis uses Gregorian February, including leap years.
    """
    if calendar in ("noleap", "365_day"):
        days = np.array([31., 31., 28.])
    elif calendar in ("standard", "gregorian", "proleptic_gregorian") and year is not None:
        days = pd.date_range(f"{year-1}-12-01", periods=3, freq="MS").days_in_month.to_numpy(float)
    else:
        raise ValueError(f"Unsupported calendar/year: {calendar}, {year}")
    data = np.moveaxis(np.asarray(monthly, dtype=np.float64), axis, 0)
    if data.shape[0] != 3:
        raise ValueError("DJF requires exactly three months")
    return np.einsum("t,t...->...", days/days.sum(), data)


def terciles(samples):
    """Exact 1/3,2/3 linear quantiles. Reject degenerate/tie-distorted terciles.

    Require every calibration sample finite. A tie is material if assigning
    equality to the lower bin leaves any bin more than one sample away from
    its ideal n/3 count; this tolerates finite-sample rounding, not zero masses.
    Returns thresholds, valid, degeneracy, material-tie, and reference counts.
    """
    samples = np.asarray(samples)
    finite = np.isfinite(samples).all(axis=0)
    # Replace unsupported columns solely to avoid all-NaN quantile warnings.
    safe = np.where(finite[None], samples, 0.0)
    thresholds = np.quantile(safe, [1 / 3, 2 / 3], axis=0, method="linear")
    degenerate = finite & (thresholds[0] >= thresholds[1])
    categories = categorize(safe, thresholds)
    counts = np.stack([(categories == j).sum(axis=0) for j in range(3)])
    tied = finite & (np.abs(counts - samples.shape[0] / 3.0) > 1.000001).any(axis=0)
    thresholds[:, ~finite] = np.nan
    counts[:, ~finite] = 0  # No fabricated lower-category observations on missing land.
    return thresholds, finite & ~degenerate & ~tied, degenerate, tied, counts


def categorize(values, thresholds):
    """Lower: x <= q1; middle: q1 < x <= q2; upper: x > q2; missing: -1."""
    values = np.asarray(values)
    out = (values > thresholds[0]).astype(np.int8) + (values > thresholds[1]).astype(
        np.int8
    )
    return np.where(np.isfinite(values) & np.isfinite(thresholds).all(axis=0), out, -1)


def probabilities(forecast, thresholds):
    """Forecast dimensions case, stochastic_member, lat, lon; complete ensembles."""
    cats = categorize(forecast, thresholds)
    result = np.stack([(cats == j).mean(axis=1) for j in range(3)], axis=1)
    return np.where((cats >= 0).all(axis=1)[:, None], result, np.nan)


def rps(probability, target):
    """Ordinary three-category RPS (unnormalized sum, terminal CDF term zero)."""
    observed = np.stack([target == j for j in range(3)], axis=1)
    score = np.sum(
        (np.cumsum(probability, axis=1) - np.cumsum(observed, axis=1)) ** 2, axis=1
    )
    return np.where((target >= 0) & np.isfinite(probability).all(axis=1), score, np.nan)


def aggregate_rpss(model_scores, climatology_scores):
    """Ratio of matched-case sums, NEVER mean of individual-case ratios."""
    matched = np.isfinite(model_scores) & np.isfinite(climatology_scores)
    count = matched.sum(axis=0)
    numerator = np.where(matched, model_scores, 0.0).sum(axis=0, dtype=np.float64)
    denominator = np.where(matched, climatology_scores, 0.0).sum(
        axis=0, dtype=np.float64
    )
    result = np.full(denominator.shape, np.nan)
    np.divide(numerator, denominator, out=result, where=(count > 0) & (denominator > 0))
    return 1.0 - result, count, numerator, denominator


def reliability(probability, target, area, mask, n_members=20):
    """Equal case weights, area-weighted cells within each case; 21 fixed bins.

    No uncertainty from spatially independent resampling is implied. Each
    case contributes total weight one; masks shared across categories.
    """
    bins = np.linspace(-0.5 / n_members, 1 + 0.5 / n_members, n_members + 2)
    valid = (target >= 0) & np.isfinite(probability).all(axis=1) & mask
    weights = np.where(valid, np.asarray(area)[None], 0.0)
    denominator = weights.sum(axis=(-2, -1), keepdims=True)
    weights = np.divide(
        weights, denominator, out=np.zeros_like(weights), where=denominator > 0
    )
    rows = []
    for category in range(3):
        p = probability[:, category]
        index = np.searchsorted(bins, p, side="right") - 1
        selected = index[valid]
        if selected.size and (selected.min() < 0 or selected.max() > n_members):
            raise ValueError("Forecast probability outside the fixed reliability bins")
        totals = np.bincount(selected, weights=weights[valid], minlength=n_members + 1)
        forecasts = np.bincount(
            selected, weights=weights[valid] * p[valid], minlength=n_members + 1
        )
        observations = np.bincount(
            selected,
            weights=weights[valid] * (target[valid] == category),
            minlength=n_members + 1,
        )
        counts = np.bincount(selected, minlength=n_members + 1)
        supported = np.sum(
            [
                np.bincount(i[v], minlength=n_members + 1) > 0
                for i, v in zip(index, valid)
            ],
            axis=0,
        )
        for b in range(n_members + 1):
            total = totals[b]
            rows.append(
                dict(
                    category=category,
                    bin=b,
                    lower=bins[b],
                    upper=bins[b + 1],
                    mean_probability=float(forecasts[b] / total) if total else np.nan,
                    observed_frequency=float(observations[b] / total)
                    if total
                    else np.nan,
                    case_equivalent_weight=float(total),
                    pair_count=int(counts[b]),
                    supported_cases=int(supported[b]),
                )
            )
    return rows
