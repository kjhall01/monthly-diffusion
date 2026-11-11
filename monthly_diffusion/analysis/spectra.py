"""Inherited Welch PSD and normalized lag autocovariance definitions."""
from collections.abc import Iterable
import numpy as np
import xarray as xr
from scipy import signal
_VALID_DETRENDS = {"none", "constant", "linear"}

def lag_autocorrelation(
    x: xr.DataArray | xr.Dataset,
    *,
    lags: Iterable[int] | None = None,
    max_lag: int | None = None,
    time_dim: str = "time",
    lag_dim: str = "lag_months",
    detrend: str = "linear",
) -> xr.DataArray | xr.Dataset:
    """Return the normalized autocovariance at non-negative integer lags.

    Missing samples are linearly interpolated along time before detrending.
    The lagged covariance uses the full-record variance as its normalization,
    so lag zero is exactly one for every non-constant finite series.
    """

    if isinstance(x, xr.Dataset):
        return xr.Dataset(
            {
                name: lag_autocorrelation(
                    value,
                    lags=lags,
                    max_lag=max_lag,
                    time_dim=time_dim,
                    lag_dim=lag_dim,
                    detrend=detrend,
                )
                for name, value in x.data_vars.items()
            }
        )
    lag_values = _resolve_lags(
        int(x.sizes[time_dim]),
        lags=lags,
        max_lag=max_lag,
    )
    detrend = _validate_detrend(detrend)
    result = xr.apply_ufunc(
        _lag_autocorrelation_1d,
        x,
        input_core_dims=[[time_dim]],
        output_core_dims=[[lag_dim]],
        vectorize=True,
        kwargs={"lags": lag_values, "detrend": detrend},
        dask="parallelized",
        dask_gufunc_kwargs={
            "allow_rechunk": True,
            "output_sizes": {lag_dim: lag_values.size},
        },
        output_dtypes=[np.float64],
    ).assign_coords({lag_dim: lag_values})
    result.attrs.update(
        {
            "statistic": "normalized autocovariance",
            "detrend": detrend,
            "lag_units": "samples",
            "missing_value_policy": "linear interpolation along time",
        }
    )
    result[lag_dim].attrs["units"] = "samples"
    return result


def _resolve_lags(
    n_time: int,
    *,
    lags: Iterable[int] | None,
    max_lag: int | None,
) -> np.ndarray:
    if n_time < 2:
        raise ValueError("Lag autocorrelation requires at least two samples.")
    if lags is not None and max_lag is not None:
        raise ValueError("Specify either lags or max_lag, not both.")
    if lags is None:
        maximum = min(n_time - 1, 120 if max_lag is None else int(max_lag))
        if maximum < 0:
            raise ValueError("max_lag must be non-negative.")
        values = np.arange(maximum + 1, dtype=np.int64)
    else:
        values = np.asarray(tuple(int(value) for value in lags), dtype=np.int64)
    if values.size == 0:
        raise ValueError("At least one lag is required.")
    if np.any(values < 0) or np.any(values >= n_time):
        raise ValueError(f"Lags must be between 0 and {n_time - 1}.")
    if np.unique(values).size != values.size:
        raise ValueError("Lag values must be unique.")
    return values


def _validate_detrend(detrend: str) -> str:
    detrend = str(detrend).lower()
    if detrend not in _VALID_DETRENDS:
        raise ValueError(
            f"detrend must be one of {sorted(_VALID_DETRENDS)}; got {detrend!r}."
        )
    return detrend


def _filled_detrended_1d(values: np.ndarray, detrend: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return np.full_like(values, np.nan)
    if not np.all(finite):
        sample = np.arange(values.size)
        values = np.interp(sample, sample[finite], values[finite])
    if detrend == "none":
        return values
    return signal.detrend(values, type=detrend)


def _lag_autocorrelation_1d(
    values: np.ndarray,
    *,
    lags: np.ndarray,
    detrend: str,
) -> np.ndarray:
    values = _filled_detrended_1d(values, detrend)
    if not np.all(np.isfinite(values)):
        return np.full(lags.size, np.nan, dtype=np.float64)
    values = values - values.mean()
    denominator = float(np.dot(values, values))
    if denominator <= 0.0:
        return np.full(lags.size, np.nan, dtype=np.float64)
    return np.asarray(
        [
            1.0
            if lag == 0
            else float(np.dot(values[:-lag], values[lag:]) / denominator)
            for lag in lags
        ],
        dtype=np.float64,
    )


def temporal_spectrum(
    x: xr.DataArray | xr.Dataset,
    *,
    time_dim: str = "time",
    dt: float = 1.0,
    frequency_dim: str = "frequency",
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> xr.DataArray | xr.Dataset:
    """Welch power spectral density along time.

    The default uses 256 samples per Hann-windowed segment (or the complete
    record when it is shorter), with 50% overlap.  In this monthly workflow
    ``dt=1`` means frequency is measured in cycles per month.
    """

    if isinstance(x, xr.Dataset):
        return xr.Dataset({
            name: temporal_spectrum(
                da,
                time_dim=time_dim,
                dt=dt,
                frequency_dim=frequency_dim,
                nperseg=nperseg,
                noverlap=noverlap,
            )
            for name, da in x.data_vars.items()
        })

    n = int(x.sizes[time_dim])
    if n < 2:
        raise ValueError("Welch PSD requires at least two samples.")
    nperseg = min(n, 256) if nperseg is None else int(nperseg)
    if not 2 <= nperseg <= n:
        raise ValueError("nperseg must be between 2 and the length of the time dimension.")
    noverlap = nperseg // 2 if noverlap is None else int(noverlap)
    if not 0 <= noverlap < nperseg:
        raise ValueError("noverlap must be non-negative and smaller than nperseg.")

    frequency = signal.welch(
        np.zeros(n, dtype=np.float64),
        fs=1.0 / dt,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )[0]
    out = xr.apply_ufunc(
        _welch_psd_1d,
        x,
        input_core_dims=[[time_dim]],
        output_core_dims=[[frequency_dim]],
        vectorize=True,
        kwargs={"nperseg": nperseg, "noverlap": noverlap, "dt": dt},
        dask="allowed",
        output_dtypes=[np.float64],
    )
    out = out.assign_coords({frequency_dim: frequency})
    out.attrs.update({
        "spectral_estimator": "Welch PSD",
        "window": "Hann",
        "nperseg": nperseg,
        "noverlap": noverlap,
        "detrend": "constant",
        "scaling": "density",
        "sampling_interval": dt,
        "frequency_units": "cycles per sample",
    })
    out[frequency_dim].attrs["units"] = "cycles per sample"
    return out


def _welch_psd_1d(
    values: np.ndarray,
    *,
    nperseg: int,
    noverlap: int,
    dt: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    output_size = nperseg // 2 + 1
    if finite.sum() < 2:
        return np.full(output_size, np.nan, dtype=np.float64)
    if not np.all(finite):
        sample = np.arange(values.size)
        values = np.interp(sample, sample[finite], values[finite])
    return signal.welch(
        values,
        fs=1.0 / dt,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )[1]
