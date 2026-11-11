"""Convenience API for in-memory or streamed forecasts.

For an editable, step-by-step workflow, see scripts/generate_ar_rollout.py.
"""

import logging
import xarray as xr
from .metadata import forecast_metadata
from ..data.dataloader import co2_values
from .forecast import Forecast
from .archive import prepare_archive, write_month, open_rollout
from .spinup import spin_up


def rollout(
    model,
    atmos,
    ocean,
    *,
    co2,
    months,
    members,
    seed,
    path,
    resume,
    open_result=True,
    gaussian_start=None,
    spinup_months=0,
    encoded_noise_start=None,
    save_spinup_latents=False,
    stochastic_start=None,
    extend=False,
):
    if extend and (not resume or path is None):
        raise ValueError("Extending requires an archive path and resume=True")
    if months < 1 or spinup_months < 0:
        raise ValueError("months must be positive and spinup_months nonnegative")
    if save_spinup_latents and (path is None or spinup_months == 0):
        raise ValueError(
            "Saving spin-up latents requires an archive path and positive spinup_months"
        )
    forecast = Forecast(
        model,
        atmos,
        ocean,
        co2=co2,
        members=members,
        seed=seed,
        gaussian_start=gaussian_start,
        encoded_noise_start=encoded_noise_start,
        stochastic_start=stochastic_start,
    )
    metadata = forecast_metadata(
        forecast,
        months=months,
        spinup_months=spinup_months,
        gaussian_start=gaussian_start,
        encoded_noise_start=encoded_noise_start,
        stochastic_start=stochastic_start,
        save_spinup_latents=save_spinup_latents,
    )
    if path is not None:
        save_initial_state = (
            any(
                start is not None
                for start in (gaussian_start, encoded_noise_start, stochastic_start)
            )
            or spinup_months > 0
        )
        path = prepare_archive(
            forecast,
            path,
            metadata,
            resume=resume,
            extend=extend,
            spinup_months=spinup_months,
            save_spinup_latents=save_spinup_latents,
            save_initial_state=save_initial_state,
        )
    spin_up(forecast, spinup_months, path=path, save_latents=save_spinup_latents)
    accumulated = []
    for _ in range(forecast.step, spinup_months + months):
        a, o = next(forecast)
        fields = xr.merge(model.decode(a, o, co2=co2), compat="no_conflicts")
        fields = fields.assign_coords(co2=co2_values(co2, fields.time))
        fields.attrs.update(
            checkpoint_sha256=metadata["checkpoint_sha256"],
            seed=seed,
            initialization_time=metadata["initialization_time"],
            history_order="current, previous",
        )
        if path is None:
            accumulated.append(fields)
        else:
            saved_step = forecast.step - spinup_months
            write_month(fields, forecast, path, spinup_months=spinup_months)
            if saved_step % 12 == 0 or saved_step == months:
                logging.getLogger(__name__).info(
                    "Saved forecast month %d/%d (%s)",
                    saved_step,
                    months,
                    str(fields.time.values[0])[:10],
                )
    if path is None:
        return xr.concat(accumulated, dim="time")
    return open_rollout(path) if open_result else path
