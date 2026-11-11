"""Describe initialization, forcing, and checkpoint identity for a forecast."""

import hashlib
import json
import numpy as np
import xarray as xr
from ..data.dataloader import next_month, co2_values


def forecast_metadata(
    forecast,
    *,
    months,
    spinup_months=0,
    gaussian_start=None,
    encoded_noise_start=None,
    stochastic_start=None,
    save_spinup_latents=False,
):
    """Validate forcing coverage and record everything needed to identify a trajectory."""
    model, co2, members, seed = (
        forecast.model,
        forecast.co2,
        forecast.members,
        forecast.seed,
    )
    dates, date = [], forecast.times[0]
    for _ in range(spinup_months + months):
        date = next_month(date)
        dates.append(date)
    times = xr.DataArray(dates, dims="time", coords={"time": dates})
    all_forcing = co2_values(
        co2, times
    )  # Validate spin-up AND saved-month coverage before any draws.
    dates = dates[spinup_months:]
    forcing = all_forcing.isel(time=slice(spinup_months, None))
    initial_digest = hashlib.sha256()
    for h in forecast.history:
        initial_digest.update(h.cpu().numpy().tobytes())
    metadata = dict(
        format_version=1,
        checkpoint_sha256=model.weight_digest(),
        initialization_sha256=initial_digest.hexdigest(),
        initialization_time=str(forecast.times[0]),
        seed=seed,
        members=members,
        months=months,
        valid_times=[str(t) for t in dates],
        co2_ppm=forcing.values.tolist(),
        config=model.config.to_dict(),
        calendar=str(times.dt.calendar),
        runtime_device=str(model.device),
        sampler="DPM-Solver++ 2M midpoint; 999,674,373,120,29,6,0",
    )
    if hasattr(model, "inference_batch_size"):
        metadata["inference_batch_size"] = model.inference_batch_size
    if hasattr(model, "run_identity"):
        metadata["run_identity"] = model.run_identity
        metadata["training_settings"] = json.loads(json.dumps(model.training_settings))
    if hasattr(model, "component_parameters"):
        metadata["component_parameters"] = json.loads(
            json.dumps(model.component_parameters)
        )
        metadata["transition_parameters"] = json.loads(
            json.dumps(model.transition_parameters)
        )
    support = getattr(getattr(model, "preprocessor", None), "ocean_support_mask", None)
    if support is not None:
        metadata["ocean_support"] = dict(
            policy="training_sst",
            file="ocean_support.nc",
            sha256=hashlib.sha256(
                np.asarray(support, dtype=np.uint8).tobytes()
            ).hexdigest(),
            variables=list(model.names["ocean"]),
        )
    if gaussian_start is not None or spinup_months:
        metadata.update(
            initialization_kind="iid_standard_normal"
            if gaussian_start is not None
            else "encoded",
            initialization_coordinates="fixed_transition_normalized",
            initialization_history_order="current, previous",
            gaussian_start=gaussian_start,
            history_rng="numpy.default_rng PCG64"
            if gaussian_start is not None
            else "not sampled",
            spinup_months=spinup_months,
            spinup_valid_times=[str(t) for t in times.values[:spinup_months]],
            spinup_co2_ppm=all_forcing.values[:spinup_months].tolist(),
        )
    if encoded_noise_start is not None or stochastic_start is not None:
        history_dates = xr.DataArray(
            forecast.times, dims="time", coords={"time": forecast.times}
        )
        metadata.update(
            initialization_kind="encoded_standard_normal_fields",
            encoded_noise_start=encoded_noise_start,
            initialization_coordinates="fixed_transition_normalized",
            initialization_history_order="current, previous",
            initialization_history_times=[str(t) for t in forecast.times],
            initialization_co2_ppm=co2_values(co2, history_dates).values.tolist(),
            field_noise_coordinates="post-transform, post-centering, post-scaling encoder inputs",
            field_noise_masks="train-fitted per-variable validity, intersected with fixed ocean support",
            history_rng="numpy.default_rng PCG64; independent members, months, components, cells and variables",
            history_component_seeds=dict(atmosphere=seed + 1000, ocean=seed + 1001),
        )
    if stochastic_start is not None:
        moments_digest = hashlib.sha256()
        for names in model.names.values():
            for name in names:
                moments_digest.update(name.encode())
                for key in ("mean", "std", "count"):
                    values = (
                        model.initialization_moments[name][key]
                        .transpose("month", "lat", "lon")
                        .values
                    )
                    moments_digest.update(np.asarray(values).tobytes())
        metadata.update(
            initialization_kind="encoded_local_monthly_noise_cycle1",
            stochastic_start=stochastic_start,
            initialization_moments_sha256=moments_digest.hexdigest(),
            initialization_moment_dates=[
                str(t) for t in model.preprocessor.training_times
            ],
            cycle_months=1,
            cycle_output="Return prediction; re-encoded current state conditions subsequent months",
            cycle_policy="All fields decoded/re-encoded on first generated month only; previous slot, noise and RNG retained",
            field_noise_coordinates="training-month/gridpoint mean + population SD * iid epsilon, in normalized input space",
        )
    if save_spinup_latents:
        metadata["spinup_archive"] = dict(
            directory="spinup_latents",
            format="annual atomic torch files",
            dimensions="time, member, node, channel",
            coordinates="fixed_transition_normalized",
            node_lat=forecast.templates[0].node_lat.values.tolist(),
            node_lon=forecast.templates[0].node_lon.values.tolist(),
            description="Every monthly atmosphere/ocean latent plus end-of-block noise/RNG/history state; no decoded fields",
        )
    return metadata
