"""Create two history months from observations, latent noise, or field noise."""

import numpy as np
import torch
import xarray as xr
from ..data.dataloader import month_features, co2_values
from ..random_state import rng_state, restore_rng


@torch.no_grad()
def initialize_history(
    model,
    atmos,
    ocean,
    *,
    co2,
    members,
    seed,
    gaussian_start=None,
    encoded_noise_start=None,
    stochastic_start=None,
):
    """Return newest-first histories, xarray templates, dates, cycling limit, and RNG."""
    if members < 1:
        raise ValueError("Forecasts require at least one member")
    if (
        sum(
            v is not None
            for v in (gaussian_start, encoded_noise_start, stochastic_start)
        )
        > 1
    ):
        raise ValueError("Choose one noise initializer, not both/multiple modes")
    if stochastic_start is not None and not getattr(
        model, "initialization_moments", {}
    ):
        raise ValueError(
            "Stochastic initialization needs training-only local moments; call model.fit_stochastic_initialization(atmos, ocean) and save the model"
        )
    cycle_months = 1 if stochastic_start is not None else 0
    random_start = next(
        (
            v
            for v in (stochastic_start, encoded_noise_start, gaussian_start)
            if v is not None
        ),
        None,
    )
    if random_start is None:
        if (
            atmos is None
            or ocean is None
            or min(atmos.sizes["time"], ocean.sizes["time"]) < 2
        ):
            raise ValueError(
                "Provide both components' adjacent history states, or stochastic_start='YYYY-MM' without physical ICs"
            )
        model._validate_pair(atmos, ocean)
        encoded = model.encode(
            atmos.isel(time=slice(-2, None)), ocean.isel(time=slice(-2, None)), co2=co2
        )
        templates = encoded
        history = [
            torch.tensor(z.values[::-1].copy(), device=model.device)
            .unsqueeze(0)
            .expand(members, -1, -1, -1)
            .clone()
            for z in encoded
        ]
        times = list(atmos.time.values[-2:][::-1])  # [current, previous]
    else:
        if atmos is not None or ocean is not None:
            raise ValueError(
                "Noise initialization does not accept physical initial states"
            )
        if not model.latent_normalizers:
            raise ValueError(
                "Gaussian initialization requires fitted transition-normalized coordinates"
            )
        first = np.datetime64(random_start, "M")
        times = [
            (first - np.timedelta64(k, "M")).astype("datetime64[ns]") for k in (1, 2)
        ]
        rng = np.random.default_rng(seed)  # Separate from diffusion's Torch RNG stream.
        history, templates = [], []
        graph = model.geometry.graphs[f"mesh_l{len(model.config.mesh_levels) - 1}"]
        for component in ("atmosphere", "ocean"):
            channels = (
                model.component_parameters[component]["latent_channels"]
                if hasattr(model, "component_parameters")
                else model.config.latent_channels
            )
            shape = (members, 2, graph.n_nodes, channels)
            values = rng.standard_normal(shape).astype("float32")
            history.append(torch.from_numpy(values).to(model.device))
            templates.append(
                xr.DataArray(
                    np.zeros(shape[2:], dtype="float32"),
                    dims=("node", "channel"),
                    coords=dict(
                        node=np.arange(graph.n_nodes),
                        channel=np.arange(channels),
                        node_lat=("node", graph.lat.cpu().numpy()),
                        node_lon=("node", graph.lon.cpu().numpy()),
                    ),
                    name=component,
                    attrs=dict(
                        coordinate_system="fixed_transition_normalized", units="1"
                    ),
                )
            )
        if encoded_noise_start is not None or stochastic_start is not None:
            # FIELD noise after trained transforms/centering/scaling, not
            # noise in kelvin/metres/etc. Either unit Gaussian, or scaled
            # by training-month/local moments for stochastic_start. Encode
            # each member's independent [current, previous] pair with
            # the real calendar, prescribed CO2 and checkpoint statics.
            # Train-fitted per-variable masks require no pre-1850 data.
            model.autoencoder.eval()
            dates = xr.DataArray(times, dims="time", coords={"time": times})
            months = month_features(dates).to(model.device)
            ppm = co2_values(co2, dates)
            forcing = torch.tensor(
                (
                    (ppm - model.preprocessor.co2_mean) / model.preprocessor.co2_std
                ).values,
                dtype=torch.float32,
                device=model.device,
            )[:, None]
            statics = (
                model.preprocessor.static_tensor(model.statics)
                .to(model.device)[None]
                .expand(2, -1, -1, -1)
            )
            for i, component in enumerate(("atmosphere", "ocean")):
                valid = np.stack(
                    [
                        np.asarray(
                            model.preprocessor.stats[name]["mask"].transpose(
                                "lat", "lon"
                            ),
                            dtype=bool,
                        )
                        for name in model.names[component]
                    ]
                )
                if component == "ocean":
                    support = getattr(model.preprocessor, "ocean_support_mask", None)
                    if support is not None:
                        valid &= np.asarray(
                            support.transpose("lat", "lon"), dtype=bool
                        )[None]
                valid = np.broadcast_to(valid, (2, *valid.shape)).copy()
                if stochastic_start is not None:
                    stats = [
                        model.initialization_moments[name].sel(
                            month=dates.dt.month.values
                        )
                        for name in model.names[component]
                    ]
                    mu = np.stack(
                        [
                            s["mean"].transpose("month", "lat", "lon").values
                            for s in stats
                        ],
                        1,
                    )
                    sd = np.stack(
                        [
                            s["std"].transpose("month", "lat", "lon").values
                            for s in stats
                        ],
                        1,
                    )
                    valid &= np.isfinite(mu) & np.isfinite(sd)
                    if (sd[valid] < 0).any() or not valid.any(axis=(-2, -1)).all():
                        raise ValueError(
                            "Missing or invalid monthly initialization support for a history variable"
                        )
                net = getattr(model.autoencoder, component)
                mask = (
                    torch.tensor(valid[:, 0], device=model.device)
                    if net.mask_aware
                    else None
                )
                rng = np.random.default_rng(seed + 1000 + i)
                draws = []
                for _ in range(members):
                    values = rng.standard_normal(valid.shape).astype("float32")
                    if stochastic_start is not None:
                        values = mu + sd * values
                    values = np.where(valid, values, np.nan).astype("float32")
                    z = net.encode(
                        torch.from_numpy(values).to(model.device),
                        statics,
                        months,
                        forcing,
                        node_mask=mask,
                    )
                    z = model.latent_normalizers[component].normalize(z)
                    if not torch.isfinite(z).all():
                        raise FloatingPointError(
                            "Encoded noise produced nonfinite latent histories"
                        )
                    draws.append(z)
                history[i] = torch.stack(draws)
    global_rng = rng_state(model.device)
    torch.manual_seed(seed)
    rng = rng_state(model.device)
    restore_rng(global_rng)
    return history, templates, times, cycle_months, rng
