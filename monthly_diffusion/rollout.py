"""Monthly forecast iteration and initialization, with no file I/O.

Forecast accepts a loaded physical-data adapter with encode/decode methods.
ClimateModel and preprocessing live in the package. Archive writing and disk
resume use the checkout helpers in workflows/.
"""

from contextlib import contextmanager
import random
import numpy as np
import pandas as pd
import torch
import xarray as xr


def rng_state(device):
    state = dict(
        torch=torch.get_rng_state(),
        numpy=np.random.get_state(),
        python=random.getstate(),
    )
    if device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"].cpu())
    if "cuda" in state:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


@contextmanager
def fixed_rng(seed, device):
    state = rng_state(device)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        restore_rng(state)


def month_features(time):
    angle = 2 * np.pi * (time.dt.month.values - 1) / 12
    return torch.tensor(
        np.stack((np.sin(angle), np.cos(angle)), -1), dtype=torch.float32
    )


def calendar_features(time, calendar="monthly"):
    """Cyclic month, or fractional year and UTC day, for autoencoder FiLM."""
    if calendar == "monthly":
        return month_features(time)
    if calendar != "day_hour":
        raise ValueError("calendar must be monthly or day_hour")
    day_fraction = (
        time.dt.hour.values + time.dt.minute.values / 60 + time.dt.second.values / 3600
    ) / 24
    year_length = np.where(time.dt.is_leap_year.values, 366, 365)
    year_fraction = (time.dt.dayofyear.values - 1 + day_fraction) / year_length
    year_angle, day_angle = 2 * np.pi * year_fraction, 2 * np.pi * day_fraction
    return torch.tensor(
        np.stack(
            [
                np.sin(year_angle),
                np.cos(year_angle),
                np.sin(day_angle),
                np.cos(day_angle),
            ],
            -1,
        ),
        dtype=torch.float32,
    )


def validate_six_hourly(time):
    """WeatherBench timestamps are UTC 00/06/12/18, with no gaps or duplicates."""
    values = np.asarray(time.values)
    if (
        time.ndim != 1
        or not values.size
        or not np.issubdtype(values.dtype, np.datetime64)
        or np.isnat(values).any()
        or (np.diff(values) != np.timedelta64(6, "h")).any()
        or (values != values.astype("datetime64[6h]")).any()
    ):
        raise ValueError("Expected contiguous six-hourly UTC timestamps at 00/06/12/18")


def validate_months(time):
    serial = time.dt.year.values * 12 + time.dt.month.values
    if time.ndim != 1 or len(serial) == 0 or (np.diff(serial) != 1).any():
        raise ValueError("Time must contain unique, contiguous, increasing months")


def next_month(time):
    """Month-start valid time, retaining a cftime calendar when present."""
    if hasattr(time, "calendar"):
        year, month = time.year + (time.month == 12), time.month % 12 + 1
        return type(time)(year, month, 1)
    return (pd.Timestamp(time).to_period("M") + 1).to_timestamp().to_datetime64()


def co2_values(co2, time):
    """Scalar means constant; a trajectory must cover every requested month."""
    if np.isscalar(co2):
        values = np.full(len(time), float(co2))
    elif isinstance(co2, xr.DataArray) and co2.dims == ("time",):
        if co2.attrs.get("units", "ppm") not in ("ppm", "ppmv", "1e-6", "1.e-6"):
            raise ValueError(
                "CO2 must already be in ppm; no unit conversion is inferred"
            )
        # Match monthly labels, preserving the physical data's calendar.
        source = co2.time.dt.year.values * 12 + co2.time.dt.month.values
        target = time.dt.year.values * 12 + time.dt.month.values
        if len(np.unique(source)) != len(source):
            raise ValueError("CO2 contains duplicate months")
        lookup = dict(zip(source.tolist(), np.asarray(co2.values).tolist()))
        missing = [int(t) for t in target if int(t) not in lookup]
        if missing:
            raise ValueError(
                f"CO2 trajectory lacks {len(missing)} requested months; no extrapolation is performed"
            )
        values = np.array([lookup[int(t)] for t in target])
    else:
        raise TypeError("co2 must be a positive scalar or monthly xarray DataArray")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("CO2 must be finite positive ppm")
    return xr.DataArray(
        values, dims="time", coords={"time": time}, attrs={"units": "ppm"}
    )


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
        graph = model.geometry.meshes[-1]
        for component in ("atmosphere", "ocean"):
            channels = model.latent_normalizers[component].mean.numel()
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


class Forecast:
    """A monthly latent iterator; each call retains the two latest history states."""

    def __init__(
        self,
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
        guidance_scale=1.0,
    ):
        self.model = model
        self.co2 = co2
        self.members = members
        self.seed = seed
        self.condition_on_co2 = getattr(model.transition, "condition_on_co2", False)
        self.guidance_scale = float(guidance_scale)
        if not np.isfinite(self.guidance_scale) or self.guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and nonnegative")
        if self.guidance_scale != 1 and not self.condition_on_co2:
            raise ValueError("CO2 guidance requires a CO2-conditioned diffusion model")
        (self.history, self.templates, self.times, self.cycle_months, self.rng) = (
            initialize_history(
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
        )
        self.noise = (None, None)
        self.step = 0

    def __iter__(self):
        return self

    @torch.no_grad()
    def __next__(self):
        model = self.model
        time = next_month(self.times[0])
        time_array = xr.DataArray([time], dims="time", coords={"time": [time]})
        ppm = co2_values(
            self.co2, time_array
        )  # fail before consuming RNG if forcing is missing
        dates = xr.DataArray(
            self.times + [time], dims="time", coords={"time": self.times + [time]}
        )
        calendar = (
            month_features(dates)
            .reshape(1, 6)
            .expand(self.members, -1)
            .to(model.device)
        )
        ah = torch.cat((self.history[0][:, 0], self.history[0][:, 1]), -1)
        oh = torch.cat((self.history[1][:, 0], self.history[1][:, 1]), -1)
        external_rng = rng_state(model.device)
        restore_rng(self.rng)
        try:
            model.transition.eval()
            if self.condition_on_co2:
                a, o, na, no = model.transition.sample(
                    calendar,
                    ah,
                    oh,
                    previous_noise=self.noise,
                    co2=float(ppm.values[0]),
                    guidance_scale=self.guidance_scale,
                )
            else:
                a, o, na, no = model.transition.sample(
                    calendar, ah, oh, previous_noise=self.noise
                )
            self.rng = rng_state(model.device)
        finally:
            restore_rng(external_rng)
        self.noise = na, no
        self.history = [
            torch.stack((a, self.history[0][:, 0]), 1),
            torch.stack((o, self.history[1][:, 0]), 1),
        ]
        self.times = [time, self.times[0]]
        self.step += 1
        outputs = []
        for value, template in zip((a, o), self.templates):
            coords = {k: v for k, v in template.coords.items() if k != "time"}
            coords.update(time=[time], member=np.arange(self.members))
            outputs.append(
                xr.DataArray(
                    value.cpu().numpy()[None],
                    dims=("time", "member", "node", "channel"),
                    coords=coords,
                    name=template.name,
                    attrs=template.attrs,
                )
            )
        if self.step <= self.cycle_months:
            # One physical projection, exactly as tested: only replace current,
            # not previous, and keep diffusion/AR1 noise and RNG continuous.
            fields = model.decode(*outputs, co2=self.co2)
            projected = [[], []]
            for member in range(self.members):
                encoded = model.encode(
                    *(ds.isel(member=member, drop=True) for ds in fields), co2=self.co2
                )
                for i, z in enumerate(encoded):
                    if not np.isfinite(z.values).all():
                        raise FloatingPointError(
                            "Initialization cycle produced nonfinite latents"
                        )
                    projected[i].append(z.values[0])
            for i in range(2):
                self.history[i][:, 0] = torch.tensor(
                    np.stack(projected[i]), device=model.device
                )
        return tuple(outputs)

    def state_dict(self):
        state = dict(
            history=[v.cpu() for v in self.history],
            noise=[None if n is None else n.cpu() for n in self.noise],
            times=self.times,
            step=self.step,
            rng=self.rng,
        )
        if self.cycle_months:
            state["cycle_months"] = self.cycle_months
        if self.condition_on_co2:
            state["guidance_scale"] = self.guidance_scale
        return state

    def load_state_dict(self, state):
        if state.get("guidance_scale", 1.0) != self.guidance_scale:
            raise ValueError("Cannot change guidance_scale when resuming a forecast")
        self.history = [v.to(self.model.device) for v in state["history"]]
        self.noise = tuple(
            None if v is None else v.to(self.model.device) for v in state["noise"]
        )
        self.times, self.step, self.rng = state["times"], state["step"], state["rng"]
        self.cycle_months = state.get("cycle_months", 0)
