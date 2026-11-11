"""Stateful monthly forecasts and atomic, resumable monthly Zarr shards.

A shard contains decoded fields AND its subsequent latent/noise/RNG state. It is
renamed into place only after both are written. A crash can leave an incomplete
`.partial` shard; it is overwritten on resume, never treated as completed data.
"""
from pathlib import Path
import hashlib
import json
import logging
import numpy as np
import torch
import xarray as xr
from numcodecs import Blosc
from .data.dataloader import next_month, month_features, co2_values
from .training.common import rng_state, restore_rng, atomic_save


class Forecast:
    @torch.no_grad()
    def __init__(self, model, atmos, ocean, *, co2, members, seed, gaussian_start=None,
                 encoded_noise_start=None, stochastic_start=None):
        if members < 1:
            raise ValueError("Forecasts require at least one member")
        if sum(v is not None for v in (gaussian_start, encoded_noise_start, stochastic_start)) > 1:
            raise ValueError("Choose one noise initializer, not both/multiple modes")
        if stochastic_start is not None and not getattr(model, "initialization_moments", {}):
            raise ValueError("Stochastic initialization needs training-only local moments; call model.fit_stochastic_initialization(atmos, ocean) and save the model")
        self.model, self.co2, self.members, self.seed = model, co2, members, seed
        self.cycle_months = 1 if stochastic_start is not None else 0
        random_start = next((v for v in (stochastic_start, encoded_noise_start, gaussian_start) if v is not None), None)
        if random_start is None:
            if atmos is None or ocean is None or min(atmos.sizes["time"], ocean.sizes["time"]) < 2:
                raise ValueError("Provide both components' adjacent history states, or stochastic_start='YYYY-MM' without physical ICs")
            model._validate_pair(atmos, ocean)
            encoded = model.encode(atmos.isel(time=slice(-2, None)), ocean.isel(time=slice(-2, None)), co2=co2)
            self.templates = encoded
            self.history = [torch.tensor(z.values[::-1].copy(), device=model.device).unsqueeze(0).expand(members, -1, -1, -1).clone()
                            for z in encoded]
            self.times = list(atmos.time.values[-2:][::-1])  # [current, previous]
        else:
            if atmos is not None or ocean is not None:
                raise ValueError("Noise initialization does not accept physical initial states")
            if not model.latent_normalizers:
                raise ValueError("Gaussian initialization requires fitted transition-normalized coordinates")
            first = np.datetime64(random_start, "M")
            self.times = [(first-np.timedelta64(k, "M")).astype("datetime64[ns]") for k in (1, 2)]
            graph = model.geometry.graphs[f"mesh_l{len(model.config.mesh_levels)-1}"]
            rng = np.random.default_rng(seed)  # Separate from diffusion's Torch RNG stream.
            self.history, self.templates = [], []
            for component in ("atmosphere", "ocean"):
                shape = (members, 2, graph.n_nodes, model.config.latent_channels)
                values = rng.standard_normal(shape).astype("float32")
                self.history.append(torch.from_numpy(values).to(model.device))
                self.templates.append(xr.DataArray(np.zeros(shape[2:], dtype="float32"), dims=("node", "channel"),
                    coords=dict(node=np.arange(graph.n_nodes), channel=np.arange(model.config.latent_channels),
                                node_lat=("node", graph.lat.cpu().numpy()), node_lon=("node", graph.lon.cpu().numpy())),
                    name=component, attrs=dict(coordinate_system="fixed_transition_normalized", units="1")))
            if encoded_noise_start is not None or stochastic_start is not None:
                # FIELD noise after trained transforms/centering/scaling, not
                # noise in kelvin/metres/etc. Either unit Gaussian, or scaled
                # by training-month/local moments for stochastic_start. Encode
                # each member's independent [current, previous] pair with
                # the real calendar, prescribed CO2 and checkpoint statics.
                # Train-fitted per-variable masks require no pre-1850 data.
                model.autoencoder.eval()
                dates = xr.DataArray(self.times, dims="time", coords={"time": self.times})
                months = month_features(dates).to(model.device)
                ppm = co2_values(co2, dates)
                forcing = torch.tensor(((ppm-model.preprocessor.co2_mean)/model.preprocessor.co2_std).values,
                                       dtype=torch.float32, device=model.device)[:, None]
                statics = model.preprocessor.static_tensor(model.statics).to(model.device)[None].expand(2, -1, -1, -1)
                for i, component in enumerate(("atmosphere", "ocean")):
                    valid = np.stack([np.asarray(model.preprocessor.stats[name]["mask"].transpose("lat", "lon"), dtype=bool)
                                      for name in model.names[component]])
                    if component == "ocean":
                        support = getattr(model.preprocessor, "ocean_support_mask", None)
                        if support is not None:
                            valid &= np.asarray(support.transpose("lat", "lon"), dtype=bool)[None]
                    valid = np.broadcast_to(valid, (2, *valid.shape)).copy()
                    if stochastic_start is not None:
                        stats = [model.initialization_moments[name].sel(month=dates.dt.month.values)
                                 for name in model.names[component]]
                        mu = np.stack([s["mean"].transpose("month", "lat", "lon").values for s in stats], 1)
                        sd = np.stack([s["std"].transpose("month", "lat", "lon").values for s in stats], 1)
                        valid &= np.isfinite(mu) & np.isfinite(sd)
                        if (sd[valid] < 0).any() or not valid.any(axis=(-2,-1)).all():
                            raise ValueError("Missing or invalid monthly initialization support for a history variable")
                    net = getattr(model.autoencoder, component)
                    mask = torch.tensor(valid[:, 0], device=model.device) if net.mask_aware else None
                    rng = np.random.default_rng(seed+1000+i)
                    draws = []
                    for _ in range(members):
                        values = rng.standard_normal(valid.shape).astype("float32")
                        if stochastic_start is not None:
                            values = mu + sd * values
                        values = np.where(valid, values, np.nan).astype("float32")
                        z = net.encode(torch.from_numpy(values).to(model.device), statics, months, forcing, node_mask=mask)
                        z = model.latent_normalizers[component].normalize(z)
                        if not torch.isfinite(z).all():
                            raise FloatingPointError("Encoded noise produced nonfinite latent histories")
                        draws.append(z)
                    self.history[i] = torch.stack(draws)
        self.noise, self.step = (None, None), 0
        global_rng = rng_state(model.device)
        torch.manual_seed(seed)
        self.rng = rng_state(model.device)
        restore_rng(global_rng)

    def __iter__(self):
        return self

    @torch.no_grad()
    def __next__(self):
        model = self.model
        time = next_month(self.times[0])
        time_array = xr.DataArray([time], dims="time", coords={"time": [time]})
        co2_values(self.co2, time_array)  # fail before consuming RNG if forcing is missing
        dates = xr.DataArray(self.times+[time], dims="time", coords={"time": self.times+[time]})
        calendar = month_features(dates).reshape(1, 6).expand(self.members, -1).to(model.device)
        ah = torch.cat((self.history[0][:, 0], self.history[0][:, 1]), -1)
        oh = torch.cat((self.history[1][:, 0], self.history[1][:, 1]), -1)
        external_rng = rng_state(model.device)
        restore_rng(self.rng)
        try:
            model.transition.eval()
            a, o, na, no = model.transition.sample(calendar, ah, oh, previous_noise=self.noise)
            self.rng = rng_state(model.device)
        finally:
            restore_rng(external_rng)
        self.noise = na, no
        self.history = [torch.stack((a, self.history[0][:, 0]), 1),
                        torch.stack((o, self.history[1][:, 0]), 1)]
        self.times = [time, self.times[0]]
        self.step += 1
        outputs = []
        for value, template in zip((a, o), self.templates):
            coords = {k: v for k, v in template.coords.items() if k != "time"}
            coords.update(time=[time], member=np.arange(self.members))
            outputs.append(xr.DataArray(value.cpu().numpy()[None], dims=("time", "member", "node", "channel"),
                                        coords=coords, name=template.name, attrs=template.attrs))
        if self.step <= self.cycle_months:
            # One physical projection, exactly as tested: only replace current,
            # not previous, and keep diffusion/AR1 noise and RNG continuous.
            fields = model.decode(*outputs, co2=self.co2)
            projected = [[], []]
            for member in range(self.members):
                encoded = model.encode(*(ds.isel(member=member, drop=True) for ds in fields), co2=self.co2)
                for i, z in enumerate(encoded):
                    if not np.isfinite(z.values).all():
                        raise FloatingPointError("Initialization cycle produced nonfinite latents")
                    projected[i].append(z.values[0])
            for i in range(2):
                self.history[i][:, 0] = torch.tensor(np.stack(projected[i]), device=model.device)
        return tuple(outputs)

    def state_dict(self):
        state = dict(history=[v.cpu() for v in self.history],
                    noise=[None if n is None else n.cpu() for n in self.noise],
                    times=self.times, step=self.step, rng=self.rng)
        if self.cycle_months:
            state["cycle_months"] = self.cycle_months
        return state

    def load_state_dict(self, state):
        self.history = [v.to(self.model.device) for v in state["history"]]
        self.noise = tuple(None if v is None else v.to(self.model.device) for v in state["noise"])
        self.times, self.step, self.rng = state["times"], state["step"], state["rng"]
        self.cycle_months = state.get("cycle_months", 0)


def open_rollout(path):
    """Lazily concatenate completed compressed monthly shards, without loading them."""
    shards = sorted(Path(path).glob("month-*.zarr"))
    if not shards:
        raise FileNotFoundError(f"No completed months in {path}")
    return xr.concat([xr.open_zarr(p, consolidated=True) for p in shards], dim="time",
                     data_vars="minimal", coords="minimal", compat="equals")


def rollout(model, atmos, ocean, *, co2, months, members, seed, path, resume, open_result=True,
            gaussian_start=None, spinup_months=0, encoded_noise_start=None, save_spinup_latents=False,
            stochastic_start=None, extend=False):
    if extend and (not resume or path is None):
        raise ValueError("Extending requires an archive path and resume=True")
    if months < 1 or spinup_months < 0:
        raise ValueError("months must be positive and spinup_months nonnegative")
    if save_spinup_latents and (path is None or spinup_months == 0):
        raise ValueError("Saving spin-up latents requires an archive path and positive spinup_months")
    forecast = Forecast(model, atmos, ocean, co2=co2, members=members, seed=seed,
                        gaussian_start=gaussian_start, encoded_noise_start=encoded_noise_start,
                        stochastic_start=stochastic_start)
    initial_state = forecast.state_dict()
    dates, date = [], forecast.times[0]
    for _ in range(spinup_months+months):
        date = next_month(date)
        dates.append(date)
    times = xr.DataArray(dates, dims="time", coords={"time": dates})
    all_forcing = co2_values(co2, times)  # Validate spin-up AND saved-month coverage before any draws.
    dates = dates[spinup_months:]
    forcing = all_forcing.isel(time=slice(spinup_months,None))
    initial_digest = hashlib.sha256()
    for h in forecast.history:
        initial_digest.update(h.cpu().numpy().tobytes())
    metadata = dict(format_version=1, checkpoint_sha256=model.weight_digest(),
        initialization_sha256=initial_digest.hexdigest(), initialization_time=str(forecast.times[0]),
        seed=seed, members=members, months=months, valid_times=[str(t) for t in dates],
        co2_ppm=forcing.values.tolist(), config=model.config.to_dict(),
        calendar=str(times.dt.calendar), runtime_device=str(model.device),
        sampler="DPM-Solver++ 2M midpoint; 999,674,373,120,29,6,0")
    support = getattr(getattr(model, "preprocessor", None), "ocean_support_mask", None)
    if support is not None:
        metadata["ocean_support"] = dict(policy="training_sst", file="ocean_support.nc",
            sha256=hashlib.sha256(np.asarray(support, dtype=np.uint8).tobytes()).hexdigest(),
            variables=list(model.names["ocean"]))
    if gaussian_start is not None or spinup_months:
        metadata.update(initialization_kind="iid_standard_normal" if gaussian_start is not None else "encoded",
                        initialization_coordinates="fixed_transition_normalized",
                        initialization_history_order="current, previous", gaussian_start=gaussian_start,
                        history_rng="numpy.default_rng PCG64" if gaussian_start is not None else "not sampled",
                        spinup_months=spinup_months,
                        spinup_valid_times=[str(t) for t in times.values[:spinup_months]],
                        spinup_co2_ppm=all_forcing.values[:spinup_months].tolist())
    if encoded_noise_start is not None or stochastic_start is not None:
        history_dates = xr.DataArray(forecast.times, dims="time", coords={"time": forecast.times})
        metadata.update(initialization_kind="encoded_standard_normal_fields", encoded_noise_start=encoded_noise_start,
                        initialization_coordinates="fixed_transition_normalized",
                        initialization_history_order="current, previous",
                        initialization_history_times=[str(t) for t in forecast.times],
                        initialization_co2_ppm=co2_values(co2, history_dates).values.tolist(),
                        field_noise_coordinates="post-transform, post-centering, post-scaling encoder inputs",
                        field_noise_masks="train-fitted per-variable validity, intersected with fixed ocean support",
                        history_rng="numpy.default_rng PCG64; independent members, months, components, cells and variables",
                        history_component_seeds=dict(atmosphere=seed+1000, ocean=seed+1001))
    if stochastic_start is not None:
        moments_digest = hashlib.sha256()
        for names in model.names.values():
            for name in names:
                moments_digest.update(name.encode())
                for key in ("mean", "std", "count"):
                    values = model.initialization_moments[name][key].transpose("month", "lat", "lon").values
                    moments_digest.update(np.asarray(values).tobytes())
        metadata.update(initialization_kind="encoded_local_monthly_noise_cycle1", stochastic_start=stochastic_start,
            initialization_moments_sha256=moments_digest.hexdigest(),
            initialization_moment_dates=[str(t) for t in model.preprocessor.training_times],
            cycle_months=1, cycle_output="Return prediction; re-encoded current state conditions subsequent months",
            cycle_policy="All fields decoded/re-encoded on first generated month only; previous slot, noise and RNG retained",
            field_noise_coordinates="training-month/gridpoint mean + population SD * iid epsilon, in normalized input space")
    if save_spinup_latents:
        metadata["spinup_archive"] = dict(directory="spinup_latents", format="annual atomic torch files",
            dimensions="time, member, node, channel", coordinates="fixed_transition_normalized",
            node_lat=forecast.templates[0].node_lat.values.tolist(),
            node_lon=forecast.templates[0].node_lon.values.tolist(),
            description="Every monthly atmosphere/ocean latent plus end-of-block noise/RNG/history state; no decoded fields")
    if path is not None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        record = path/"forecast.json"
        if record.exists():
            if not resume:
                raise FileExistsError("Forecast exists; use resume=True")
            previous = json.loads(record.read_text())
            if extend and previous["initialization_sha256"] != metadata["initialization_sha256"]:
                mutable = {"months", "valid_times", "co2_ppm", "initialization_sha256"}
                if {k: v for k, v in previous.items() if k not in mutable} != {k: v for k, v in metadata.items() if k not in mutable}:
                    raise ValueError("Forecast identity/configuration/forcing differs from archive")
                # Extension means continuation of the ARCHIVED trajectory, not
                # a fresh initialization. MPS re-encoding need not be bitwise
                # reproducible. Check the initial current slot against the
                # previous slot retained in the first forecast, then discard
                # the re-encoded histories; load_state_dict below restores the
                # actual complete history/noise/RNG state at the saved endpoint.
                if spinup_months:
                    raise ValueError("Spin-up initialization differs from archive")
                first = torch.load(path/"month-000001.zarr"/"forecast_state.pt", weights_only=False, map_location="cpu")
                if np.datetime64(first["times"][1]) != np.datetime64(forecast.times[0]):
                    raise ValueError("Initialization date differs from archived first history")
                differences = []
                for fresh, saved in zip(forecast.history, first["history"]):
                    fresh, saved = fresh[:, 0].cpu(), saved[:, 1]
                    torch.testing.assert_close(fresh, saved, rtol=1e-5, atol=1e-5)
                    differences.append(float((fresh-saved).abs().max()))
                logging.getLogger(__name__).info("Using archived initialization identity; re-encoding max absolute differences (atmosphere, ocean): %s; fresh histories discarded", differences)
                metadata["initialization_sha256"] = previous["initialization_sha256"]
            extending = previous != metadata
            if extending:
                # Only the horizon may change. In particular, every prescribed
                # forcing value already used must match, not just its filename.
                prefix = dict(metadata, months=previous["months"],
                              valid_times=metadata["valid_times"][:previous["months"]],
                              co2_ppm=metadata["co2_ppm"][:previous["months"]])
                if not extend or months <= previous["months"] or prefix != previous:
                    different = [k for k in previous.keys() | prefix.keys() if previous.get(k) != prefix.get(k)]
                    raise ValueError(f"Forecast identity/configuration/forcing differs from archive: {different}")
            shards = sorted(path.glob("month-*.zarr"))
            if shards:
                forecast.load_state_dict(torch.load(shards[-1]/"forecast_state.pt", weights_only=False, map_location="cpu"))
                if [p.name for p in shards] != [f"month-{i:06d}.zarr" for i in range(1, forecast.step-spinup_months+1)]:
                    raise ValueError("Forecast archive has missing monthly shards")
            if extending:
                if len(shards) != previous["months"]:
                    raise ValueError("Finish the original forecast horizon before extending it")
                if np.datetime64(forecast.times[0]) != np.datetime64(previous["valid_times"][-1]):
                    raise ValueError("Restart state date differs from the last archived forecast month")
                # Keep the original identity, then commit the new horizon
                # atomically. Completed fields and restart states are untouched.
                backup = path/f"forecast_through_{previous['months']:06d}.json"
                if backup.exists() and backup.read_bytes() != record.read_bytes():
                    raise ValueError("Conflicting original-horizon provenance")
                backup.write_bytes(record.read_bytes())
                pending = path/"forecast.partial.json"
                pending.write_text(json.dumps(metadata, indent=2))
                pending.replace(record)
                logging.getLogger(__name__).info("Extending saved forecast from %s: %d -> %d months; restoring latent/noise/RNG state",
                    previous["valid_times"][-1], previous["months"], months)
        else:
            if support is not None:
                # Keep the actual training-fitted mask beside the forecast so
                # streaming diagnostics can apply it to reference data too,
                # without instantiating the model or inferring masks anew.
                support.astype("int8").to_dataset(name="ocean_support").to_netcdf(path/"ocean_support.partial.nc")
                (path/"ocean_support.partial.nc").replace(path/"ocean_support.nc")
            record.write_text(json.dumps(metadata, indent=2))
        if support is not None:
            archived_support = xr.load_dataset(path/"ocean_support.nc").ocean_support
            xr.testing.assert_equal(archived_support.astype(bool), support.astype(bool).rename("ocean_support"))
        if (gaussian_start is not None or encoded_noise_start is not None or stochastic_start is not None or spinup_months) and not (path/"initial_state.pt").exists():
            # Keep the initial random histories for audit, not decoded spin-up fields.
            torch.save(initial_state, path/"initial_state.pt")
        if save_spinup_latents:
            spinup_path = path/"spinup_latents"
            spinup_path.mkdir(exist_ok=True)
            blocks = sorted(spinup_path.glob("year-*.pt"))
            if [p.name for p in blocks] != [f"year-{i:04d}.pt" for i in range(1, len(blocks)+1)]:
                raise ValueError("Spin-up archive has missing annual blocks")
            if len(blocks) > (spinup_months+11)//12:
                raise ValueError("Spin-up archive exceeds requested duration")
            if forecast.step > spinup_months and len(blocks) != (spinup_months+11)//12:
                raise ValueError("Decoded archive is missing retained spin-up latents")
            if forecast.step == 0 and blocks:
                state = torch.load(blocks[-1], weights_only=False, map_location="cpu")["state"]
                if state["step"] != min(12*len(blocks), spinup_months):
                    raise ValueError("Spin-up block and restart step disagree")
                forecast.load_state_dict(state)
        elif forecast.step == 0 and (path/"spinup_state.pt").exists():
            forecast.load_state_dict(torch.load(path/"spinup_state.pt", weights_only=False, map_location="cpu"))
    spinup_times, spinup_a, spinup_o = [], [], []
    while forecast.step < spinup_months:
        a, o = next(forecast)  # Latent-only, except the optional first-month initialization cycle.
        if save_spinup_latents:
            spinup_times.append(a.time.values[0])
            spinup_a.append(a.values[0])
            spinup_o.append(o.values[0])
        if forecast.step % 12 == 0 or forecast.step == spinup_months:
            if save_spinup_latents:
                atomic_save(dict(time=np.asarray(spinup_times),
                    latents=dict(atmosphere=np.stack(spinup_a), ocean=np.stack(spinup_o)),
                    state=forecast.state_dict()), spinup_path/f"year-{(forecast.step+11)//12:04d}.pt")
                spinup_times, spinup_a, spinup_o = [], [], []
            elif path is not None:
                atomic_save(forecast.state_dict(), path/"spinup_state.pt")
            logging.getLogger(__name__).info("Latent-only spin-up %d/%d months", forecast.step, spinup_months)
    accumulated = []
    for _ in range(forecast.step, spinup_months+months):
        a, o = next(forecast)
        fields = xr.merge(model.decode(a, o, co2=co2), compat="no_conflicts")
        fields = fields.assign_coords(co2=co2_values(co2, fields.time))
        fields.attrs.update(checkpoint_sha256=metadata["checkpoint_sha256"], seed=seed,
                            initialization_time=metadata["initialization_time"], history_order="current, previous")
        if path is None:
            accumulated.append(fields)
        else:
            saved_step = forecast.step-spinup_months
            final = path/f"month-{saved_step:06d}.zarr"
            partial = path/f"month-{saved_step:06d}.partial"
            codec = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
            encoding = {name: dict(dtype="float32", compressor=codec,
                chunks=(1, 1, min(180, fields.sizes["lat"]), min(360, fields.sizes["lon"]))) for name in fields.data_vars}
            fields.to_zarr(partial, mode="w", consolidated=True, encoding=encoding)
            torch.save(forecast.state_dict(), partial/"forecast_state.pt")
            partial.rename(final)
            if saved_step % 12 == 0 or saved_step == months:
                logging.getLogger(__name__).info("Saved forecast month %d/%d (%s)", saved_step, months, str(fields.time.values[0])[:10])
    if path is None:
        return xr.concat(accumulated, dim="time")
    # CLI workflows read bounded blocks later; avoid constructing a thousands-
    # of-shards Xarray/Dask object just to throw it away at process exit.
    return open_rollout(path) if open_result else path
