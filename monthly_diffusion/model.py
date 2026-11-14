"""Physical encode/decode and forecasts around already-created neural modules.

ClimateModel does not construct architectures, read a config or train networks.
The loader supplies the two AEs, transition, and their fitted preprocessing.
"""

import hashlib
import numpy as np
import torch
import xarray as xr
from monthly_diffusion.checkpoints import checkpoint_weights
from monthly_diffusion.checkpoints import model_record
from monthly_diffusion.checkpoints import atomic_save
from monthly_diffusion.rollout import co2_values, month_features, next_month
from monthly_diffusion.preprocessing import validate_fields, validate_units
from monthly_diffusion.preprocessing import fit_initialization_moments


class ClimateModel:
    def __init__(
        self,
        atmosphere,
        ocean,
        transition,
        *,
        preprocessor,
        statics,
        atmosphere_normalizer,
        ocean_normalizer,
        batch_size=4,
    ):
        self.autoencoder = torch.nn.Module()
        self.autoencoder.atmosphere = atmosphere
        self.autoencoder.ocean = ocean
        self.transition = transition
        self.preprocessor = preprocessor
        self.statics = statics
        self.device = next(atmosphere.parameters()).device
        self.geometry = atmosphere.encoder.geometry
        self.latent_normalizers = {
            "atmosphere": atmosphere_normalizer,
            "ocean": ocean_normalizer,
        }
        for component, network in (("atmosphere", atmosphere), ("ocean", ocean)):
            if network.preprocessor is None:
                network.preprocessor = preprocessor.for_component(component)
            network.latent_normalizer = self.latent_normalizers[component]
        self.names = {
            component: [
                v.name for v in preprocessor.variables if v.component == component
            ]
            for component in ("atmosphere", "ocean")
        }
        self.inference_batch_size = batch_size
        self.training_settings = {}
        self.initialization_moments = {}
        self.stages = {}

    def _validate_pair(self, atmosphere, ocean):
        for component, ds in (("atmosphere", atmosphere), ("ocean", ocean)):
            validate_fields(ds, self.names[component], self.statics)
            validate_units(
                ds, [v for v in self.preprocessor.variables if v.component == component]
            )
        if not np.array_equal(atmosphere.time, ocean.time):
            raise ValueError(
                "Atmosphere/ocean timestamps must match exactly, including calendar"
            )

    @torch.no_grad()
    def encode(self, atmos=None, ocean=None, *, co2):
        """Physical states -> transition-normalized latents; omitted component -> None."""
        if not self.latent_normalizers:
            raise RuntimeError(
                "Train both stages or load a prepared model with fixed latent statistics"
            )
        self.autoencoder.eval()
        outputs = []
        for component, ds in (("atmosphere", atmos), ("ocean", ocean)):
            if ds is None:
                outputs.append(None)
                continue
            validate_fields(ds, self.names[component], self.statics)
            validate_units(
                ds, [v for v in self.preprocessor.variables if v.component == component]
            )
            months = month_features(ds.time).to(self.device)
            forcing = torch.tensor(
                (
                    (co2_values(co2, ds.time) - self.preprocessor.co2_mean)
                    / self.preprocessor.co2_std
                ).values,
                dtype=torch.float32,
                device=self.device,
            )[:, None]
            latents = []
            batch = getattr(
                self,
                "inference_batch_size",
                self.training_settings.get("atmosphere", {}).get("batch_size", 4),
            )
            for start in range(0, len(ds.time), batch):
                physical = self.preprocessor.encode(
                    ds.isel(time=slice(start, start + batch)), component
                ).compute()
                values = np.stack(
                    [
                        physical[v].transpose("time", "lat", "lon").values
                        for v in self.names[component]
                    ],
                    1,
                )
                x = torch.from_numpy(values).to(self.device)
                node_mask = (
                    torch.isfinite(x[:, 0])
                    if getattr(self.autoencoder, component, None).mask_aware
                    else None
                )
                z = getattr(self.autoencoder, component).encode(
                    x,
                    months[start : start + batch],
                    forcing[start : start + batch],
                    node_mask=node_mask,
                )
                latents.append(self.latent_normalizers[component].normalize(z).cpu())
            geometry = getattr(self.autoencoder, component).encoder.geometry
            graph = geometry.meshes[-1]
            outputs.append(
                xr.DataArray(
                    torch.cat(latents).numpy(),
                    dims=("time", "node", "channel"),
                    coords={
                        "time": ds.time,
                        "node": np.arange(graph.n_nodes),
                        "channel": np.arange(
                            getattr(
                                self.autoencoder, component
                            ).encoder.latent_projection.out_features
                        ),
                        "node_lat": ("node", graph.lat.numpy()),
                        "node_lon": ("node", graph.lon.numpy()),
                    },
                    name=component,
                    attrs={
                        "coordinate_system": "fixed_transition_normalized",
                        "units": "1",
                    },
                )
            )
        return tuple(outputs)

    @torch.no_grad()
    def decode(self, atmos=None, ocean=None, *, co2):
        """Transition-normalized xarray latents -> masked physical-unit fields."""
        self.autoencoder.eval()
        outputs = []
        for component, z in (("atmosphere", atmos), ("ocean", ocean)):
            if z is None:
                outputs.append(None)
                continue
            with_member = "member" in z.dims
            dims = (
                ("time", "member", "node", "channel")
                if with_member
                else ("time", "node", "channel")
            )
            values = z.transpose(*dims).values
            members = z.sizes.get("member", 1)
            values = values.reshape(-1, *values.shape[-2:])
            month = month_features(z.time).repeat_interleave(members, 0).to(self.device)
            forcing = torch.tensor(
                (
                    (co2_values(co2, z.time) - self.preprocessor.co2_mean)
                    / self.preprocessor.co2_std
                ).values,
                dtype=torch.float32,
                device=self.device,
            ).repeat_interleave(members)[:, None]
            decoded = []
            batch = getattr(
                self,
                "inference_batch_size",
                self.training_settings.get("atmosphere", {}).get("batch_size", 4),
            )
            ocean_mask = None
            if component == "ocean" and self.autoencoder.ocean.mask_aware:
                # Prefer the train-fitted SST availability mask so training and
                # latent decoding use the same ocean support.  The LSM fallback
                # keeps the atomic wrapper usable with custom ocean variables.
                support = self.preprocessor.stats.get("sst", {}).get("mask")
                if support is None:
                    support = (1.0 - self.statics.lsm).clip(0.0, 1.0)
                ocean_mask = torch.as_tensor(
                    np.asarray(support), dtype=torch.float32, device=self.device
                )
            for start in range(0, len(values), batch):
                latent = torch.as_tensor(
                    values[start : start + batch],
                    dtype=torch.float32,
                    device=self.device,
                )
                latent = self.latent_normalizers[component].denormalize(latent)
                fields = getattr(self.autoencoder, component).decode(
                    latent,
                    month[start : start + batch],
                    forcing[start : start + batch],
                    node_mask=(
                        None
                        if ocean_mask is None
                        else ocean_mask[None].expand(len(latent), -1, -1)
                    ),
                )
                decoded.append(fields.cpu().numpy())
            values = np.concatenate(decoded).reshape(
                len(z.time),
                members,
                len(self.names[component]),
                len(self.statics.lat),
                len(self.statics.lon),
            )
            coords = dict(time=z.time, lat=self.statics.lat, lon=self.statics.lon)
            if with_member:
                coords["member"] = z.member
                fields = {
                    name: (("time", "member", "lat", "lon"), values[:, :, i])
                    for i, name in enumerate(self.names[component])
                }
            else:
                fields = {
                    name: (("time", "lat", "lon"), values[:, 0, i])
                    for i, name in enumerate(self.names[component])
                }
            outputs.append(
                self.preprocessor.decode(xr.Dataset(fields, coords=coords), component)
            )
        return tuple(outputs)

    def fit_stochastic_initialization(self, atmos, ocean):
        """Fit local monthly Gaussian input moments on EXACT checkpoint training dates.

        This does not train/change any network or normalization. For older
        checkpoints, call once with physical xarray datasets (extra dates are
        ignored), then save a new export. The AE scripts save these moments during preprocessing.
        Moments live in already transformed/centered/scaled field coordinates.
        Zero SD stays zero; fewer than two finite samples means unavailable.
        Only one variable is materialized at a time, not the full training state.
        """
        return fit_initialization_moments(self, atmos, ocean)

    def initialize(
        self,
        atmos=None,
        ocean=None,
        *,
        co2,
        members=5,
        seed=20260716,
        gaussian_start=None,
        encoded_noise_start=None,
        stochastic_start=None,
        guidance_scale=1.0,
    ):
        """Return a stateful latent iterator, with observed or stochastic histories.

        Without physical ICs, ``stochastic_start='1850-01'`` draws independent
        normalized fields matched to training-month/gridpoint mean and SD,
        then encodes the preceding two months. The FIRST next() predicts
        January, decodes/re-encodes it once and feeds that projection back into
        the current history slot. It returns the January *prediction*, exactly
        as in the cycling-duration experiment. Later steps are latent-only.
        No months are skipped and terminal noise/RNG histories are not reset.
        CO2 must cover both history months and every generated month (or be a
        scalar). Training-fitted initialization_moments must be present.
        """
        from monthly_diffusion.rollout import Forecast

        return Forecast(
            self,
            atmos,
            ocean,
            co2=co2,
            members=members,
            seed=seed,
            gaussian_start=gaussian_start,
            encoded_noise_start=encoded_noise_start,
            stochastic_start=stochastic_start,
            guidance_scale=guidance_scale,
        )

    def rollout(
        self,
        atmos=None,
        ocean=None,
        *,
        co2,
        months,
        members=5,
        seed=20260716,
        path=None,
        resume=False,
        open_result=True,
        gaussian_start=None,
        spinup_months=0,
        encoded_noise_start=None,
        save_spinup_latents=False,
        stochastic_start=None,
        extend=False,
        guidance_scale=1.0,
    ):
        """Generate ``months`` saved months, optionally after discarded latent-only spin-up.

        ``gaussian_start='1845-01'`` uses independent N(0,1) history latents
        for November/December 1844, in fixed transition-normalized coordinates.
        No physical initial states are encoded in this mode.
        Alternatively, ``encoded_noise_start`` encodes independent N(0,1)
        preprocessed fields with train-fitted masks and the history months'
        calendar/CO2 conditioning. ``save_spinup_latents=True`` retains every
        spin-up latent in resumable annual blocks without decoding it.
        ``stochastic_start`` instead uses local training-month noise moments
        and one initial predict/decode/re-encode cycle, then latent-only steps.
        That first generated month counts toward spinup_months, if requested;
        otherwise it is the first saved prediction. See initialize().
        ``resume=True, extend=True`` continues a completed archive to a longer
        horizon, requiring identical checkpoint/initialization/forcing history.
        Archive writing requires the checkout's workflows/ helpers; in-memory
        forecasts work with the installed package alone.
        """
        if path is None:
            if extend or resume or save_spinup_latents:
                raise ValueError(
                    "Archive resume and saved spin-up latents require a path"
                )
            if months < 1 or spinup_months < 0:
                raise ValueError(
                    "months must be positive and spinup_months nonnegative"
                )
            forecast = self.initialize(
                atmos,
                ocean,
                co2=co2,
                members=members,
                seed=seed,
                gaussian_start=gaussian_start,
                encoded_noise_start=encoded_noise_start,
                stochastic_start=stochastic_start,
                guidance_scale=guidance_scale,
            )
            # Check the complete forcing coverage before generating any months.
            dates, date = [], forecast.times[0]
            for _ in range(spinup_months + months):
                date = next_month(date)
                dates.append(date)
            co2_values(co2, xr.DataArray(dates, dims="time", coords={"time": dates}))
            attrs = dict(
                checkpoint_sha256=self.weight_digest(),
                seed=seed,
                initialization_time=str(forecast.times[0]),
                history_order="current, previous",
            )
            if self.transition.condition_on_co2:
                attrs["guidance_scale"] = float(guidance_scale)
            for _ in range(spinup_months):
                next(forecast)
            results = []
            for _ in range(months):
                atmosphere_latent, ocean_latent = next(forecast)
                fields = xr.merge(self.decode(atmosphere_latent, ocean_latent, co2=co2))
                fields = fields.assign_coords(co2=co2_values(co2, fields.time))
                fields.attrs.update(attrs)
                results.append(fields)
            return xr.concat(results, dim="time")

        from workflows.forecast import rollout

        return rollout(
            self,
            atmos,
            ocean,
            co2=co2,
            months=months,
            members=members,
            seed=seed,
            path=path,
            resume=resume,
            open_result=open_result,
            gaussian_start=gaussian_start,
            spinup_months=spinup_months,
            encoded_noise_start=encoded_noise_start,
            save_spinup_latents=save_spinup_latents,
            stochastic_start=stochastic_start,
            extend=extend,
            guidance_scale=guidance_scale,
        )

    def weight_digest(self, *, autoencoder_only=False):
        digest = hashlib.sha256()
        modules = (
            (self.autoencoder,)
            if autoencoder_only
            else (self.autoencoder, self.transition)
        )
        for module in modules:
            if module is None:
                continue
            for name, tensor in checkpoint_weights(module).items():
                digest.update(name.encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def save(self, path):
        """Trusted local torch checkpoint: parameters plus complete scientific state."""
        payload = dict(
            format_version=1,
            config=model_record(self),
            statics=self.statics,
            preprocessing=self.preprocessor,
            normalizers=self.latent_normalizers,
            stages=self.stages,
            initialization_moments=self.initialization_moments,
            autoencoder=checkpoint_weights(self.autoencoder),
            transition=(
                None if self.transition is None else self.transition.state_dict()
            ),
            torch_version=str(torch.__version__),
            # Torch attention can choose a different execution path for frozen
            # weights. Preserve that choice across an export/reload.
            frozen_modules=[
                name
                for name in ("autoencoder", "transition")
                if getattr(self, name) is not None
                and not any(p.requires_grad for p in getattr(self, name).parameters())
            ],
        )
        if hasattr(self, "component_parameters"):
            payload["component_parameters"] = self.component_parameters
            if hasattr(self, "transition_parameters"):
                payload["transition_parameters"] = self.transition_parameters
        if hasattr(self, "run_identity"):
            payload["run_identity"] = self.run_identity
            payload["training_settings"] = self.training_settings
        if hasattr(self, "inference_batch_size"):
            payload["inference_batch_size"] = self.inference_batch_size
        atomic_save(payload, path)

    @classmethod
    def load(cls, path, *, device=None):
        from .loading import load_model

        model = load_model(path, device=device)
        if not isinstance(model, cls):
            raise TypeError("This file does not contain a complete climate model")
        return model
