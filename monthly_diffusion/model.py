"""Model state and the xarray encode/decode API.

Training lives in the scripts. This physical-data interface holds loaded neural
modules as .autoencoder and .transition for inference and scientific experiments.
"""

from pathlib import Path
import hashlib
import logging
import numpy as np
import torch
import xarray as xr

from .modules.mesh import (
    make_equiangular_grid_from_xarray,
    build_mesh_hierarchy_context,
)
from .modules.autoencoder import AutoencoderConfig, CoupledAutoencoder
from .components.coupled_latent_diffusion import (
    CoupledLatentDiffusion,
    CoupledDiffusionConfig,
)
from .data.configs import ModelConfig
from .data.preprocessing import Preprocessor
from .data.dataloader import (
    co2_values,
    month_features,
    validate_fields,
    validate_units,
)
from .checkpoints import atomic_save
from .inference.moments import fit_initialization_moments

log = logging.getLogger(__name__)


class MonthlyDiffusionV1:
    def __init__(self, *, statics=None, config=None, device=None, **kwargs):
        if config is None:
            self.config = ModelConfig(**kwargs)
        elif isinstance(config, (str, Path)):
            self.config = ModelConfig.from_yaml(config, **kwargs)
        else:
            values = (
                config.to_dict() if isinstance(config, ModelConfig) else dict(config)
            )
            values.update(kwargs)
            self.config = ModelConfig(**values)
        self.device = torch.device(
            device
            or (
                "mps"
                if torch.backends.mps.is_available()
                else "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )
        self.statics = statics
        self.preprocessor = Preprocessor(
            self.config.variables, self.config.centering, self.config.ocean_support
        )
        self.latent_normalizers = {}
        self.initialization_moments = {}
        self.stages = {}
        self.autoencoder = self.transition = None

    def _build(self):
        if self.statics is None:
            raise ValueError(
                "Supply static phi_s and lsm fields once, at construction or prepare()"
            )
        self.statics = self.statics[["phi_s", "lsm"]].transpose("lat", "lon").load()
        grid = make_equiangular_grid_from_xarray(
            self.statics.lat, self.statics.lon, name="grid"
        )
        c = self.config
        self.geometry = build_mesh_hierarchy_context(
            nlat=len(self.statics.lat),
            nlon=len(self.statics.lon),
            mesh_levels=c.mesh_levels,
            rotation_degrees=c.rotation,
            grid=grid,
        )
        self.names = {
            component: [v.name for v in c.variables if v.component == component]
            for component in ("atmosphere", "ocean")
        }
        if not all(self.names.values()):
            raise ValueError(
                "MonthlyDiffusionV1 requires both components; use atomic modules for a single component"
            )
        ae_config = AutoencoderConfig(
            c.ae_width,
            c.latent_channels,
            tuple(c.mesh_levels),
            tuple(c.refiners),
            tuple(c.rotation),
            tuple(c.co2_film_levels),
        )
        self.autoencoder = CoupledAutoencoder(
            len(self.names["atmosphere"]),
            len(self.names["ocean"]),
            self.geometry,
            (len(self.statics.lat), len(self.statics.lon)),
            ae_config,
            ocean_mask_aware=c.ocean_mask_aware,
        ).to(self.device)
        self.transition = CoupledLatentDiffusion(
            CoupledDiffusionConfig(
                atmosphere_latent_channels=c.latent_channels,
                ocean_latent_channels=c.latent_channels,
                n_latent_tokens=self.geometry.graphs[
                    f"mesh_l{len(c.mesh_levels) - 1}"
                ].n_nodes,
                width=c.transition_width,
                depth=c.transition_depth,
                n_heads=c.transition_heads,
                ffn_ratio=c.transition_ffn_ratio,
                time_embedding_channels=c.time_embedding_channels,
                atmosphere_tau=c.atmosphere_tau,
                ocean_tau=c.ocean_tau,
                coupling=c.coupling,
                cross_gradients=c.cross_gradients,
            )
        ).to(self.device)

    def prepare(self, atmosphere, ocean, *, co2, statics=None):
        """Fit physical statistics on these TRAINING states only, then build modules."""
        if statics is not None:
            self.statics = statics
        if self.autoencoder is not None:
            raise RuntimeError(
                "Already prepared; load a checkpoint or construct a fresh wrapper"
            )
        torch.manual_seed(self.config.autoencoder_training.seed)
        self._build()
        self._validate_pair(atmosphere, ocean)
        forcing = co2_values(co2, atmosphere.time)
        self.preprocessor.fit(atmosphere, ocean, self.statics, forcing)
        return self

    def _validate_pair(self, atmosphere, ocean):
        for component, ds in (("atmosphere", atmosphere), ("ocean", ocean)):
            validate_fields(ds, self.names[component], self.statics)
            validate_units(
                ds, [v for v in self.config.variables if v.component == component]
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
                ds, [v for v in self.config.variables if v.component == component]
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
            static = self.preprocessor.static_tensor(self.statics).to(self.device)[None]
            latents = []
            batch = getattr(
                self,
                "inference_batch_size",
                self.config.autoencoder_training.batch_size,
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
                    static.expand(len(x), -1, -1, -1),
                    months[start : start + batch],
                    forcing[start : start + batch],
                    node_mask=node_mask,
                )
                latents.append(self.latent_normalizers[component].normalize(z).cpu())
            geometry = getattr(self.autoencoder, component).encoder.context.geometry
            graph = geometry.graphs[geometry.metadata["mesh_level_names"][-1]]
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
            static = self.preprocessor.static_tensor(self.statics).to(self.device)[None]
            decoded = []
            batch = getattr(
                self,
                "inference_batch_size",
                self.config.autoencoder_training.batch_size,
            )
            ocean_mask = None
            if component == "ocean" and self.config.ocean_mask_aware:
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
                    static.expand(len(latent), -1, -1, -1),
                    month[start : start + batch],
                    forcing[start : start + batch],
                    node_mask=None
                    if ocean_mask is None
                    else ocean_mask[None].expand(len(latent), -1, -1),
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
        from .inference.forecast import Forecast

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
        """
        from .inference.rollout import rollout

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
        )

    def weight_digest(self, *, autoencoder_only=False):
        digest = hashlib.sha256()
        modules = (
            (self.autoencoder,)
            if autoencoder_only
            else (self.autoencoder, self.transition)
        )
        for module in modules:
            for name, tensor in module.state_dict().items():
                digest.update(name.encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def save(self, path):
        """Trusted local torch checkpoint: parameters plus complete scientific state."""
        payload = dict(
            format_version=1,
            config=self.config.to_dict(),
            statics=self.statics,
            preprocessing=self.preprocessor,
            normalizers=self.latent_normalizers,
            stages=self.stages,
            initialization_moments=self.initialization_moments,
            autoencoder=self.autoencoder.state_dict(),
            transition=self.transition.state_dict(),
            torch_version=str(torch.__version__),
        )
        if hasattr(self, "component_parameters"):
            payload["component_parameters"] = self.component_parameters
            payload["transition_parameters"] = self.transition_parameters
        if hasattr(self, "run_identity"):
            payload["run_identity"] = self.run_identity
            payload["training_settings"] = self.training_settings
        if hasattr(self, "inference_batch_size"):
            payload["inference_batch_size"] = self.inference_batch_size
        atomic_save(payload, path)

    @classmethod
    def load(cls, path, *, device=None):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("format_version") != 1:
            raise ValueError(
                "Not a monthly_diffusion standalone checkpoint; legacy importing is not supported"
            )
        model = cls(statics=saved["statics"], config=saved["config"], device=device)
        if "component_parameters" in saved:
            from .model_loading import build_autoencoder

            model.names = {
                c: [v.name for v in model.config.variables if v.component == c]
                for c in ("atmosphere", "ocean")
            }
            model.autoencoder = torch.nn.Module()
            model.component_parameters = saved["component_parameters"]
            model.transition_parameters = saved["transition_parameters"]
            for component, parameters in model.component_parameters.items():
                model.autoencoder.add_module(
                    component,
                    build_autoencoder(
                        parameters,
                        model.statics,
                        len(model.names[component]),
                        model.device,
                    ),
                )
            model.geometry = model.autoencoder.atmosphere.encoder.context.geometry
            model.transition = CoupledLatentDiffusion(**model.transition_parameters).to(
                model.device
            )
        else:
            model._build()
        model.autoencoder.load_state_dict(saved["autoencoder"])
        model.transition.load_state_dict(saved["transition"])
        model.preprocessor, model.latent_normalizers = (
            saved["preprocessing"],
            saved["normalizers"],
        )
        model.stages = saved["stages"]
        if "inference_batch_size" in saved:
            model.inference_batch_size = saved["inference_batch_size"]
        if "run_identity" in saved:
            model.run_identity = saved["run_identity"]
            model.training_settings = saved["training_settings"]
        model.initialization_moments = saved.get("initialization_moments", {})
        model.autoencoder.eval()
        model.transition.eval()
        return model
