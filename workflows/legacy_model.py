"""Model state and the xarray encode/decode API.

Training lives in the scripts. This physical-data interface holds loaded neural
modules as .autoencoder and .transition for inference and scientific experiments.
"""

from monthly_diffusion.checkpoints import load_checkpoint
from monthly_diffusion.checkpoints import load_autoencoder_weights
import logging
import torch

from monthly_diffusion.mesh import (
    make_equiangular_grid_from_xarray,
    build_mesh_hierarchy,
)
from monthly_diffusion.autoencoder import CoupledAutoencoder
from monthly_diffusion.diffusion import CoupledLatentDiffusion
from workflows.legacy_records import ModelSettings
from workflows.legacy_records import split_record
from monthly_diffusion.preprocessing import Preprocessor
from workflows.data.dataloader import (
    co2_values,
)

log = logging.getLogger(__name__)


from monthly_diffusion.model import ClimateModel


class MonthlyDiffusionV1(ClimateModel):
    """Compatibility for old paired scripts and combined checkpoint files."""

    def __init__(self, *, statics=None, settings=None, device=None):
        self.config = settings if settings is not None else ModelSettings()
        self.training_settings = {}
        self.device = torch.device(
            device
            or (
                "mps"
                if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available() else "cpu"
            )
        )
        self.statics = statics
        self.preprocessor = Preprocessor(
            self.config.variables,
            self.config.centering,
            self.config.ocean_support,
            gridpoint_std_floor=self.config.gridpoint_std_floor,
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
        self.geometry = build_mesh_hierarchy(
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
        self.autoencoder = CoupledAutoencoder(
            len(self.names["atmosphere"]),
            len(self.names["ocean"]),
            self.geometry,
            (len(self.statics.lat), len(self.statics.lon)),
            statics=self.preprocessor.static_tensor(self.statics),
            width=c.ae_width,
            latent_channels=c.latent_channels,
            refiners=tuple(c.refiners),
            co2_film_levels=tuple(c.co2_film_levels),
            condition_on_co2=c.condition_on_co2,
            ocean_mask_aware=c.ocean_mask_aware,
            refiner_mlp_ratio=c.refiner_mlp_ratio,
            residual_film_only=c.residual_film_only,
            mask_policy=c.mask_policy,
        ).to(self.device)
        self.transition = CoupledLatentDiffusion(
            atmosphere_latent_channels=c.latent_channels,
            ocean_latent_channels=c.latent_channels,
            n_latent_tokens=self.geometry.meshes[-1].n_nodes,
            width=c.transition_width,
            depth=c.transition_depth,
            n_heads=c.transition_heads,
            ffn_ratio=c.transition_ffn_ratio,
            time_embedding_channels=c.time_embedding_channels,
            atmosphere_tau=c.atmosphere_tau,
            ocean_tau=c.ocean_tau,
            coupling=c.coupling,
            cross_gradients=c.cross_gradients,
        ).to(self.device)

    def prepare(self, atmosphere, ocean, *, co2, statics=None, seed=20260716):
        """Fit physical statistics on these TRAINING states only, then build modules."""
        if statics is not None:
            self.statics = statics
        if self.autoencoder is not None:
            raise RuntimeError(
                "Already prepared; load a checkpoint or construct a fresh wrapper"
            )
        torch.manual_seed(seed)
        if self.statics is None:
            raise ValueError(
                "Supply static phi_s and lsm fields before preparing the model"
            )
        forcing = co2_values(co2, atmosphere.time)
        self.preprocessor.fit(atmosphere, ocean, self.statics, forcing)
        self._build()
        self._validate_pair(atmosphere, ocean)
        return self

    @classmethod
    def load(cls, path, *, device=None):
        saved = load_checkpoint(path, map_location="cpu", weights_only=False)
        if saved.get("format_version") != 1:
            raise ValueError(
                "Not a monthly_diffusion standalone checkpoint; legacy importing is not supported"
            )
        settings, training = split_record(saved["config"])
        model = cls(statics=saved["statics"], settings=settings, device=device)
        model.preprocessor = saved["preprocessing"]
        model.training_settings = training
        if "component_parameters" in saved:
            from monthly_diffusion.loading import build_autoencoder

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
                        static_fields=model.preprocessor.static_tensor(model.statics),
                    ),
                )
            model.geometry = model.autoencoder.atmosphere.encoder.geometry
            model.transition = CoupledLatentDiffusion(**model.transition_parameters).to(
                model.device
            )
        else:
            model._build()
        load_autoencoder_weights(model.autoencoder, saved["autoencoder"])
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
        for name in saved.get("frozen_modules", []):
            getattr(model, name).requires_grad_(False)
        return model
