"""Read historical run records without bringing config machinery into the models."""

from pathlib import Path
import json
import yaml
from dataclasses import dataclass, field, asdict
from monthly_diffusion.preprocessing import Variable, mdv1_variables
from monthly_diffusion.checkpoints import (
    training_defaults,
    model_record as model_record,
)


@dataclass
class ModelSettings:
    variables: list[Variable] = field(default_factory=mdv1_variables)
    centering: str = "annual_zonal"
    gridpoint_std_floor: float = 0.05
    ae_width: int = 102
    latent_channels: int = 4
    refiner_mlp_ratio: float = 0  # historical combined checkpoints; new scripts use 2
    residual_film_only: bool = False  # missing in historical records
    mask_policy: str = "legacy"  # missing in historical records
    mesh_levels: tuple[int, ...] = (4, 3, 2)
    refiners: tuple[int, ...] = (6, 6, 4)
    rotation: tuple[float, float, float] = (17.0, 23.0, 11.0)
    # Empty means every mesh level, i.e. the original MDv1 behavior.
    co2_film_levels: tuple[int, ...] = ()
    condition_on_co2: bool = True  # absent in historical records
    transition_width: int = 132
    transition_depth: int = 4
    transition_heads: int = 4
    transition_ffn_ratio: float = 1.55
    time_embedding_channels: int = 64
    atmosphere_tau: float = 0.5
    ocean_tau: float = 6.0
    coupling: str = "jacobi"
    cross_gradients: bool = False
    ocean_mask_aware: bool = False
    # Legacy per-variable support remains the default for existing checkpoints.
    # "sst" excludes all ocean targets outside train-fitted SST availability.
    ocean_support: str = "per_variable"

    def __post_init__(self):
        self.variables = [
            Variable(**v) if isinstance(v, dict) else v for v in self.variables
        ]
        if len({v.name for v in self.variables}) != len(self.variables):
            raise ValueError("Variable names must be unique across components")
        if any(v.component not in ("atmosphere", "ocean") for v in self.variables):
            raise ValueError("Variable components must be atmosphere or ocean")
        if (
            min(
                self.ae_width,
                self.latent_channels,
                self.transition_width,
                self.transition_depth,
                self.transition_heads,
            )
            < 1
        ):
            raise ValueError("Widths, channels, depth and heads must be positive")
        if self.transition_width % self.transition_heads:
            raise ValueError("Transition width must be divisible by attention heads")
        if min(self.atmosphere_tau, self.ocean_tau) < 0:
            raise ValueError("Noise timescales cannot be negative")
        if self.ocean_support not in ("per_variable", "sst"):
            raise ValueError("ocean_support must be per_variable or sst")
        if self.mask_policy not in ("support", "legacy"):
            raise ValueError("mask_policy must be support or legacy")
        if self.ocean_support == "sst" and not any(
            v.name == "sst" and v.component == "ocean" for v in self.variables
        ):
            raise ValueError("SST-based ocean support requires ocean variable sst")

    def to_dict(self):
        # YAML-safe lists, rather than python/tuple tags.

        result = json.loads(json.dumps(asdict(self)))
        # Keep old forecast identities byte-for-byte compatible: introducing
        # an opt-in policy must not invalidate existing resumable archives.
        if self.ocean_support == "per_variable":
            result.pop("ocean_support")
        if self.centering != "gridpoint":
            result.pop("gridpoint_std_floor")
        if self.refiner_mlp_ratio == 0:
            result.pop("refiner_mlp_ratio")
        if not self.residual_film_only:
            result.pop("residual_film_only")
        if self.mask_policy == "legacy":
            result.pop("mask_policy")
        if self.condition_on_co2:
            result.pop("condition_on_co2")
        return result


def split_record(record):
    """Separate old nested training metadata from the flat model parameters."""
    values = dict(record)
    training = {
        component: dict(training_defaults(), **values.pop(key, {}))
        for component, key in (
            ("atmosphere", "autoencoder_training"),
            ("diffusion", "diffusion_training"),
        )
    }
    return ModelSettings(**values), training


def read_settings(path):
    return split_record(yaml.safe_load(Path(path).read_text()) or {})
