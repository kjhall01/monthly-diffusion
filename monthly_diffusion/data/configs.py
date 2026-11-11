"""Explicit scientific choices; xarray supplies coordinates and unit metadata."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
import yaml


@dataclass(frozen=True)
class Variable:
    name: str
    component: str
    units: str = ""
    transform: str = "identity"
    scale: float = 1.0
    eps: float = 1.0e-6
    weight: float = 1.0


def mdv1_variables():
    """Canonical names are model-facing; source adapters must rename explicitly."""
    atmosphere = [
        ("sfcWind", "m s^-1"),
        ("t2m", "K"),
        ("tcwv", "kg m^-2"),
        ("mtpr", "kg m^-2 s^-1"),
        ("mslp", "Pa"),
        ("olr", "W m^-2"),
        ("z500", "m^2 s^-2"),
        ("tau300_700", "m^2 s^-2"),
        ("t850", "K"),
        ("t500", "K"),
        ("u200", "m s^-1"),
        ("v200", "m s^-1"),
        ("u50", "m s^-1"),
        ("ssrd", "W m^-2"),
        ("tau_x", "N m^-2"),
        ("tau_y", "N m^-2"),
    ]
    ocean = [
        ("sst", "degrees C"),
        ("ssh", "m"),
        ("sss", "psu"),
        ("mld", "m"),
        ("sic", "dimensionless"),
        ("sivol", "m"),
        ("d20", "m"),
        ("ohc300", "J m^-2"),
        ("ohc700_minus_ohc300", "J m^-2"),
    ]
    scales = dict(
        sfcWind=5.0, tcwv=10.0, mtpr=1.0e-5, ssrd=100.0, mld=5.0, d20=50.0, sivol=0.1
    )
    result = []
    for component, fields in (("atmosphere", atmosphere), ("ocean", ocean)):
        for name, units in fields:
            weight = (
                0.5 / 14
                if component == "atmosphere" and name not in ("tau_x", "tau_y")
                else 0.05
            )
            if name in ("sic", "sivol"):
                weight /= 2
            result.append(
                Variable(
                    name,
                    component,
                    units,
                    "logit"
                    if name == "sic"
                    else "softplus_inverse"
                    if name in scales
                    else "identity",
                    scales.get(name, 1.0),
                    1.0e-4 if name == "sic" else 1.0e-6,
                    weight,
                )
            )
    return result


@dataclass
class TrainingConfig:
    batch_size: int = 4
    max_epochs: int = 1000
    patience: int = 5
    min_delta: float = 1.0e-4
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    optimizer_eps: float = 1.0e-6
    ema_decay: float = 0.995
    seed: int = 20260716
    validation_seeds: tuple[int, ...] = (271828, 314159)
    rollout_steps: int = 3
    loss_weight: float = 0.01

    def __post_init__(self):
        if min(self.batch_size, self.max_epochs, self.patience, self.rollout_steps) < 1:
            raise ValueError("Training counts must be positive")
        if not self.validation_seeds or not 0 <= self.ema_decay < 1:
            raise ValueError("Require validation seeds and 0 <= EMA decay < 1")
        if (
            min(self.learning_rate, self.optimizer_eps, self.loss_weight) <= 0
            or self.min_delta < 0
        ):
            raise ValueError(
                "Invalid learning rate, optimizer epsilon, loss scale, or min_delta"
            )


@dataclass
class ModelConfig:
    variables: list[Variable] = field(default_factory=mdv1_variables)
    centering: str = "annual_zonal"
    ae_width: int = 102
    latent_channels: int = 4
    mesh_levels: tuple[int, ...] = (4, 3, 2)
    refiners: tuple[int, ...] = (6, 6, 4)
    rotation: tuple[float, float, float] = (17.0, 23.0, 11.0)
    # Empty means every mesh level, i.e. the original MDv1 behavior.
    co2_film_levels: tuple[int, ...] = ()
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
    autoencoder_training: TrainingConfig = field(default_factory=TrainingConfig)
    diffusion_training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        self.variables = [
            Variable(**v) if isinstance(v, dict) else v for v in self.variables
        ]
        for name in ("autoencoder_training", "diffusion_training"):
            value = getattr(self, name)
            if isinstance(value, dict):
                setattr(self, name, TrainingConfig(**value))
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
        if self.ocean_support == "sst" and not any(
            v.name == "sst" and v.component == "ocean" for v in self.variables
        ):
            raise ValueError("SST-based ocean support requires ocean variable sst")

    def to_dict(self):
        # YAML-safe lists, rather than python/tuple tags.
        import json

        result = json.loads(json.dumps(asdict(self)))
        # Keep old forecast identities byte-for-byte compatible: introducing
        # an opt-in policy must not invalidate existing resumable archives.
        if self.ocean_support == "per_variable":
            result.pop("ocean_support")
        return result

    @classmethod
    def from_yaml(cls, path, **overrides):
        values = yaml.safe_load(Path(path).read_text()) or {}
        values.update(overrides)
        return cls(**values)

    def save(self, path):
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
