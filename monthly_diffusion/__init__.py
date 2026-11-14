"""Spherical autoencoders, coupled diffusion and monthly rollout.

Models own their fitted preprocessing and latent normalizers. Save/load utilities
are included here; editable training loops and analysis stay in the repository.
"""

from .autoencoder import (
    Encoder,
    Decoder,
    Autoencoder,
    CoupledAutoencoder,
)
from .diffusion import (
    SphericalTransformer,
    SphericalDiT,
    LatentDiffusion,
    CoupledLatentDiffusion,
)
from .rollout import Forecast
from .preprocessing import Preprocessor, LatentNormalizer, Variable, select_variables
from .model import ClimateModel
from .loading import load_model, save_model, load_autoencoder, load_diffusion, load_models

__version__ = "1.0.0.dev0"
