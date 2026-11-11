"""Load selected networks from one standard run directory.

load_autoencoder(run, 'atmosphere') -> (torch module, saved metadata)
load_diffusion(run)                -> (torch module, saved metadata)
load_models(run)                   -> physical encode/decode and forecast interface

No training, data fitting, latent generation, or checkpoint selection occurs here.
"""

from copy import deepcopy
from pathlib import Path
import torch
import xarray as xr
from .run_files import file_digest, load_latents, read_run, model_identity
from .modules.mesh import (
    make_equiangular_grid_from_xarray,
    build_mesh_hierarchy_context,
)
from .modules.autoencoder import MeshHierarchyAutoencoder
from .components.coupled_latent_diffusion import CoupledLatentDiffusion
from .data.configs import ModelConfig


def default_device(device=None):
    return torch.device(
        device
        or (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )


def build_autoencoder(parameters, statics, channels, device):
    """Rebuild saved geometry and one AE; no companion network is allocated."""
    grid = make_equiangular_grid_from_xarray(statics.lat, statics.lon, name="grid")
    geometry = build_mesh_hierarchy_context(
        nlat=len(statics.lat),
        nlon=len(statics.lon),
        mesh_levels=parameters["mesh_levels"],
        rotation_degrees=parameters["rotation"],
        grid=grid,
    )
    return MeshHierarchyAutoencoder(
        channels, geometry, (len(statics.lat), len(statics.lon)), **parameters
    ).to(device)


def load_autoencoder(run_directory, component, *, device=None):
    if component not in ("atmosphere", "ocean"):
        raise ValueError("Choose atmosphere or ocean")
    directory = Path(run_directory) / component
    saved = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
    if saved.get("format_version") != 2 or saved["setup_digest"] != file_digest(
        directory / "setup.pt"
    ):
        raise ValueError(
            f"{component} checkpoint format or preprocessing identity differs"
        )
    setup = torch.load(directory / "setup.pt", map_location="cpu", weights_only=False)
    if setup["parameters"] != saved["parameters"]:
        raise ValueError("Selected AE constructor parameters differ from its setup")
    model = build_autoencoder(
        saved["parameters"],
        setup["statics"],
        len(setup["preprocessor"].variables),
        default_device(device),
    )
    model.load_state_dict(saved["model"])
    model.eval().requires_grad_(False)
    return model, dict(setup, **saved)


def load_diffusion(run_directory, *, device=None):
    directory = Path(run_directory) / "diffusion"
    saved = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
    if saved.get("format_version") != 2 or saved["setup_digest"] != file_digest(
        directory / "setup.pt"
    ):
        raise ValueError("Diffusion checkpoint format or setup identity differs")
    setup = torch.load(directory / "setup.pt", map_location="cpu", weights_only=False)
    for component, digests in setup["ae_digests"].items():
        for filename, digest in digests.items():
            if file_digest(Path(run_directory) / component / filename) != digest:
                raise ValueError(
                    f"Diffusion was trained with different {component} {filename}"
                )
    if setup["parameters"] != saved["parameters"]:
        raise ValueError(
            "Selected diffusion constructor parameters differ from its setup"
        )
    model = CoupledLatentDiffusion(**saved["parameters"]).to(default_device(device))
    model.load_state_dict(saved["model"])
    model.eval().requires_grad_(False)
    return model, dict(setup, **saved)


def load_models(run_directory, *, device=None, diffusion=True, batch_size=4):
    """Load a complete run for physical inference, or just both AEs for reconstruction.

    An old run containing a combined model.pt remains supported for inference.
    New runs can use independent AE widths/channels; their latent meshes must match.
    """
    from .model import MonthlyDiffusionV1

    directory, device = Path(run_directory), default_device(device)
    if not (directory / "run.json").exists():
        model = MonthlyDiffusionV1.load(directory / "model.pt", device=device)
        if "autoencoder" not in model.stages or (
            diffusion and "diffusion" not in model.stages
        ):
            raise ValueError("The requested legacy model stages are incomplete")
        if (
            hasattr(model, "inference_batch_size")
            or batch_size != model.config.autoencoder_training.batch_size
        ):
            model.inference_batch_size = batch_size
        return model
    read_run(directory)
    atmosphere, a = load_autoencoder(directory, "atmosphere", device=device)
    ocean, o = load_autoencoder(directory, "ocean", device=device)
    ac, oc = load_latents(directory, "atmosphere"), load_latents(directory, "ocean")
    ap, op = a["preprocessor"], o["preprocessor"]
    xr.testing.assert_identical(a["statics"], o["statics"])
    xr.testing.assert_identical(ap.training_co2, op.training_co2)
    for key in ("co2_mean", "co2_std", "topo_mean", "topo_std"):
        if getattr(ap, key) != getattr(op, key):
            raise ValueError("Atmosphere/ocean normalization conditioning differs")
    for split in ("training_time", "validation_time"):
        xr.testing.assert_identical(ac[split], oc[split])
    for coordinate in ("lat", "lon"):
        ag = atmosphere.encoder.context.geometry
        og = ocean.encoder.context.geometry
        at = ag.graphs[ag.metadata["mesh_level_names"][-1]]
        ot = og.graphs[og.metadata["mesh_level_names"][-1]]
        if not torch.equal(getattr(at, coordinate), getattr(ot, coordinate)):
            raise ValueError(
                "Atmosphere and ocean must use identical latent mesh coordinates"
            )
    p = a["parameters"]
    # ModelConfig is used only to describe older physical-inference metadata.
    # Network construction uses the individual saved parameters, not this record.
    config = ModelConfig(
        variables=ap.variables + op.variables,
        centering=ap.centering,
        ae_width=p["width"],
        latent_channels=p["latent_channels"],
        mesh_levels=p["mesh_levels"],
        refiners=p["refiners"],
        rotation=p["rotation"],
        co2_film_levels=p["co2_film_levels"],
        ocean_mask_aware=o["parameters"]["mask_aware"],
        ocean_support=op.ocean_support,
    )
    model = MonthlyDiffusionV1(statics=a["statics"], config=config, device=device)
    model.autoencoder = torch.nn.Module()
    model.autoencoder.add_module("atmosphere", atmosphere)
    model.autoencoder.add_module("ocean", ocean)
    model.geometry = atmosphere.encoder.context.geometry
    model.component_parameters = {
        "atmosphere": a["parameters"],
        "ocean": o["parameters"],
    }
    model.names = {
        c: [v.name for v in info["preprocessor"].variables]
        for c, info in (("atmosphere", a), ("ocean", o))
    }
    model.preprocessor = deepcopy(ap)
    model.preprocessor.variables += op.variables
    model.preprocessor.stats.update(op.stats)
    model.preprocessor.source_attrs.update(op.source_attrs)
    model.preprocessor.ocean_support, model.preprocessor.ocean_support_mask = (
        op.ocean_support,
        op.ocean_support_mask,
    )
    model.latent_normalizers = {
        "atmosphere": ac["normalizer"],
        "ocean": oc["normalizer"],
    }
    model.initialization_moments = {
        **a["initialization_moments"],
        **o["initialization_moments"],
    }
    model.stages = {
        "autoencoder": {
            c: {"epoch": info["epoch"], "score": info["score"]}
            for c, info in (("atmosphere", a), ("ocean", o))
        }
    }
    if diffusion:
        model.transition, saved = load_diffusion(directory, device=device)
        model.transition_parameters = saved["parameters"]
        model.stages["diffusion"] = {"epoch": saved["epoch"], "score": saved["score"]}
        for key, value in saved["parameters"].items():
            name = {
                "width": "transition_width",
                "depth": "transition_depth",
                "n_heads": "transition_heads",
                "ffn_ratio": "transition_ffn_ratio",
            }.get(key, key)
            if hasattr(config, name):
                setattr(config, name, value)
    model.training_settings = {
        "atmosphere": a["training_settings"],
        "ocean": o["training_settings"],
    }
    if diffusion:
        model.training_settings["diffusion"] = saved["training_settings"]
    # Keep the legacy metadata fields truthful, and retain both AE records above.
    for stage, settings in model.training_settings.items():
        if stage == "ocean":
            continue
        target = (
            config.diffusion_training
            if stage == "diffusion"
            else config.autoencoder_training
        )
        for name, value in settings.items():
            name = {"epochs": "max_epochs", "validation_seed": "validation_seeds"}.get(
                name, name
            )
            if name == "validation_seeds" and isinstance(value, int):
                value = (value,)
            if hasattr(target, name):
                setattr(target, name, value)
    model.inference_batch_size = batch_size
    model.run_identity = model_identity(directory, diffusion=diffusion)
    model.autoencoder.eval()
    return model
