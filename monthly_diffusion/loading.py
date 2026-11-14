"""Load selected networks from one standard run directory.

load_autoencoder(run, 'atmosphere') -> Autoencoder, with its fitted preprocessing
load_diffusion(run)                -> CoupledLatentDiffusion
load_models(run)                   -> physical encode/decode and forecast interface

No training, data fitting, latent generation, or checkpoint selection occurs here.
"""

from monthly_diffusion.checkpoints import load_checkpoint
from monthly_diffusion.checkpoints import load_autoencoder_weights
from copy import deepcopy
from pathlib import Path
import torch
import xarray as xr
from monthly_diffusion.checkpoints import (
    file_digest,
    load_latents,
    read_run,
    model_identity,
    atomic_save,
    autoencoder_parameters,
    diffusion_parameters,
    checkpoint_weights,
    training_defaults,
)
from monthly_diffusion.mesh import (
    make_equiangular_grid_from_xarray,
    build_mesh_hierarchy,
)
from monthly_diffusion.autoencoder import Autoencoder
from monthly_diffusion.diffusion import CoupledLatentDiffusion
from monthly_diffusion.model import ClimateModel


def default_device(device=None):
    return torch.device(
        device
        or (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
    )


def build_autoencoder(
    parameters,
    statics,
    channels,
    device,
    *,
    static_fields,
    preprocessor=None,
    latent_normalizer=None,
):
    """Rebuild saved geometry and one AE; no companion network is allocated."""
    grid = make_equiangular_grid_from_xarray(statics.lat, statics.lon, name="grid")
    geometry = build_mesh_hierarchy(
        nlat=len(statics.lat),
        nlon=len(statics.lon),
        mesh_levels=parameters["mesh_levels"],
        rotation_degrees=parameters["rotation"],
        grid=grid,
    )
    return Autoencoder(
        channels,
        geometry,
        (len(statics.lat), len(statics.lon)),
        statics=static_fields,
        preprocessor=preprocessor,
        latent_normalizer=latent_normalizer,
        width=parameters["width"],
        latent_channels=parameters["latent_channels"],
        refiners=parameters["refiners"],
        co2_film_levels=parameters["co2_film_levels"],
        condition_on_co2=parameters.get("condition_on_co2", True),
        calendar=parameters.get("calendar", "monthly"),
        mask_aware=parameters["mask_aware"],
        refiner_mlp_ratio=parameters.get("refiner_mlp_ratio", 0),
        residual_film_only=parameters.get("residual_film_only", False),
        mask_policy=parameters.get("mask_policy", "legacy"),
    ).to(device)


def load_autoencoder(
    run_directory, component=None, *, device=None, return_metadata=False
):
    """Load an AE bundle, a component directory, or load_autoencoder(run, component)."""
    path = Path(run_directory)
    if path.is_file():
        model = load_model(path, device=device)
        if not isinstance(model, Autoencoder):
            raise TypeError("This file is not an autoencoder")
        return (model, {}) if return_metadata else model
    if component is None:
        run_directory, component = path.parent, path.name
    if component not in ("atmosphere", "ocean"):
        raise ValueError("Choose atmosphere or ocean")
    directory = Path(run_directory) / component
    saved = load_checkpoint(
        directory / "model.pt", map_location="cpu", weights_only=False
    )
    if saved.get("format_version") != 2 or saved["setup_digest"] != file_digest(
        directory / "setup.pt"
    ):
        raise ValueError(
            f"{component} checkpoint format or preprocessing identity differs"
        )
    setup = load_checkpoint(
        directory / "setup.pt", map_location="cpu", weights_only=False
    )
    if setup["parameters"] != saved["parameters"]:
        raise ValueError("Selected AE constructor parameters differ from its setup")
    model = build_autoencoder(
        saved["parameters"],
        setup["statics"],
        len(setup["preprocessor"].variables),
        default_device(device),
        static_fields=setup["preprocessor"].static_tensor(setup["statics"]),
        preprocessor=setup["preprocessor"],
    )
    load_autoencoder_weights(model, saved["model"])
    model.eval().requires_grad_(False)
    if (directory / "latents.pt").exists():
        model.latent_normalizer = load_latents(run_directory, component)["normalizer"]
    elif "latent_normalizer" in saved:
        # Streaming weather runs save only channel moments, never a latent archive.
        model.latent_normalizer = saved["latent_normalizer"]
    return (model, dict(setup, **saved)) if return_metadata else model


def load_diffusion(run_directory, *, device=None, return_metadata=False):
    """Load a standalone diffusion bundle or the selected diffusion in a run."""
    if Path(run_directory).is_file():
        model = load_model(run_directory, device=device)
        if not isinstance(model, CoupledLatentDiffusion):
            raise TypeError("This file is not coupled latent diffusion")
        return (model, {}) if return_metadata else model
    directory = Path(run_directory) / "diffusion"
    saved = load_checkpoint(
        directory / "model.pt", map_location="cpu", weights_only=False
    )
    if saved.get("format_version") != 2 or saved["setup_digest"] != file_digest(
        directory / "setup.pt"
    ):
        raise ValueError("Diffusion checkpoint format or setup identity differs")
    setup = load_checkpoint(
        directory / "setup.pt", map_location="cpu", weights_only=False
    )
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
    return (model, dict(setup, **saved)) if return_metadata else model


def load_models(run_directory, *, device=None, diffusion=True, batch_size=4):
    """Load a complete run for physical inference, or just both AEs for reconstruction.

    An old run containing a combined model.pt remains supported for inference.
    New runs can use independent AE widths/channels; their latent meshes must match.
    """
    directory, device = Path(run_directory), default_device(device)
    if not (directory / "run.json").exists():
        model = load_model(directory / "model.pt", device=device)
        if "autoencoder" not in model.stages or (
            diffusion and "diffusion" not in model.stages
        ):
            raise ValueError("The requested legacy model stages are incomplete")
        if hasattr(
            model, "inference_batch_size"
        ) or batch_size != model.training_settings.get("atmosphere", {}).get(
            "batch_size", 4
        ):
            model.inference_batch_size = batch_size
        return model
    read_run(directory)
    atmosphere, atmosphere_saved = load_autoencoder(
        directory, "atmosphere", device=device, return_metadata=True
    )
    ocean, ocean_saved = load_autoencoder(
        directory, "ocean", device=device, return_metadata=True
    )
    atmosphere_cache = load_latents(directory, "atmosphere")
    ocean_cache = load_latents(directory, "ocean")
    atmosphere_preprocessor = atmosphere_saved["preprocessor"]
    ocean_preprocessor = ocean_saved["preprocessor"]
    xr.testing.assert_identical(atmosphere_saved["statics"], ocean_saved["statics"])
    xr.testing.assert_identical(
        atmosphere_preprocessor.training_co2, ocean_preprocessor.training_co2
    )
    for key in ("co2_mean", "co2_std", "topo_mean", "topo_std"):
        if getattr(atmosphere_preprocessor, key) != getattr(ocean_preprocessor, key):
            raise ValueError("Atmosphere/ocean normalization conditioning differs")
    for split in ("training_time", "validation_time"):
        xr.testing.assert_identical(atmosphere_cache[split], ocean_cache[split])
    for coordinate in ("lat", "lon"):
        atmosphere_mesh = atmosphere.encoder.geometry.meshes[-1]
        ocean_mesh = ocean.encoder.geometry.meshes[-1]
        if not torch.equal(
            getattr(atmosphere_mesh, coordinate), getattr(ocean_mesh, coordinate)
        ):
            raise ValueError(
                "Atmosphere and ocean must use identical latent mesh coordinates"
            )
    # Both AEs use the same conditioning statistics. Merge only their variable
    # transforms so encode/decode can accept both physical datasets together.
    preprocessor = deepcopy(atmosphere_preprocessor)
    preprocessor.variables += ocean_preprocessor.variables
    preprocessor.stats.update(ocean_preprocessor.stats)
    preprocessor.source_attrs.update(ocean_preprocessor.source_attrs)
    preprocessor.ocean_support = ocean_preprocessor.ocean_support
    preprocessor.ocean_support_mask = ocean_preprocessor.ocean_support_mask
    transition, diffusion_saved = None, None
    if diffusion:
        transition, diffusion_saved = load_diffusion(
            directory, device=device, return_metadata=True
        )
    model = ClimateModel(
        atmosphere,
        ocean,
        transition,
        preprocessor=preprocessor,
        statics=atmosphere_saved["statics"],
        atmosphere_normalizer=atmosphere_cache["normalizer"],
        ocean_normalizer=ocean_cache["normalizer"],
        batch_size=batch_size,
    )
    # The remaining records describe saved artifacts for exports/archive resume;
    # they do not construct or configure any networks.
    model.component_parameters = {
        "atmosphere": atmosphere_saved["parameters"],
        "ocean": ocean_saved["parameters"],
    }
    model.initialization_moments = {
        **atmosphere_saved["initialization_moments"],
        **ocean_saved["initialization_moments"],
    }
    model.stages = {
        "autoencoder": {
            c: {"epoch": info["epoch"], "score": info["score"]}
            for c, info in (("atmosphere", atmosphere_saved), ("ocean", ocean_saved))
        }
    }
    if diffusion:
        model.transition_parameters = diffusion_saved["parameters"]
        model.stages["diffusion"] = {
            "epoch": diffusion_saved["epoch"],
            "score": diffusion_saved["score"],
        }
    model.training_settings = {
        "atmosphere": atmosphere_saved["training_settings"],
        "ocean": ocean_saved["training_settings"],
    }
    if diffusion:
        model.training_settings["diffusion"] = diffusion_saved["training_settings"]
    model.run_identity = model_identity(directory, diffusion=diffusion)
    model.autoencoder.eval()
    return model


def save_model(model, path):
    """Write a portable file containing the network and its fitted scientific state."""
    if isinstance(model, Autoencoder):
        static_fields = (
            model.encoder.film[0]
            .static_fields[0]
            .T.reshape(2, *model.decoder.grid_shape)
            .cpu()
        )
        record = dict(
            format_version=3,
            kind="autoencoder",
            parameters=autoencoder_parameters(model),
            channels=model.encoder.input_projection.in_features,
            latitudes=model.encoder.geometry.grid.lat.reshape(model.decoder.grid_shape)[
                :, 0
            ]
            .cpu()
            .numpy(),
            longitudes=model.encoder.geometry.grid.lon.reshape(
                model.decoder.grid_shape
            )[0]
            .cpu()
            .numpy(),
            static_fields=static_fields,
            preprocessor=model.preprocessor,
            latent_normalizer=model.latent_normalizer,
            frozen=not any(p.requires_grad for p in model.parameters()),
            model=checkpoint_weights(model),
        )
        atomic_save(record, path)
    elif isinstance(model, CoupledLatentDiffusion):
        atomic_save(
            dict(
                format_version=3,
                kind="diffusion",
                parameters=diffusion_parameters(model),
                frozen=not any(p.requires_grad for p in model.parameters()),
                model=model.state_dict(),
            ),
            path,
        )
    elif isinstance(model, ClimateModel):
        model.save(path)
    else:
        raise TypeError(
            "Expected an Autoencoder, CoupledLatentDiffusion or ClimateModel"
        )
    return Path(path)


def load_model(path, *, device=None):
    """Load a portable AE/diffusion file or a combined climate-model checkpoint."""
    path, device = Path(path), default_device(device)
    saved = load_checkpoint(path)
    if saved.get("format_version") == 3:
        if saved["kind"] == "autoencoder":
            parameters = saved["parameters"]
            coordinates = xr.Dataset(
                coords={"lat": saved["latitudes"], "lon": saved["longitudes"]}
            )
            model = build_autoencoder(
                parameters,
                coordinates,
                saved["channels"],
                device,
                static_fields=saved["static_fields"],
                preprocessor=saved["preprocessor"],
                latent_normalizer=saved["latent_normalizer"],
            )
            load_autoencoder_weights(model, saved["model"])
        elif saved["kind"] == "diffusion":
            model = CoupledLatentDiffusion(**saved["parameters"]).to(device)
            model.load_state_dict(saved["model"])
        else:
            raise ValueError("Unrecognized model kind")
        return model.eval().requires_grad_(not saved.get("frozen", True))
    if saved.get("format_version") == 2:
        # Selected run checkpoints keep fitted data in setup.pt and latents.pt.
        if "atmosphere_latent_channels" in saved["parameters"]:
            return load_diffusion(path.parent.parent, device=device)
        return load_autoencoder(path.parent, device=device)
    if saved.get("format_version") != 1:
        raise ValueError("Unrecognized model checkpoint format")

    # Historical combined files contain flat constructor records, not live configs.
    record, preprocessor = saved["config"], saved["preprocessing"]
    component_parameters = saved.get("component_parameters")
    if component_parameters is None:
        component_parameters = {}
        for component in ("atmosphere", "ocean"):
            component_parameters[component] = dict(
                width=record["ae_width"],
                latent_channels=record["latent_channels"],
                mesh_levels=record["mesh_levels"],
                refiners=record["refiners"],
                rotation=record["rotation"],
                co2_film_levels=record.get("co2_film_levels", ()),
                condition_on_co2=record.get("condition_on_co2", True),
                refiner_mlp_ratio=record.get("refiner_mlp_ratio", 0),
                residual_film_only=record.get("residual_film_only", False),
                mask_policy=record.get("mask_policy", "legacy"),
                mask_aware=component == "ocean"
                and record.get("ocean_mask_aware", False),
            )
    components = []
    for component in ("atmosphere", "ocean"):
        prepared = preprocessor.for_component(component)
        components.append(
            build_autoencoder(
                component_parameters[component],
                saved["statics"],
                len(prepared.variables),
                device,
                static_fields=prepared.static_tensor(saved["statics"]),
                preprocessor=prepared,
                latent_normalizer=saved["normalizers"][component],
            )
        )
    transition_parameters = saved.get("transition_parameters")
    if transition_parameters is None:
        transition_parameters = dict(
            atmosphere_latent_channels=components[0].encoder.latent_channels,
            ocean_latent_channels=components[1].encoder.latent_channels,
            n_latent_tokens=components[0].encoder.geometry.meshes[-1].n_nodes,
            width=record["transition_width"],
            depth=record["transition_depth"],
            n_heads=record["transition_heads"],
            ffn_ratio=record["transition_ffn_ratio"],
            time_embedding_channels=record["time_embedding_channels"],
            atmosphere_tau=record["atmosphere_tau"],
            ocean_tau=record["ocean_tau"],
            coupling=record["coupling"],
            cross_gradients=record["cross_gradients"],
        )
    transition = None
    if saved["transition"] is not None:
        transition = CoupledLatentDiffusion(**transition_parameters).to(device)
        transition.load_state_dict(saved["transition"])
        transition.eval()
    model = ClimateModel(
        *components,
        transition,
        preprocessor=preprocessor,
        statics=saved["statics"],
        atmosphere_normalizer=saved["normalizers"]["atmosphere"],
        ocean_normalizer=saved["normalizers"]["ocean"],
        batch_size=saved.get("inference_batch_size", 4),
    )
    load_autoencoder_weights(model.autoencoder, saved["autoencoder"])
    model.autoencoder.eval()
    for name in saved.get("frozen_modules", []):
        getattr(model, name).requires_grad_(False)
    model.stages = saved["stages"]
    model.initialization_moments = saved.get("initialization_moments", {})
    # Preserve historical archive metadata exactly; these records are not live configs.
    model._checkpoint_config = record
    if "component_parameters" in saved:
        model.component_parameters = component_parameters
        if "transition_parameters" in saved:
            model.transition_parameters = transition_parameters
    if "inference_batch_size" not in saved:
        del model.inference_batch_size
    model.training_settings = saved.get(
        "training_settings",
        {
            "atmosphere": dict(
                training_defaults(), **record.get("autoencoder_training", {})
            ),
            "diffusion": dict(
                training_defaults(), **record.get("diffusion_training", {})
            ),
        },
    )
    if "run_identity" in saved:
        model.run_identity = saved["run_identity"]
    return model
