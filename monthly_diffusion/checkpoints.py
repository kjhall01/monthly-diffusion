"""Write a complete checkpoint before replacing its previous version."""

from pathlib import Path
import pickle
import hashlib
import json
import re
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
import torch


class _LegacyUnpickler(pickle.Unpickler):
    """Read saved preprocessing objects from before the package was simplified."""

    def find_class(self, module, name):
        module = {
            "monthly_diffusion.data.configs": "monthly_diffusion.preprocessing",
            "workflows.settings": "monthly_diffusion.preprocessing",
            "workflows.variables": "monthly_diffusion.preprocessing",
            "workflows.data.preprocessing": "monthly_diffusion.preprocessing",
            "monthly_diffusion.data.preprocessing": "monthly_diffusion.preprocessing",
        }.get(module, module)
        return super().find_class(module, name)


def load_checkpoint(path, *, map_location="cpu", weights_only=False):
    """Load a trusted run file, including the old preprocessing module paths."""
    if weights_only:
        return torch.load(path, map_location=map_location, weights_only=True)
    reader = SimpleNamespace(
        __name__="checkpoint_pickle", Unpickler=_LegacyUnpickler, load=pickle.load
    )
    return torch.load(
        path, map_location=map_location, weights_only=False, pickle_module=reader
    )


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def current_name(old_name):
    name = old_name
    if ".context.positions." in name:
        if "encoder.context.positions.0.net." in name:
            return name.replace("encoder.context.positions.0.net.", "encoder.position.")
        return None
    if "decoder.context.grid_xyz" in name:
        return None
    name = name.replace(".context.conditions.", ".film.")
    name = name.replace(".context.pools.", ".static_pools.")
    name = name.replace(".context.grid_xyz", ".grid_xyz")
    if ".film." in name:
        name = name.replace(".static.net.", ".static.")
        name = name.replace(".time.month.", ".month.")
        name = name.replace(".time.net.", ".month_affine.")
        name = name.replace(".time.co2_shift.", ".co2_shift.")
    return re.sub(
        r"\.refiners\.(\d+)\.layers\.(\d+)\.", r".refiners.\1.\2.spatial.", name
    )


def translate_weights(weights):
    translated = {}
    for name, value in weights.items():
        new_name = current_name(name)
        if new_name is not None:
            translated[new_name] = value
    return translated


def load_autoencoder_weights(model, weights):
    model.load_state_dict(translate_weights(weights))
    if any(".context." in name for name in weights):
        model._legacy_weights = weights


def restore_autoencoder_training(model, optimizer, ema, saved):
    """Translate old parameter names/order, including an interrupted AdamW/EMA."""
    load_autoencoder_weights(model, saved["model"])
    ema.state = translate_weights(saved["ema"])
    optimizer_state = saved["optimizer"]
    if any(".context." in name for name in saved["model"]):
        optimizer_state = deepcopy(optimizer_state)
        parameters = dict(model.named_parameters(remove_duplicate=False))
        old_parameters, seen = [], set()
        # state_dict follows module registration order, just as parameters() did.
        # Keep slots for the old unused position heads, but skip shared FiLM aliases.
        for old_name in saved["model"]:
            name = current_name(old_name)
            if name in parameters and id(parameters[name]) not in seen:
                old_parameters.append(name)
                seen.add(id(parameters[name]))
            elif old_name.startswith("encoder.context.positions.") and name is None:
                old_parameters.append(None)
        groups = optimizer_state["param_groups"]
        if len(groups) != 1 or len(groups[0]["params"]) != len(old_parameters):
            raise ValueError("Unrecognized historical autoencoder optimizer layout")
        old_ids = dict(zip(old_parameters, groups[0]["params"]))
        current_ids = optimizer.state_dict()["param_groups"][0]["params"]
        state = {}
        for (name, _), current_id in zip(model.named_parameters(), current_ids):
            old_id = old_ids[name]
            if old_id in optimizer_state["state"]:
                state[current_id] = optimizer_state["state"][old_id]
        groups[0]["params"] = current_ids
        optimizer_state["state"] = state
    optimizer.load_state_dict(optimizer_state)


def checkpoint_weights(model):
    """Return live weights in the format in which this particular model was loaded."""
    current = model.state_dict()
    if hasattr(model, "_legacy_weights"):
        result = {}
        for name, old_value in model._legacy_weights.items():
            mapped = current_name(name)
            result[name] = old_value if mapped is None else current[mapped]
        return result
    if hasattr(model, "atmosphere") and hasattr(model, "ocean"):
        if hasattr(model.atmosphere, "_legacy_weights") or hasattr(
            model.ocean, "_legacy_weights"
        ):
            result = {}
            for component in ("atmosphere", "ocean"):
                for name, value in checkpoint_weights(
                    getattr(model, component)
                ).items():
                    result[f"{component}.{name}"] = value
            return result
    return current


def training_defaults():
    return dict(
        batch_size=4,
        max_epochs=1000,
        patience=5,
        min_delta=1e-4,
        learning_rate=1e-3,
        weight_decay=1e-4,
        optimizer_eps=1e-6,
        ema_decay=0.995,
        seed=20260716,
        validation_seeds=(271828, 314159),
        rollout_steps=3,
        loss_weight=0.01,
    )


def model_record(model):
    """Historical metadata shape, retained so existing archives can resume."""
    if hasattr(model, "_checkpoint_config"):
        record = dict(model._checkpoint_config)
    elif hasattr(model, "config"):
        record = model.config.to_dict()
    else:
        encoder = model.autoencoder.atmosphere.encoder
        record = dict(
            variables=[asdict(variable) for variable in model.preprocessor.variables],
            centering=model.preprocessor.centering,
            ae_width=encoder.width,
            latent_channels=encoder.latent_channels,
            mesh_levels=encoder.geometry.mesh_levels,
            refiners=encoder.refiner_counts,
            rotation=encoder.geometry.rotation,
            co2_film_levels=encoder.co2_film_levels,
            ocean_mask_aware=model.autoencoder.ocean.mask_aware,
        )
        if encoder.refiner_mlp_ratio:
            record["refiner_mlp_ratio"] = encoder.refiner_mlp_ratio
        if encoder.residual_film_only:
            record["residual_film_only"] = True
        if not encoder.condition_on_co2:
            record["condition_on_co2"] = False
        ocean_encoder = model.autoencoder.ocean.encoder
        if ocean_encoder.mask_policy != "legacy":
            record["mask_policy"] = ocean_encoder.mask_policy
        if model.preprocessor.ocean_support != "per_variable":
            record["ocean_support"] = model.preprocessor.ocean_support
        if model.preprocessor.centering == "gridpoint":
            record["gridpoint_std_floor"] = model.preprocessor.gridpoint_std_floor
        # Old archive records include these fields even for AE-only models.
        diffusion = getattr(model, "transition_parameters", {})
        record.update(
            transition_width=diffusion.get("width", 132),
            transition_depth=diffusion.get("depth", 4),
            transition_heads=diffusion.get("n_heads", 4),
            transition_ffn_ratio=diffusion.get("ffn_ratio", 1.55),
            time_embedding_channels=diffusion.get("time_embedding_channels", 64),
            atmosphere_tau=diffusion.get("atmosphere_tau", 0.5),
            ocean_tau=diffusion.get("ocean_tau", 6.0),
            coupling=diffusion.get("coupling", "jacobi"),
            cross_gradients=diffusion.get("cross_gradients", False),
        )
    for component, key in (
        ("atmosphere", "autoencoder_training"),
        ("diffusion", "diffusion_training"),
    ):
        settings = dict(training_defaults())
        for name, value in model.training_settings.get(component, {}).items():
            name = {"epochs": "max_epochs", "validation_seed": "validation_seeds"}.get(
                name, name
            )
            if name == "validation_seeds" and isinstance(value, int):
                value = (value,)
            if name in settings:
                settings[name] = value
        record[key] = settings
    return json.loads(json.dumps(record))


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_run(directory):
    path = Path(directory) / "run.json"
    if not path.exists():
        # Existing combined-checkpoint runs remain readable for generation.
        import yaml

        return dict(
            format_version=1,
            source=yaml.safe_load((Path(directory) / "data_sources.yaml").read_text()),
        )
    record = json.loads(path.read_text())
    if record.get("format_version") != 2:
        raise ValueError("Unrecognized run directory format")
    return record


def load_latents(directory, component):
    """Read a fixed training/validation cache and verify the selected AE identity."""
    directory = Path(directory) / component
    cache = load_checkpoint(
        directory / "latents.pt", map_location="cpu", weights_only=False
    )
    if cache["model_digest"] != file_digest(directory / "model.pt") or cache[
        "setup_digest"
    ] != file_digest(directory / "setup.pt"):
        raise ValueError(
            f"{component} latent cache belongs to a different AE or preprocessing"
        )
    return cache


def model_identity(directory, *, diffusion=False):
    """Identify only the models needed by this product, including preprocessing."""
    directory = Path(directory)
    if not (directory / "run.json").exists():
        return {"legacy_model": file_digest(directory / "model.pt")}
    components = ["atmosphere", "ocean"] + (["diffusion"] if diffusion else [])
    return {
        component: {
            filename: file_digest(directory / component / filename)
            for filename in ("model.pt", "setup.pt")
        }
        for component in components
    }


def autoencoder_parameters(model):
    encoder = model.encoder
    parameters = dict(
        width=encoder.width,
        latent_channels=encoder.latent_channels,
        mesh_levels=encoder.geometry.mesh_levels,
        rotation=encoder.geometry.rotation,
        refiners=encoder.refiner_counts,
        co2_film_levels=encoder.co2_film_levels,
        condition_on_co2=encoder.condition_on_co2,
        mask_aware=model.mask_aware,
        refiner_mlp_ratio=encoder.refiner_mlp_ratio,
        residual_film_only=encoder.residual_film_only,
        mask_policy=encoder.mask_policy,
    )
    if getattr(encoder, "calendar", "monthly") != "monthly":
        parameters["calendar"] = encoder.calendar
    return parameters


def diffusion_parameters(model):
    head = model.atmosphere
    block = head.blocks[0]
    parameters = dict(
        atmosphere_latent_channels=model.atmosphere_latent_channels,
        ocean_latent_channels=model.ocean_latent_channels,
        n_latent_tokens=model.n_latent_tokens,
        width=head.hidden_channels,
        depth=head.n_layers,
        n_heads=block.self_attention.num_heads,
        ffn_ratio=model.ffn_ratio,
        time_embedding_channels=head.time_embedding_channels,
        atmosphere_tau=model.atmosphere_tau,
        ocean_tau=model.ocean_tau,
        coupling=model.coupling.value,
        cross_gradients=model.cross_gradients,
    )
    # Omit the new keys for old models so existing archive/resume identities hold.
    if model.condition_on_co2:
        parameters.update(
            condition_on_co2=True,
            co2_dropout=model.co2_dropout,
            co2_mean=head.co2_mean,
            co2_std=head.co2_std,
        )
    return parameters
