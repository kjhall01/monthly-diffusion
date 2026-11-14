"""File operations for the training scripts. No training loops live here.

Dictionaries below describe files on disk. They never configure a neural module.
The scripts construct models and optimizers directly, then hand them here to save.
"""

from dataclasses import dataclass, field
from pathlib import Path
from monthly_diffusion.autoencoder import Autoencoder
from monthly_diffusion.rollout import rng_state, restore_rng
from monthly_diffusion import checkpoints
from monthly_diffusion.checkpoints import (
    load_checkpoint,
    autoencoder_parameters,
    diffusion_parameters,
)
from monthly_diffusion.checkpoints import (
    load_autoencoder_weights,
    restore_autoencoder_training,
)
from workflows.run_files import file_digest, save_history, save_selection
from workflows.data.dataloader import co2_values
from monthly_diffusion.preprocessing import component_moments
from .checks import check_component_data, check_resume_settings


@dataclass
class TrainingProgress:
    """The counters and measurements that must survive an interrupted run."""

    epoch: int = 0
    optimizer_steps: int = 0
    best_validation: float = float("inf")
    patience_best: float = float("inf")
    stale_epochs: int = 0
    history: list = field(default_factory=list)


def load_preprocessor(directory):
    return load_checkpoint(Path(directory) / "setup.pt")["preprocessor"]


def save_autoencoder_setup(
    directory, model, preprocessor, statics, training, validation, co2
):
    """Save fitted preprocessing, or check that an existing run still describes it."""
    directory = Path(directory)
    path = directory / "setup.pt"
    component = preprocessor.variables[0].component
    parameters = autoencoder_parameters(model)
    if path.exists():
        saved = load_checkpoint(path)
        check_component_data(
            training, validation, statics, preprocessor.variables, co2, saved
        )
        old_parameters = dict(saved["parameters"])
        old_parameters.setdefault("refiner_mlp_ratio", 0)
        old_parameters.setdefault("residual_film_only", False)
        old_parameters.setdefault("condition_on_co2", True)
        old_parameters.setdefault("mask_policy", "legacy")
        if old_parameters != parameters:
            raise ValueError("AE architecture changed; choose a new run directory")
        if (
            saved["preprocessor"].centering != preprocessor.centering
            or saved["preprocessor"].ocean_support != preprocessor.ocean_support
            or getattr(saved["preprocessor"], "mask_constant_cells", False)
            != getattr(preprocessor, "mask_constant_cells", False)
            or (
                preprocessor.centering == "gridpoint"
                and saved["preprocessor"].gridpoint_std_floor
                != preprocessor.gridpoint_std_floor
            )
        ):
            raise ValueError("Preprocessing changed; choose a new run directory")
        return
    check_component_data(training, validation, statics, preprocessor.variables, co2)
    checkpoints.atomic_save(
        dict(
            parameters=parameters,
            preprocessor=preprocessor,
            statics=statics,
            validation_times=validation.time.values,
            validation_co2=co2_values(co2, validation.time),
            validation_attrs=dict(validation.attrs),
            initialization_moments=component_moments(preprocessor, training, component),
        ),
        path,
    )


def save_diffusion_setup(directory, model, *, training_co2=None, validation_co2=None):
    directory = Path(directory)
    record = dict(
        parameters=diffusion_parameters(model),
        ae_digests={
            component: {
                name: file_digest(directory.parent / component / name)
                for name in ("model.pt", "setup.pt", "latents.pt")
            }
            for component in ("atmosphere", "ocean")
        },
    )
    path = directory / "setup.pt"
    if model.condition_on_co2:
        if training_co2 is None or validation_co2 is None:
            raise ValueError("Save the exact training/validation CO2 used by diffusion")
        record["co2_ppm"] = dict(
            training=training_co2.tolist(), validation=validation_co2.tolist()
        )
    if path.exists():
        if load_checkpoint(path) != record:
            raise ValueError(
                "Diffusion architecture or AE inputs changed; choose a new run"
            )
    else:
        checkpoints.atomic_save(record, path)


def load_training_progress(
    directory,
    model,
    optimizer,
    ema,
    *,
    resume,
    batch_size,
    epochs,
    patience,
    min_delta,
    seed,
    validation_seeds,
    rollout_steps=None,
    loss_weight=None,
):
    """Read counters/weights/optimizer/RNG; return zero counters for a fresh run."""
    directory = Path(directory)
    group = optimizer.param_groups[0]
    options = dict(
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        min_delta=min_delta,
        seed=seed,
        learning_rate=group["lr"],
        weight_decay=group["weight_decay"],
        optimizer_eps=group["eps"],
        ema_decay=ema.decay,
    )
    if isinstance(model, Autoencoder):
        options["validation_seed"] = validation_seeds[0]
        # Old weighted losses have a different scale and checkpoint ranking.
        options["reconstruction_loss"] = "equal_variable_area_mse"
    else:
        options.update(
            validation_seeds=tuple(validation_seeds),
            rollout_steps=rollout_steps,
            loss_weight=loss_weight,
        )
    progress = TrainingProgress()
    if (directory / "model.pt").exists():
        if not resume:
            raise FileExistsError(
                "Selected model already exists; use a new run or RESUME=True"
            )
        return progress
    latest = directory / "latest.pt"
    if latest.exists():
        if not resume:
            raise FileExistsError("Interrupted training exists; set RESUME=True")
        saved = load_checkpoint(latest, map_location=next(model.parameters()).device)
        check_resume_settings(saved["settings"], options)
        if saved["setup_digest"] != file_digest(directory / "setup.pt"):
            raise ValueError("Training inputs changed since the checkpoint")
        if isinstance(model, Autoencoder):
            restore_autoencoder_training(model, optimizer, ema, saved)
        else:
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            ema.state = saved["ema"]
        progress = TrainingProgress(
            saved["epoch"],
            saved["step"],
            saved["best"],
            saved["patience_best"],
            saved["stale"],
            saved["history"],
        )
        restore_rng(saved["rng"])
    checkpoints.atomic_save(options, directory / "training_options.pt")
    return progress


def save_best_weights(directory, ema, epoch, score):
    checkpoints.atomic_save(
        dict(model=ema.state, epoch=epoch, score=score), Path(directory) / "best.pt"
    )


def save_training_progress(directory, model, optimizer, ema, progress):
    directory = Path(directory)
    checkpoints.atomic_save(
        dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            ema=ema.state,
            epoch=progress.epoch,
            step=progress.optimizer_steps,
            history=progress.history,
            best=progress.best_validation,
            patience_best=progress.patience_best,
            stale=progress.stale_epochs,
            settings=load_checkpoint(directory / "training_options.pt"),
            setup_digest=file_digest(directory / "setup.pt"),
            rng=rng_state(next(model.parameters()).device),
        ),
        directory / "latest.pt",
    )
    save_history(progress.history, directory / "history.csv")


def add_autoencoder_history(
    progress,
    train_loss,
    validation_loss,
    seconds,
    *,
    variables,
    train_rmse,
    validation_rmse,
    train_normalized_rmse,
    validation_normalized_rmse,
):
    row = dict(
        epoch=progress.epoch,
        optimizer_steps=progress.optimizer_steps,
        train_loss=train_loss,
        val_loss=validation_loss,
        seconds=seconds,
        stale_epochs=progress.stale_epochs,
    )
    for index, variable in enumerate(variables):
        row[f"train_rmse_{variable.name}"] = float(train_rmse[index])
        row[f"val_rmse_{variable.name}"] = float(validation_rmse[index])
        row[f"train_normalized_rmse_{variable.name}"] = float(
            train_normalized_rmse[index]
        )
        row[f"val_normalized_rmse_{variable.name}"] = float(
            validation_normalized_rmse[index]
        )
    progress.history.append(row)


def add_diffusion_history(progress, training, validation, seconds):
    row = dict(
        epoch=progress.epoch,
        optimizer_steps=progress.optimizer_steps,
        seconds=seconds,
        stale_epochs=progress.stale_epochs,
    )
    row.update(
        {"train_" + name: float(value) for name, value in training.metrics().items()}
    )
    row.update(
        {"val_" + name: float(value) for name, value in validation.metrics().items()}
    )
    progress.history.append(row)


def select_best_weights(directory, model):
    directory = Path(directory)
    path = directory / "model.pt"
    if not path.exists():
        setup = load_checkpoint(directory / "setup.pt")
        save_selection(
            directory,
            parameters=setup["parameters"],
            setup_digest=file_digest(directory / "setup.pt"),
            training_settings=load_checkpoint(directory / "training_options.pt"),
        )
    selected = load_checkpoint(path, map_location=next(model.parameters()).device)
    if selected["setup_digest"] != file_digest(directory / "setup.pt"):
        raise ValueError("Selected weights belong to different preprocessing")
    if isinstance(model, Autoencoder):
        load_autoencoder_weights(model, selected["model"])
    else:
        model.load_state_dict(selected["model"])
    model.eval().requires_grad_(False)
    return selected["epoch"]


def save_latent_cache(
    directory, training, validation, normalizer, training_time, validation_time
):
    directory = Path(directory)
    checkpoints.atomic_save(
        dict(
            training=normalizer.normalize(training),
            validation=normalizer.normalize(validation),
            normalizer=normalizer,
            training_time=training_time,
            validation_time=validation_time,
            model_digest=file_digest(directory / "model.pt"),
            setup_digest=file_digest(directory / "setup.pt"),
        ),
        directory / "latents.pt",
    )
