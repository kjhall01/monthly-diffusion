"""The shared run directory: source record, atomic files, and identity checks.

No model construction, training, or workflow dispatch lives here.
"""

from pathlib import Path
import csv
import hashlib
import json
import torch
from .checkpoints import atomic_save


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def create_run(directory, source, training_period, validation_period):
    """Create the source/split record once; never silently repurpose a run."""
    directory = Path(directory)
    record = dict(
        format_version=2,
        source=source,
        training_period=[training_period.start, training_period.stop],
        validation_period=[validation_period.start, validation_period.stop],
    )
    record = json.loads(json.dumps(record, default=str))
    path = directory / "run.json"
    if path.exists():
        if read_run(directory) != record:
            raise ValueError("Run source or periods changed; choose a new RUN_DIR")
    else:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(
                "Use a new RUN_DIR; this directory already contains another run"
            )
        directory.mkdir(parents=True, exist_ok=True)
        pending = directory / "run.partial.json"
        pending.write_text(json.dumps(record, indent=2) + "\n")
        pending.replace(path)
    return record


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


def save_history(history, path):
    path = Path(path)
    pending = path.with_suffix(".partial.csv")
    with pending.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    pending.replace(path)


def save_selection(directory, *, parameters, setup_digest, training_settings):
    """Copy the chosen EMA weights into the component's final model file."""
    directory = Path(directory)
    selected = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
    selected.update(
        format_version=2,
        parameters=parameters,
        setup_digest=setup_digest,
        training_settings=training_settings,
    )
    atomic_save(selected, directory / "model.pt")
    return selected


def load_latents(directory, component):
    """Read a fixed training/validation cache and verify the selected AE identity."""
    directory = Path(directory) / component
    cache = torch.load(directory / "latents.pt", map_location="cpu", weights_only=False)
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


def check_script_arguments():
    """Reject old CLI flags before a plain script can start the wrong experiment."""
    import sys

    if len(sys.argv) > 1:
        raise SystemExit(
            "These scripts use editable Python parameters. Set RUN_DIR and the other values in the file, then run it without command-line arguments."
        )
