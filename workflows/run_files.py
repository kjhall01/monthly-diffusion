"""The shared run directory: source record, atomic files, and identity checks.

No model construction, training, or workflow dispatch lives here.
"""

from monthly_diffusion.checkpoints import load_checkpoint
from pathlib import Path
import csv
import json
from monthly_diffusion.checkpoints import atomic_save

from monthly_diffusion.checkpoints import (
    file_digest as file_digest,
    read_run,
    load_latents as load_latents,
    model_identity as model_identity,
)


def create_run(directory, *, state_path, co2_path, training_period, validation_period):
    """Create the source/split record once; never silently repurpose a run."""
    directory = Path(directory)
    record = dict(
        format_version=2,
        source=dict(state_zarr=str(state_path), co2_netcdf=str(co2_path)),
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


def save_history(history, path):
    path = Path(path)
    pending = path.with_suffix(".partial.csv")
    with pending.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(dict.fromkeys(key for row in history for key in row)),
        )
        writer.writeheader()
        writer.writerows(history)
    pending.replace(path)


def save_selection(directory, *, parameters, setup_digest, training_settings):
    """Copy the chosen EMA weights into the component's final model file."""
    directory = Path(directory)
    selected = load_checkpoint(
        directory / "best.pt", map_location="cpu", weights_only=False
    )
    selected.update(
        format_version=2,
        parameters=parameters,
        setup_digest=setup_digest,
        training_settings=training_settings,
    )
    atomic_save(selected, directory / "model.pt")
    return selected


def check_script_arguments():
    """Reject old CLI flags before a plain script can start the wrong experiment."""
    import sys

    if len(sys.argv) > 1:
        raise SystemExit(
            "These scripts use editable Python parameters. Set RUN_DIR and the other values in the file, then run it without command-line arguments."
        )


def run_inputs(directory):
    """Return the source paths and split slices recorded by atmosphere training."""
    record = read_run(directory)
    return (
        Path(record["source"]["state_zarr"]),
        Path(record["source"]["co2_netcdf"]),
        slice(*record["training_period"]),
        slice(*record["validation_period"]),
    )
