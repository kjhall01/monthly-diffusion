"""Read the data saved in a run's source record; return actual datasets/tensors."""

from pathlib import Path
import numpy as np
from . import sources
from workflows.run_files import read_run, load_latents
from monthly_diffusion.checkpoints import load_checkpoint


def load_reference_data(directory):
    """Open physical fields and statics using the source recorded for this run."""
    source = read_run(directory)["source"]
    return sources.open_canonical_state(
        source["state_zarr"], expected_attrs=source.get("expected_attrs")
    )


def load_run_co2(directory):
    source = read_run(directory)["source"]
    return sources.open_co2(
        source["co2_netcdf"], expected_sha256=source.get("co2_sha256")
    )


def reference_path(directory):
    return read_run(directory)["source"]["state_zarr"]


def load_run_data(directory):
    atmosphere, ocean, statics = load_reference_data(directory)
    return atmosphere, ocean, statics, load_run_co2(directory)


def load_training_data(directory, component):
    run = read_run(directory)
    atmosphere, ocean, statics, co2 = load_run_data(directory)
    fields = atmosphere if component == "atmosphere" else ocean
    training = fields.sel(time=slice(*run["training_period"]))
    validation = fields.sel(time=slice(*run["validation_period"]))
    statics = statics[["phi_s", "lsm"]].transpose("lat", "lon").load()
    return training, validation, statics, co2


def load_training_latents(directory):
    """Return two training arrays, two validation arrays and their shared dates."""
    atmosphere = load_latents(directory, "atmosphere")
    ocean = load_latents(directory, "ocean")
    for split in ("training", "validation"):
        if not np.array_equal(atmosphere[split + "_time"], ocean[split + "_time"]):
            raise ValueError("Atmosphere/ocean latent dates differ")
        if atmosphere[split].shape[1] != ocean[split].shape[1]:
            raise ValueError("Coupled diffusion requires the same latent mesh")
    a = load_checkpoint(Path(directory) / "atmosphere/model.pt")["parameters"]
    o = load_checkpoint(Path(directory) / "ocean/model.pt")["parameters"]
    if (a["mesh_levels"][-1], tuple(a["rotation"])) != (
        o["mesh_levels"][-1],
        tuple(o["rotation"]),
    ):
        raise ValueError("Both AE caches must use the same latent mesh coordinates")
    return (
        atmosphere["training"],
        ocean["training"],
        atmosphere["validation"],
        ocean["validation"],
        atmosphere["training_time"],
        atmosphere["validation_time"],
    )
