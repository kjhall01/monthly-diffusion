"""Atomic annual reconstructions, with latents and physical fields kept together."""

from pathlib import Path
import hashlib
import json
import numpy as np
import xarray as xr
from numcodecs import Blosc


def checkpoint_digest(path):
    """Bind generated products to the complete saved scientific state."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_reconstruction_archive(path, metadata, *, resume=False):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    record = path / "reconstruction.json"
    if record.exists():
        if not resume:
            raise FileExistsError(
                "Reconstruction exists; use --resume or another output"
            )
        if json.loads(record.read_text()) != metadata:
            raise ValueError(
                "Checkpoint, source, dates, or forcing differ from reconstruction archive"
            )
    else:
        if list(path.glob("year-*.zarr")):
            raise ValueError("Annual shards exist without reconstruction provenance")
        pending = path / "reconstruction.partial.json"
        pending.write_text(json.dumps(metadata, indent=2))
        pending.replace(record)
    return path


def completed_year(path, times):
    """Only a fully renamed shard counts; validate its calendar before skipping."""
    path = Path(path)
    if not path.exists():
        return False
    with xr.open_zarr(path, consolidated=True) as saved:
        if not np.array_equal(saved.time.values, times.values):
            raise ValueError(f"Reconstruction dates differ in {path}")
        if not {"atmosphere_latent", "ocean_latent"}.issubset(saved.data_vars):
            raise ValueError(f"Reconstruction shard is missing its latents: {path}")
    return True


def write_reconstruction_year(fields, latents, path, *, co2):
    """Save fixed transition-normalized latents and float32 physical fields."""
    path = Path(path)
    partial = path.with_suffix(".partial")
    if path.exists():
        raise FileExistsError(path)
    products = xr.merge(
        [
            *fields,
            latents[0].rename("atmosphere_latent"),
            latents[1].rename("ocean_latent"),
        ],
        compat="no_conflicts",
    ).assign_coords(co2=co2)
    codec = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    encoding = {
        name: {
            "dtype": "float32",
            "compressor": codec,
            "chunks": tuple(
                1 if dim == "time" else size for dim, size in field.sizes.items()
            ),
        }
        for name, field in products.data_vars.items()
    }
    products.to_zarr(partial, mode="w", consolidated=True, encoding=encoding)
    partial.rename(path)


def open_reconstruction(path):
    """Open annual shards lazily; callers may select fields or saved latents."""
    shards = sorted(Path(path).glob("year-*.zarr"))
    if not shards:
        raise FileNotFoundError(f"No completed reconstruction years in {path}")
    return xr.open_mfdataset(
        shards,
        engine="zarr",
        combine="nested",
        concat_dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="equals",
        join="exact",
    )
