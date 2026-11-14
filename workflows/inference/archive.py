"""Monthly forecast files, provenance, and exact resume/extension checks."""

from monthly_diffusion.checkpoints import load_checkpoint
from pathlib import Path
import json
import logging
import numpy as np
import torch
import xarray as xr
from numcodecs import Blosc


def open_rollout(path):
    """Lazily concatenate completed compressed monthly shards, without loading them."""
    shards = sorted(Path(path).glob("month-*.zarr"))
    if not shards:
        raise FileNotFoundError(f"No completed months in {path}")
    return xr.concat(
        [xr.open_zarr(p, consolidated=True) for p in shards],
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="equals",
    )


def prepare_archive(
    forecast,
    path,
    metadata,
    *,
    resume=False,
    extend=False,
    spinup_months=0,
    save_spinup_latents=False,
    save_initial_state=False,
):
    """Check archive identity and restore its last complete history/noise/RNG state."""
    if extend and not resume:
        raise ValueError("Extending requires an archive path and resume=True")
    model = forecast.model
    months = metadata["months"]
    initial_state = forecast.state_dict()
    support = getattr(getattr(model, "preprocessor", None), "ocean_support_mask", None)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    record = path / "forecast.json"
    if record.exists():
        if not resume:
            raise FileExistsError("Forecast exists; use resume=True")
        previous = json.loads(record.read_text())
        if (
            extend
            and previous["initialization_sha256"] != metadata["initialization_sha256"]
        ):
            mutable = {"months", "valid_times", "co2_ppm", "initialization_sha256"}
            if {k: v for k, v in previous.items() if k not in mutable} != {
                k: v for k, v in metadata.items() if k not in mutable
            }:
                raise ValueError(
                    "Forecast identity/configuration/forcing differs from archive"
                )
            # Extension means continuation of the ARCHIVED trajectory, not
            # a fresh initialization. MPS re-encoding need not be bitwise
            # reproducible. Check the initial current slot against the
            # previous slot retained in the first forecast, then discard
            # the re-encoded histories; load_state_dict below restores the
            # actual complete history/noise/RNG state at the saved endpoint.
            if spinup_months:
                raise ValueError("Spin-up initialization differs from archive")
            first = load_checkpoint(
                path / "month-000001.zarr" / "forecast_state.pt",
                weights_only=False,
                map_location="cpu",
            )
            if np.datetime64(first["times"][1]) != np.datetime64(forecast.times[0]):
                raise ValueError(
                    "Initialization date differs from archived first history"
                )
            differences = []
            for fresh, saved in zip(forecast.history, first["history"]):
                fresh, saved = fresh[:, 0].cpu(), saved[:, 1]
                torch.testing.assert_close(fresh, saved, rtol=1e-5, atol=1e-5)
                differences.append(float((fresh - saved).abs().max()))
            logging.getLogger(__name__).info(
                "Using archived initialization identity; re-encoding max absolute differences (atmosphere, ocean): %s; fresh histories discarded",
                differences,
            )
            metadata["initialization_sha256"] = previous["initialization_sha256"]
        extending = previous != metadata
        if extending:
            # Only the horizon may change. In particular, every prescribed
            # forcing value already used must match, not just its filename.
            prefix = dict(
                metadata,
                months=previous["months"],
                valid_times=metadata["valid_times"][: previous["months"]],
                co2_ppm=metadata["co2_ppm"][: previous["months"]],
            )
            if not extend or months <= previous["months"] or prefix != previous:
                different = [
                    k
                    for k in previous.keys() | prefix.keys()
                    if previous.get(k) != prefix.get(k)
                ]
                raise ValueError(
                    f"Forecast identity/configuration/forcing differs from archive: {different}"
                )
        shards = sorted(path.glob("month-*.zarr"))
        if shards:
            forecast.load_state_dict(
                load_checkpoint(
                    shards[-1] / "forecast_state.pt",
                    weights_only=False,
                    map_location="cpu",
                )
            )
            if [p.name for p in shards] != [
                f"month-{i:06d}.zarr"
                for i in range(1, forecast.step - spinup_months + 1)
            ]:
                raise ValueError("Forecast archive has missing monthly shards")
        if extending:
            if len(shards) != previous["months"]:
                raise ValueError(
                    "Finish the original forecast horizon before extending it"
                )
            if np.datetime64(forecast.times[0]) != np.datetime64(
                previous["valid_times"][-1]
            ):
                raise ValueError(
                    "Restart state date differs from the last archived forecast month"
                )
            # Keep the original identity, then commit the new horizon
            # atomically. Completed fields and restart states are untouched.
            backup = path / f"forecast_through_{previous['months']:06d}.json"
            if backup.exists() and backup.read_bytes() != record.read_bytes():
                raise ValueError("Conflicting original-horizon provenance")
            backup.write_bytes(record.read_bytes())
            pending = path / "forecast.partial.json"
            pending.write_text(json.dumps(metadata, indent=2))
            pending.replace(record)
            logging.getLogger(__name__).info(
                "Extending saved forecast from %s: %d -> %d months; restoring latent/noise/RNG state",
                previous["valid_times"][-1],
                previous["months"],
                months,
            )
    else:
        if support is not None:
            # Keep the actual training-fitted mask beside the forecast so
            # streaming diagnostics can apply it to reference data too,
            # without instantiating the model or inferring masks anew.
            support.astype("int8").to_dataset(name="ocean_support").to_netcdf(
                path / "ocean_support.partial.nc"
            )
            (path / "ocean_support.partial.nc").replace(path / "ocean_support.nc")
        record.write_text(json.dumps(metadata, indent=2))
    if support is not None:
        archived_support = xr.load_dataset(path / "ocean_support.nc").ocean_support
        xr.testing.assert_equal(
            archived_support.astype(bool), support.astype(bool).rename("ocean_support")
        )
    if save_initial_state and not (path / "initial_state.pt").exists():
        # Keep the initial random histories for audit, not decoded spin-up fields.
        torch.save(initial_state, path / "initial_state.pt")
    if save_spinup_latents:
        spinup_path = path / "spinup_latents"
        spinup_path.mkdir(exist_ok=True)
        blocks = sorted(spinup_path.glob("year-*.pt"))
        if [p.name for p in blocks] != [
            f"year-{i:04d}.pt" for i in range(1, len(blocks) + 1)
        ]:
            raise ValueError("Spin-up archive has missing annual blocks")
        if len(blocks) > (spinup_months + 11) // 12:
            raise ValueError("Spin-up archive exceeds requested duration")
        if forecast.step > spinup_months and len(blocks) != (spinup_months + 11) // 12:
            raise ValueError("Decoded archive is missing retained spin-up latents")
        if forecast.step == 0 and blocks:
            state = load_checkpoint(blocks[-1], weights_only=False, map_location="cpu")[
                "state"
            ]
            if state["step"] != min(12 * len(blocks), spinup_months):
                raise ValueError("Spin-up block and restart step disagree")
            forecast.load_state_dict(state)
    elif forecast.step == 0 and (path / "spinup_state.pt").exists():
        forecast.load_state_dict(
            load_checkpoint(path / "spinup_state.pt", weights_only=False, map_location="cpu")
        )
    return path


def write_month(fields, forecast, path, *, spinup_months=0):
    """Commit fields and restart state together; incomplete writes stay .partial."""
    path = Path(path)
    saved_step = forecast.step - spinup_months
    final = path / f"month-{saved_step:06d}.zarr"
    partial = path / f"month-{saved_step:06d}.partial"
    codec = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    encoding = {
        name: dict(
            dtype="float32",
            compressor=codec,
            chunks=(1, 1, min(180, fields.sizes["lat"]), min(360, fields.sizes["lon"])),
        )
        for name in fields.data_vars
    }
    fields.to_zarr(partial, mode="w", consolidated=True, encoding=encoding)
    torch.save(forecast.state_dict(), partial / "forecast_state.pt")
    partial.rename(final)
