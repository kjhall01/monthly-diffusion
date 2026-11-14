"""WeatherBench2 selection and bounded-memory cloud reads. No disk data cache."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json

import dask
import numpy as np
import torch
import xarray as xr

from monthly_diffusion.checkpoints import (
    atomic_save,
    autoencoder_parameters,
    load_checkpoint,
)
from monthly_diffusion.rollout import calendar_features


def open_weatherbench(path):
    """Read native Zarr chunks anonymously; ordinary local Zarr paths also work."""
    options = dict(token="anon") if str(path).startswith("gs://") else None
    return xr.open_zarr(
        str(path), consolidated=True, chunks={}, storage_options=options
    )


def select_weather_fields(dataset, surface_variables, pressure_variables, levels):
    """One channel per surface variable or pressure variable/level; no unit conversion."""
    dataset = dataset.rename({"latitude": "lat", "longitude": "lon"}).sortby("lat")
    fields = {name: dataset[name] for name in surface_variables}
    for name in pressure_variables:
        for level in levels:
            fields[f"{name}_{level}"] = dataset[name].sel(level=level, drop=True)
    result = xr.Dataset(fields, attrs=dict(dataset.attrs))
    if "time" in result.dims:
        return result.transpose("time", "lat", "lon")
    return result.transpose("hour", "dayofyear", "lat", "lon")


def weather_statics(dataset):
    return (
        dataset[["geopotential_at_surface", "land_sea_mask"]]
        .rename(
            {
                "latitude": "lat",
                "longitude": "lon",
                "geopotential_at_surface": "phi_s",
                "land_sea_mask": "lsm",
            }
        )
        .sortby("lat")
        .transpose("lat", "lon")
        .compute()
    )


def weather_slices(fields, batch_size, *, shuffle=False, seed=0, max_batches=None):
    """Keep each native time chunk intact; group by climatology block for RAM reuse.

    Shuffling changes calendar-block order and the time chunks within each group.
    Thus chunks from different years mix without re-reading a cloud chunk for
    each individual frame. A bounded export subset is chosen BEFORE grouping.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    chunks = fields.chunks.get("time") if fields.chunks else None
    if chunks is None:
        chunks = [
            min(batch_size, len(fields.time) - i)
            for i in range(0, len(fields.time), batch_size)
        ]
    if max(chunks) > batch_size:
        raise ValueError(
            "BATCH_SIZE must accommodate one native time chunk (8 for WB2)"
        )
    batches, start, stop = [], 0, 0
    for size in chunks:
        if stop + size - start > batch_size:
            batches.append(slice(start, stop))
            start = stop
        stop += size
    if stop > start:
        batches.append(slice(start, stop))
    rng = np.random.default_rng(seed)
    if max_batches is not None and len(batches) > max_batches:
        if max_batches < 1:
            raise ValueError("max_batches must be positive")
        batches = [
            batches[i] for i in rng.choice(len(batches), max_batches, replace=False)
        ]
    groups = {}
    day = fields.time.dt.dayofyear.values
    for batch in batches:
        groups.setdefault(int((day[batch.start] - 1) // 30), []).append(batch)
    keys = list(groups)
    if shuffle:
        rng.shuffle(keys)
    for key in keys:
        if shuffle:
            rng.shuffle(groups[key])
        yield from groups[key]


def read_weather_batch(fields, selection):
    # Bound concurrent chunk decompression as well as the number of batches.
    with dask.config.set(scheduler="threads", num_workers=4):
        return fields.isel(time=selection).compute()


def stream_weather(fields, batch_size, *, shuffle=False, seed=0, max_batches=None):
    """Prefetch one physical batch in RAM while the caller trains on the current one."""
    selections = iter(
        weather_slices(
            fields, batch_size, shuffle=shuffle, seed=seed, max_batches=max_batches
        )
    )
    first = next(selections, None)
    if first is None:
        raise ValueError("No weather timestamps selected")
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(read_weather_batch, fields, first)
        for selection in selections:
            physical = future.result()
            future = executor.submit(read_weather_batch, fields, selection)
            yield physical
        yield future.result()


def weather_tensors(physical, preprocessor, device):
    normalized = preprocessor.encode(physical, "atmosphere")
    names = [v.name for v in preprocessor.variables]
    state = torch.as_tensor(
        np.stack([normalized[n].values for n in names], axis=1), device=device
    )
    target = torch.as_tensor(
        np.stack(
            [
                physical[n].where(preprocessor.stats[n]["data_mask"]).values
                for n in names
            ],
            axis=1,
        ),
        dtype=torch.float32,
        device=device,
    )
    return state, target, calendar_features(physical.time, "day_hour").to(device)


def record_weather_run(
    directory,
    *,
    source,
    climatology,
    surface_variables,
    pressure_variables,
    levels,
    training_period,
    validation_period,
):
    """Save source/split choices once. These dictionaries are disk records only."""
    record = dict(
        source=str(source),
        climatology=str(climatology),
        surface_variables=surface_variables,
        pressure_variables=pressure_variables,
        levels=list(levels),
        training_period=[training_period.start, training_period.stop],
        validation_period=[validation_period.start, validation_period.stop],
    )
    path = Path(directory) / "run.json"
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise ValueError(
                "Weather sources, variables or dates changed; choose a new RUN_DIR"
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2) + "\n")


def save_weather_setup(
    directory, model, preprocessor, statics, training_time, validation_time
):
    path = Path(directory) / "setup.pt"
    parameters = autoencoder_parameters(model)
    if path.exists():
        saved = load_checkpoint(path)
        if saved["parameters"] != parameters:
            raise ValueError("AE architecture changed; choose a new RUN_DIR")
        if not np.array_equal(
            saved["preprocessor"].training_times, training_time
        ) or not np.array_equal(saved["validation_times"], validation_time):
            raise ValueError("Weather timestamps changed; choose a new RUN_DIR")
        xr.testing.assert_identical(saved["statics"], statics)
        return
    atomic_save(
        dict(
            parameters=parameters,
            preprocessor=preprocessor,
            statics=statics,
            validation_times=validation_time.values,
        ),
        path,
    )
