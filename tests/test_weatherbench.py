"""Weather calendar, streaming statistics, and a real script train/export cycle."""

import csv
import pickle

import numpy as np
import pytest
import torch
import xarray as xr

from monthly_diffusion import Autoencoder, Variable, load_autoencoder
from monthly_diffusion.rollout import (
    calendar_features,
    month_features,
    validate_six_hourly,
)
from monthly_diffusion.weather import WeatherPreprocessor
from workflows.data.weatherbench import (
    open_weatherbench,
    select_weather_fields,
    weather_statics,
    weather_slices,
    stream_weather,
    weather_tensors,
)
from workflows.training.losses import reconstruction_mse


@pytest.fixture
def weather_data(tmp_path):
    lat, lon = np.linspace(-90, 90, 18), np.arange(0, 360, 10.0)
    hour, day, levels = np.array([0, 6, 12, 18]), np.arange(1, 367), [500, 850]
    shape = (4, 366, len(lon), len(lat))
    season = np.sin(2 * np.pi * (day[None, :, None, None] - 1) / 366)
    daily = np.cos(2 * np.pi * hour[:, None, None, None] / 24)
    mean = np.broadcast_to(
        280 + 10 * season + daily + lat[None, None, None, :] / 10, shape
    )
    climate = xr.Dataset(
        {
            "2m_temperature": (
                ("hour", "dayofyear", "longitude", "latitude"),
                mean.astype("float32"),
            ),
            "total_precipitation_6hr": (
                ("hour", "dayofyear", "longitude", "latitude"),
                np.broadcast_to(0.001 + 0.0003 * season + 0.0001 * daily, shape).astype(
                    "float32"
                ),
            ),
            "temperature": (
                ("hour", "dayofyear", "level", "longitude", "latitude"),
                np.stack([mean - 30, mean - 10], axis=2).astype("float32"),
            ),
        },
        coords=dict(
            hour=hour, dayofyear=day, longitude=lon, latitude=lat, level=levels
        ),
    )
    climate["2m_temperature"].attrs["units"] = "K"
    climate.temperature.attrs["units"] = "K"
    climate.total_precipitation_6hr.attrs["units"] = "m"
    train_time = xr.date_range("2000-02-28", periods=16, freq="6h")
    val_time = xr.date_range("2001-02-28", periods=8, freq="6h")
    dates = xr.DataArray(np.concatenate([train_time, val_time]), dims="time")
    dates = dates.assign_coords(time=dates)
    raw = climate.sel(hour=dates.dt.hour, dayofyear=dates.dt.dayofyear).drop_vars(
        ["hour", "dayofyear"]
    )
    rng = np.random.default_rng(42)
    for name in raw:
        scale = 0.0001 if name == "total_precipitation_6hr" else 2.0
        raw[name] = raw[name] + xr.DataArray(
            rng.normal(size=raw[name].shape).astype("float32") * scale,
            dims=raw[name].dims,
        )
        raw[name].attrs = dict(climate[name].attrs)
    raw["2m_temperature"].values[:16, 3, 3] = 7.0
    raw["2m_temperature"].values[16:, 3, 3] = 9.0
    raw["2m_temperature"].values[:, 0, 0] = np.nan
    raw["geopotential_at_surface"] = (
        ("longitude", "latitude"),
        rng.normal(size=(len(lon), len(lat))).astype("float32"),
    )
    raw["land_sea_mask"] = xr.zeros_like(raw.geopotential_at_surface)
    source, clim_path = tmp_path / "era5.zarr", tmp_path / "climatology.zarr"
    raw.chunk({"time": 8, "level": 2}).to_zarr(source, consolidated=True)
    climate.chunk({"dayofyear": 30, "hour": 4, "level": 1}).to_zarr(
        clim_path, consolidated=True
    )
    surfaces = ["2m_temperature", "total_precipitation_6hr"]
    raw = open_weatherbench(source)
    fields = select_weather_fields(raw, surfaces, ["temperature"], levels)
    baseline = select_weather_fields(
        open_weatherbench(clim_path), surfaces, ["temperature"], levels
    )
    return source, clim_path, fields, baseline, weather_statics(raw)


def test_calendar_is_cyclic_in_year_and_day_and_preserves_monthly():
    time = xr.DataArray(xr.date_range("2000-02-28", periods=12, freq="6h"), dims="time")
    features = calendar_features(time, "day_hour")
    assert features.shape == (12, 4)
    torch.testing.assert_close(features[0, 2:], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(
        features[1, 2:], torch.tensor([1.0, 0.0]), atol=1e-7, rtol=0
    )
    torch.testing.assert_close(features[0, 2:], features[4, 2:])
    assert not torch.equal(features[0, :2], features[4, :2])
    torch.testing.assert_close(
        calendar_features(time), month_features(time), atol=0, rtol=0
    )
    validate_six_hourly(time)
    with pytest.raises(ValueError, match="six-hourly"):
        validate_six_hourly(time.isel(time=[0, 2]))
    with pytest.raises(ValueError, match="six-hourly"):
        validate_six_hourly(time + np.timedelta64(1, "h"))


def test_streaming_statistics_climatology_masks_and_serialization(weather_data):
    _, _, fields, baseline, statics = weather_data
    training, validation = fields.isel(time=slice(0, 16)), fields.isel(
        time=slice(16, None)
    )
    variables = [Variable(n, "atmosphere", fields[n].attrs["units"]) for n in fields]
    pre = WeatherPreprocessor(variables, baseline)
    pre.fit_batches(stream_weather(training, 8), statics, training.time, training.attrs)
    climate = (
        baseline.sel(dayofyear=training.time.dt.dayofyear, hour=training.time.dt.hour)
        .drop_vars(["hour", "dayofyear"])
        .compute()
    )
    anomalies = (training.compute() - climate).transpose("time", "lat", "lon")
    for name in fields:
        stats = pre.stats[name]
        xr.testing.assert_allclose(
            stats["center"].where(stats["data_mask"]),
            anomalies[name].mean("time"),
            atol=1e-8,
        )
        expected_std = np.sqrt(
            ((anomalies[name] - anomalies[name].mean("time")) ** 2).mean("time")
        ).clip(min=stats["std_floor"])
        xr.testing.assert_allclose(
            stats["std"].where(stats["data_mask"]), expected_std, rtol=2e-5
        )
    encoded = pre.encode(validation.compute(), "atmosphere")
    restored = pre.decode(encoded, "atmosphere")
    expected = validation.compute()
    expected["2m_temperature"].values[:, 3, 3] = 7
    xr.testing.assert_allclose(restored, expected, atol=2e-5)
    state, target, _ = weather_tensors(validation.compute(), pre, "cpu")
    assert torch.isnan(state[:, 0, 3, 3]).all()
    assert torch.all(target[:, 0, 3, 3] == 9)
    physical = pre.decode_tensor(
        torch.nan_to_num(state), "atmosphere", time=validation.time
    )
    assert torch.all(physical[:, 0, 3, 3] == 7)
    assert reconstruction_mse(physical, target, torch.tensor(statics.lat.values))[0] > 0
    np.testing.assert_allclose(
        physical.numpy(),
        np.stack([expected[n].values for n in fields], 1),
        atol=2e-5,
        equal_nan=True,
    )
    # The source is disk-backed and lazy. Neither streamed states nor decoded
    # climatology blocks may be embedded in the exported preprocessing object.
    assert len(pre._climatology_blocks) <= 2
    blob = pickle.dumps(pre)
    assert len(blob) < 1_000_000
    loaded = pickle.loads(blob)
    assert not loaded._climatology_blocks
    xr.testing.assert_identical(loaded.decode(encoded, "atmosphere"), restored)
    with pytest.raises(ValueError, match="timestamps"):
        loaded.decode_tensor(state, "atmosphere")
    # Match WB2's dayofyear lookup: March 1 non-leap and Feb 29 leap are day 60.
    dates = xr.DataArray(
        np.array(["2000-02-29T06", "2001-03-01T06"], dtype="datetime64[h]"), dims="time"
    )
    actual = loaded.baseline(dates)
    xr.testing.assert_equal(
        actual.isel(time=0, drop=True), actual.isel(time=1, drop=True)
    )


def test_chunk_batches_are_bounded_complete_and_reproducible():
    time = xr.date_range("2000-01-01", periods=800, freq="6h")
    fields = (
        xr.Dataset({"x": ("time", np.arange(len(time)))}, coords={"time": time})
        .chunk({"time": 8})
        .isel(time=slice(3, -2))
    )
    first = list(weather_slices(fields, 16, shuffle=True, seed=2))
    assert first == list(weather_slices(fields, 16, shuffle=True, seed=2))
    assert first != list(weather_slices(fields, 16, shuffle=True, seed=3))
    assert max(s.stop - s.start for s in first) <= 16
    np.testing.assert_array_equal(
        np.sort(np.concatenate([np.arange(s.start, s.stop) for s in first])),
        np.arange(len(fields.time)),
    )
    observed = np.concatenate(
        [b.x.values for b in stream_weather(fields, 16, shuffle=True, seed=2)]
    )
    np.testing.assert_array_equal(np.sort(observed), fields.x.values)
    with pytest.raises(ValueError, match="native time chunk"):
        list(weather_slices(fields, 4))
    assert len(list(weather_slices(fields, 16, seed=2, max_batches=4))) == 4


def test_weather_training_script_and_portable_physical_reconstruction(
    weather_data, tmp_path, run_script
):
    source, clim_path, fields, _, _ = weather_data
    run = tmp_path / "run"
    params = dict(
        RUN_DIR=run,
        DATA=str(source),
        CLIMATOLOGY=str(clim_path),
        SURFACE_VARIABLES=["2m_temperature", "total_precipitation_6hr"],
        PRESSURE_VARIABLES=["temperature"],
        LEVELS=[500, 850],
        TRAIN_PERIOD=slice("2000-02-28", "2000-03-02"),
        VALIDATION_PERIOD=slice("2001-02-28", "2001-03-01"),
        CLIMATOLOGY_YEARS=(2000, 2000),
        DEVICE="cpu",
        THREADS=1,
        WIDTH=8,
        LATENT_CHANNELS=4,
        MESH_LEVELS=(1, 0),
        REFINERS=(1, 1),
        BATCH_SIZE=8,
        EPOCHS=1,
        LATENT_NORMALIZATION_BATCHES=2,
    )
    result = run_script("train_weatherbench_autoencoder", **params)
    model = Autoencoder.load(run / "atmosphere/autoencoder.pt", device="cpu")
    from_run = load_autoencoder(run, "atmosphere", device="cpu")
    assert model.encoder.calendar == model.decoder.calendar == "day_hour"
    assert not model.encoder.condition_on_co2
    assert model.encoder.film[1].month[0].in_features == 4
    assert model.latent_normalizer.fitted and from_run.latent_normalizer.fitted
    validation = fields.sel(time=params["VALIDATION_PERIOD"]).compute()
    predicted = model.reconstruct(validation, batch_size=8)
    xr.testing.assert_identical(
        from_run.reconstruct(validation, batch_size=8), predicted
    )
    with (run / "atmosphere/history.csv").open() as f:
        history = list(csv.DictReader(f))
    row = history[-1]
    area = np.cos(np.deg2rad(validation.lat)).clip(min=0)
    for name in validation:
        error = predicted[name] - validation[name]
        expected = np.sqrt(
            (
                (error**2 * area).sum(("lat", "lon"))
                / area.broadcast_like(error).where(error.notnull()).sum(("lat", "lon"))
            ).mean()
        )
        assert float(row["val_rmse_" + name]) == pytest.approx(
            float(expected), rel=2e-5
        )
    assert not (run / "atmosphere/latents.pt").exists()
    assert not list(run.rglob("*.zarr"))
    assert not list(run.rglob("*.npy"))
    assert result["preprocessor"].stats["2m_temperature"]["constant_mask"][3, 3]
    before = (run / "atmosphere/model.pt").read_bytes()
    run_script("train_weatherbench_autoencoder", **params, RESUME=True)
    assert (run / "atmosphere/model.pt").read_bytes() == before
    with pytest.raises(ValueError, match="preprocessing changed"):
        run_script(
            "train_weatherbench_autoencoder",
            **params,
            RESUME=True,
            GRIDPOINT_STD_FLOOR=0.1,
        )


def test_small_humidity_variations_are_not_lost_to_absolute_floor(weather_data):
    _, _, fields, baseline, statics = weather_data
    training = fields.isel(time=slice(0, 16))
    humidity = xr.Dataset({"q50": (training.temperature_500 - 250) * 1e-8})
    climate = xr.Dataset({"q50": (baseline.temperature_500 - 250) * 1e-8})
    pre = WeatherPreprocessor([Variable("q50", "atmosphere")], climate)
    pre.fit_batches(stream_weather(humidity, 8), statics, humidity.time, {})
    assert pre.stats["q50"]["std_floor"] < 1e-8
    assert float(pre.stats["q50"]["std"].max()) < 1e-6
    z = pre.encode(humidity.compute(), "atmosphere").q50
    np.testing.assert_allclose(z.std("time"), 1.0, rtol=1e-5)
