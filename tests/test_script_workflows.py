"""The historical paired script retains checkpoint resume and forecast behavior."""

from pathlib import Path
import runpy
import sys

import numpy as np
import xarray as xr
import yaml

from monthly_diffusion import MonthlyDiffusionV1, open_rollout, load_models
from monthly_diffusion.data import sources

ROOT = Path(__file__).resolve().parents[1]


def test_legacy_imports_keep_the_public_model_class():
    # Importing the old class module must not replace the package's class export
    # with a module object in a notebook that mixes old and new import styles.
    import monthly_diffusion
    from monthly_diffusion.MonthlyDiffusionV1 import MonthlyDiffusionV1 as legacy
    from monthly_diffusion.model import MonthlyDiffusionV1 as current
    from monthly_diffusion.forecast import Forecast as old_forecast
    from monthly_diffusion.inference.forecast import Forecast

    assert monthly_diffusion.MonthlyDiffusionV1 is legacy is current
    assert old_forecast is Forecast


def test_legacy_training_and_forecast_scripts(tiny_data, tmp_path, monkeypatch):
    atmosphere, ocean, statics, _, config = tiny_data
    # Two examples of each calendar month give stochastic initialization a
    # finite local variance. Validation uses different, earlier timestamps.
    validation_times = xr.date_range("1958-01-01", periods=24, freq="MS")
    validation = (
        atmosphere.assign_coords(time=validation_times),
        ocean.assign_coords(time=validation_times),
    )
    forcing_times = xr.date_range("1958-01-01", "1970-12-01", freq="MS")
    co2 = xr.DataArray(
        np.linspace(310, 350, len(forcing_times)),
        dims="time",
        coords={"time": forcing_times},
    )
    complete_atmosphere = xr.concat([validation[0], atmosphere], "time")
    complete_ocean = xr.concat([validation[1], ocean], "time")
    # Substitute only source I/O. Training, normalization, caching, model
    # serialization, sampling, decoding, and archive writes all run for real.
    monkeypatch.setattr(
        sources,
        "open_canonical_state",
        lambda *args, **kwargs: (complete_atmosphere, complete_ocean, statics),
    )
    monkeypatch.setattr(sources, "open_co2", lambda *args, **kwargs: co2)
    source_file = tmp_path / "sources.yaml"
    source_file.write_text(
        yaml.safe_dump({"state_zarr": "fixture", "co2_netcdf": "fixture"})
    )
    config_file = tmp_path / "model.yaml"
    config.save(config_file)

    arguments = [
        str(ROOT / "scripts/legacy/train_model.py"),
        str(source_file),
        "--config",
        str(config_file),
        "--output",
        str(tmp_path / "script"),
        "--device",
        "cpu",
        "--threads",
        "1",
        "--load-in-memory",
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    runpy.run_path(arguments[0], run_name="__main__")
    checkpoint = tmp_path / "script/model.pt"
    actual = MonthlyDiffusionV1.load(checkpoint, device="cpu")
    expected = actual

    cache_time = (tmp_path / "script/latents.pt").stat().st_mtime_ns
    monkeypatch.setattr(sys, "argv", arguments + ["--resume"])
    runpy.run_path(arguments[0], run_name="__main__")
    resumed = MonthlyDiffusionV1.load(checkpoint, device="cpu")
    assert resumed.weight_digest() == expected.weight_digest()
    assert (tmp_path / "script/latents.pt").stat().st_mtime_ns == cache_time

    for mode in (
        "physical",
        "gaussian_start",
        "encoded_noise_start",
        "stochastic_start",
    ):
        output = tmp_path / mode
        initial_states = (
            (validation[0].isel(time=slice(0, 2)), validation[1].isel(time=slice(0, 2)))
            if mode == "physical"
            else ()
        )
        options = {} if mode == "physical" else {mode: "1958-03"}
        expected_fields = expected.rollout(
            *initial_states, co2=co2, months=3, members=2, seed=41, **options
        )
        forecast_arguments = [
            str(ROOT / "scripts/legacy/forecast.py"),
            str(checkpoint),
            str(source_file),
            "--output",
            str(output),
            "--months",
            "2",
            "--members",
            "2",
            "--seed",
            "41",
            "--device",
            "cpu",
            "--threads",
            "1",
            "--require-trained",
        ]
        if mode != "physical":
            forecast_arguments += ["--" + mode.replace("_", "-"), "1958-03"]
        monkeypatch.setattr(sys, "argv", forecast_arguments)
        runpy.run_path(forecast_arguments[0], run_name="__main__")
        first_month = output / "month-000001.zarr/.zmetadata"
        first_write = first_month.stat().st_mtime_ns
        # Extend through the public CLI and verify it preserves existing shards.
        forecast_arguments[forecast_arguments.index("--months") + 1] = "3"
        monkeypatch.setattr(sys, "argv", forecast_arguments + ["--resume", "--extend"])
        runpy.run_path(forecast_arguments[0], run_name="__main__")
        with open_rollout(output) as archived:
            # Monthly archives intentionally store float32, as before this refactor.
            xr.testing.assert_identical(
                archived.compute(), expected_fields.astype("float32")
            )
        assert first_month.stat().st_mtime_ns == first_write

        # The directory loader also continues existing combined-checkpoint archives.
        loaded = load_models(tmp_path / "script", device="cpu")
        extended = loaded.rollout(
            *initial_states,
            co2=co2,
            months=4,
            members=2,
            seed=41,
            path=output,
            resume=True,
            extend=True,
            **options,
        )
        expected_extended = expected.rollout(
            *initial_states, co2=co2, months=4, members=2, seed=41, **options
        )
        xr.testing.assert_identical(
            extended.compute(), expected_extended.astype("float32")
        )
        assert first_month.stat().st_mtime_ns == first_write
