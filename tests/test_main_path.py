"""Exercise the main scripts with real training, serialization and inference."""

from pathlib import Path
import runpy
import sys
import numpy as np
import pytest
import torch
import xarray as xr
import yaml
from monthly_diffusion import MonthlyDiffusionV1, open_rollout
from monthly_diffusion.data import sources
from monthly_diffusion.inference.reconstruction import open_reconstruction
from monthly_diffusion.training import latents

ROOT = Path(__file__).resolve().parents[1]


def run_script(monkeypatch, name, run, *arguments):
    script = ROOT / "scripts" / (name + ".py")
    monkeypatch.setattr(
        sys, "argv", [str(script), "--run", str(run), *map(str, arguments)]
    )
    runpy.run_path(str(script), run_name="__main__")


def assert_weights_equal(actual, expected):
    for name, value in actual.state_dict().items():
        torch.testing.assert_close(
            value, expected.state_dict()[name], rtol=0, atol=0, equal_nan=True
        )


def test_seven_stage_path(tiny_data, tmp_path, monkeypatch):
    a, o, statics, _, config = tiny_data
    val_times = xr.date_range("1958-01-01", periods=24, freq="MS")
    all_a = xr.concat([a.assign_coords(time=val_times), a], "time")
    all_o = xr.concat([o.assign_coords(time=val_times), o], "time")
    times = xr.date_range("1958-01-01", "1970-12-01", freq="MS")
    co2 = xr.DataArray(
        np.linspace(310, 350, len(times)), dims="time", coords={"time": times}
    )
    monkeypatch.setattr(
        sources, "open_canonical_state", lambda *args, **kwargs: (all_a, all_o, statics)
    )
    monkeypatch.setattr(sources, "open_co2", lambda *args, **kwargs: co2)
    source = tmp_path / "source.yaml"
    source.write_text(
        yaml.safe_dump({"state_zarr": "fixture", "co2_netcdf": "fixture"})
    )
    config_file = tmp_path / "config.yaml"
    config.save(config_file)
    run = tmp_path / "run"
    initial = MonthlyDiffusionV1(statics=statics, config=config, device="cpu")
    initial.prepare(a, o, co2=co2)
    common = ("--device", "cpu", "--threads", "1")
    run_script(
        monkeypatch,
        "train_atmosphere_autoencoder",
        run,
        "--data",
        source,
        "--config",
        config_file,
        *common,
    )
    atmos = MonthlyDiffusionV1.load(run / "model.pt", device="cpu")
    assert (
        "atmosphere_autoencoder" in atmos.stages and "autoencoder" not in atmos.stages
    )
    assert_weights_equal(atmos.autoencoder.ocean, initial.autoencoder.ocean)
    assert_weights_equal(atmos.transition, initial.transition)
    assert atmos.weight_digest(autoencoder_only=True) != initial.weight_digest(
        autoencoder_only=True
    )
    with pytest.raises(ValueError, match="separate AE stages"):
        atmos.train(
            a,
            o,
            validation=(all_a.sel(time="1958"), all_o.sel(time="1958")),
            co2=co2,
            directory=run,
            resume=True,
        )

    # Simulate interruption after ocean selection, before latent caching finishes.
    cache_latents = latents.cache_latents

    def interrupt_cache(*args, **kwargs):
        raise RuntimeError("simulated cache interruption")

    monkeypatch.setattr(latents, "cache_latents", interrupt_cache)
    with pytest.raises(RuntimeError, match="simulated cache interruption"):
        run_script(monkeypatch, "train_ocean_autoencoder", run, *common)
    ocean_selected = MonthlyDiffusionV1.load(run / "model.pt", device="cpu")
    assert (
        "ocean_autoencoder" in ocean_selected.stages
        and "autoencoder" not in ocean_selected.stages
    )
    monkeypatch.setattr(latents, "cache_latents", cache_latents)
    run_script(monkeypatch, "train_ocean_autoencoder", run, "--resume", *common)
    both = MonthlyDiffusionV1.load(run / "model.pt", device="cpu")
    assert_weights_equal(both.autoencoder.atmosphere, atmos.autoencoder.atmosphere)
    assert_weights_equal(both.autoencoder.ocean, ocean_selected.autoencoder.ocean)
    assert_weights_equal(both.transition, initial.transition)
    assert "autoencoder" in both.stages and both.latent_normalizers
    cache_time = (run / "latents.pt").stat().st_mtime_ns
    run_script(monkeypatch, "train_latent_diffusion", run, *common)
    trained = MonthlyDiffusionV1.load(run / "model.pt", device="cpu")
    assert trained.weight_digest(autoencoder_only=True) == both.weight_digest(
        autoencoder_only=True
    )
    assert trained.weight_digest() != both.weight_digest()
    assert (run / "latents.pt").stat().st_mtime_ns == cache_time
    for name in (
        "train_atmosphere_autoencoder",
        "train_ocean_autoencoder",
        "train_latent_diffusion",
    ):
        with pytest.raises(SystemExit) as stopped:
            run_script(monkeypatch, name, run, "--resume", *common)
        assert stopped.value.code == 0
    assert (
        MonthlyDiffusionV1.load(run / "model.pt", device="cpu").weight_digest()
        == trained.weight_digest()
    )

    run_script(
        monkeypatch,
        "generate_ae_reconstruction",
        run,
        "--start",
        "1966-01",
        "--end",
        "1967-12",
        *common,
    )
    za, zo = trained.encode(a, o, co2=co2)
    expected = xr.merge(trained.decode(za, zo, co2=co2))
    with open_reconstruction(run / "ae_reconstruction") as reconstructed:
        xr.testing.assert_allclose(
            reconstructed[["t2m", "sst"]].drop_vars("co2").compute(), expected
        )
        xr.testing.assert_allclose(
            reconstructed.atmosphere_latent.drop_vars("co2").compute(),
            za.rename("atmosphere_latent"),
        )
    first = run / "ae_reconstruction/year-1966.zarr/.zmetadata"
    written = first.stat().st_mtime_ns
    run_script(
        monkeypatch,
        "generate_ae_reconstruction",
        run,
        "--start",
        "1966-01",
        "--end",
        "1967-12",
        "--resume",
        *common,
    )
    assert first.stat().st_mtime_ns == written
    with pytest.raises(ValueError, match="differ from reconstruction archive"):
        run_script(
            monkeypatch,
            "generate_ae_reconstruction",
            run,
            "--start",
            "1966-01",
            "--end",
            "1966-12",
            "--resume",
            *common,
        )

    ar_args = (
        "--stochastic-start",
        "1966-01",
        "--members",
        "2",
        "--seed",
        "41",
        *common,
    )
    run_script(monkeypatch, "generate_ar_rollout", run, "--months", "12", *ar_args)
    first_ar = run / "ar_rollout/month-000001.zarr/.zmetadata"
    ar_written = first_ar.stat().st_mtime_ns
    run_script(
        monkeypatch,
        "generate_ar_rollout",
        run,
        "--months",
        "24",
        "--resume",
        "--extend",
        *ar_args,
    )
    expected = trained.rollout(
        co2=co2, months=24, members=2, seed=41, stochastic_start="1966-01"
    )
    with open_rollout(run / "ar_rollout") as archived:
        xr.testing.assert_identical(archived.compute(), expected.astype("float32"))
    assert first_ar.stat().st_mtime_ns == ar_written

    # The main-path default starts in January 1850 with a prescribed CO2 prefix.
    default_output = run / "default_start"
    run_script(
        monkeypatch,
        "generate_ar_rollout",
        run,
        "--output",
        default_output,
        "--months",
        "1",
        "--members",
        "1",
        "--seed",
        "41",
        *common,
    )
    expected_default = trained.rollout(
        co2=float(co2[0]), months=1, members=1, seed=41, stochastic_start="1850-01"
    )
    with open_rollout(default_output) as archived:
        xr.testing.assert_identical(
            archived.compute(), expected_default.astype("float32")
        )

    # A short but complete source supports the same analysis as a 251-year run.
    monkeypatch.setattr(
        sources, "open_canonical_state", lambda *args, **kwargs: (a, o, statics)
    )
    analysis_args = (
        "--baseline",
        "1966",
        "1967",
        "--early",
        "1966",
        "1966",
        "--late",
        "1967",
        "1967",
        "--no-plots",
    )
    for kind, name in (
        ("ae", "analyze_ae_reconstruction"),
        ("ar", "analyze_ar_rollout"),
    ):
        run_script(monkeypatch, name, run, *analysis_args)
        with xr.open_dataset(run / f"analysis/{kind}/sst_maps.nc") as maps:
            assert "response_difference" in maps
            assert maps.climatological_bias.isel(lon=0).isnull().all()
        with xr.open_dataset(run / f"analysis/{kind}/t2m_series.nc") as series:
            assert abs(float(series.reference_global_anomaly.mean("time"))) < 1e-4
            assert ("global_rmse" in series) == (kind == "ae")


def test_scalar_forcing_rejects_a_prefix_before_opening_files(tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as stopped:
        run_script(
            monkeypatch,
            "generate_ar_rollout",
            tmp_path / "absent",
            "--constant-co2",
            "300",
            "--hold-first-co2-back-to",
            "1849-11",
            "--threads",
            "1",
        )
    assert stopped.value.code == 2
