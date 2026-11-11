"""The editable scripts own training; a run directory carries all later stages."""

import ast
from pathlib import Path
import numpy as np
import pytest
import torch
import xarray as xr
from monthly_diffusion import (
    load_autoencoder,
    load_diffusion,
    load_models,
    MonthlyDiffusionV1,
    open_rollout,
)
from monthly_diffusion.data import sources
from monthly_diffusion.run_files import file_digest, load_latents
from monthly_diffusion.inference.reconstruction import open_reconstruction
from monthly_diffusion.modules.autoencoder import MeshHierarchyAutoencoder
from monthly_diffusion.components.coupled_latent_diffusion import CoupledLatentDiffusion


@pytest.fixture
def experiment(tiny_data, tmp_path, monkeypatch):
    a, o, statics, _, config = tiny_data
    val_time = xr.date_range("1958-01-01", periods=24, freq="MS")
    all_a = xr.concat([a.assign_coords(time=val_time), a], "time")
    all_o = xr.concat([o.assign_coords(time=val_time), o], "time")
    time = xr.date_range("1958-01-01", "1970-12-01", freq="MS")
    co2 = xr.DataArray(
        np.linspace(310, 350, len(time)), dims="time", coords={"time": time}
    )
    monkeypatch.setattr(
        sources, "open_canonical_state", lambda *args, **kwargs: (all_a, all_o, statics)
    )
    monkeypatch.setattr(sources, "open_co2", lambda *args, **kwargs: co2)
    params = dict(
        RUN_DIR=tmp_path / "run",
        DEVICE="cpu",
        THREADS=1,
        WIDTH=8,
        MESH_LEVELS=(1, 0),
        REFINERS=(1, 1),
        EPOCHS=1,
    )
    return a, o, co2, config, params


def unexpected_network(*args, **kwargs):
    raise AssertionError("This script must not construct this network")


def test_seven_scripts_and_component_loading(experiment, run_script, monkeypatch):
    a, o, co2, config, params = experiment
    run = params["RUN_DIR"]
    with monkeypatch.context() as isolated:
        isolated.setattr(CoupledLatentDiffusion, "__init__", unexpected_network)
        run_script(
            "train_atmosphere_autoencoder",
            **params,
            SPECIFICATIONS=[config.variables[0]],
        )
    assert not (run / "ocean").exists() and not (run / "diffusion").exists()
    network, info = load_autoencoder(run, "atmosphere", device="cpu")
    assert isinstance(network, MeshHierarchyAutoencoder)
    assert [v.name for v in info["preprocessor"].variables] == ["t2m"]
    assert not hasattr(MonthlyDiffusionV1, "train")
    before = file_digest(run / "atmosphere/model.pt")
    # Different component widths/channels are independent saved constructor values.
    run_script(
        "train_ocean_autoencoder",
        **{**params, "WIDTH": 12, "LATENT_CHANNELS": 6},
        SPECIFICATIONS=[config.variables[1]],
    )
    assert file_digest(run / "atmosphere/model.pt") == before
    both = load_models(run, device="cpu", diffusion=False)
    assert both.transition is None
    run_script(
        "generate_ae_reconstruction",
        RUN_DIR=run,
        DEVICE="cpu",
        THREADS=1,
        START="1966-01",
        END="1967-12",
    )
    latents = both.encode(a, o, co2=co2)
    expected = xr.merge(both.decode(*latents, co2=co2))
    with open_reconstruction(run / "ae_reconstruction") as saved:
        xr.testing.assert_allclose(
            saved[["t2m", "sst"]].drop_vars("co2").compute(), expected
        )
        assert saved.atmosphere_latent.sizes["atmosphere_channel"] == 4
        assert saved.ocean_latent.sizes["ocean_channel"] == 6
        xr.testing.assert_allclose(
            saved.ocean_latent.drop_vars("co2")
            .rename({"ocean_channel": "channel"})
            .compute(),
            latents[1].rename("ocean_latent"),
        )
    first = run / "ae_reconstruction/year-1966.zarr/.zmetadata"
    modified = first.stat().st_mtime_ns
    fingerprints = {
        str(p): file_digest(p)
        for c in ("atmosphere", "ocean")
        for p in (run / c).glob("*.pt")
    }
    with monkeypatch.context() as isolated:
        isolated.setattr(MeshHierarchyAutoencoder, "__init__", unexpected_network)
        run_script(
            "train_latent_diffusion",
            RUN_DIR=run,
            DEVICE="cpu",
            THREADS=1,
            EPOCHS=1,
            WIDTH=8,
            DEPTH=1,
            HEADS=2,
            TIME_EMBEDDING_CHANNELS=8,
        )
    assert all(file_digest(p) == digest for p, digest in fingerprints.items())
    transition, _ = load_diffusion(run, device="cpu")
    assert transition.config.ocean_latent_channels == 6
    trained = load_models(run, device="cpu")
    # Adding diffusion does not invalidate an existing reconstruction archive.
    run_script(
        "generate_ae_reconstruction",
        RUN_DIR=run,
        DEVICE="cpu",
        THREADS=1,
        START="1966-01",
        END="1967-12",
        RESUME=True,
    )
    assert first.stat().st_mtime_ns == modified
    trained.save(run / "export.pt")
    restored = MonthlyDiffusionV1.load(run / "export.pt", device="cpu")
    assert restored.weight_digest() == trained.weight_digest()
    for actual, expected_latent in zip(restored.encode(a, o, co2=co2), latents):
        xr.testing.assert_identical(actual, expected_latent)

    for initialization in ("stochastic", "gaussian", "encoded_noise", "observed"):
        output = run / initialization
        start = "1966-03" if initialization == "observed" else "1966-01"
        options = dict(
            RUN_DIR=run,
            OUTPUT=output,
            DEVICE="cpu",
            THREADS=1,
            MONTHS=2,
            MEMBERS=2,
            START=start,
            INITIALIZATION=initialization,
            SEED=41,
        )
        run_script("generate_ar_rollout", **options)
        first_ar = output / "month-000001.zarr/.zmetadata"
        wrote = first_ar.stat().st_mtime_ns
        run_script(
            "generate_ar_rollout",
            **{**options, "MONTHS": 3, "RESUME": True, "EXTEND": True},
        )
        states = (
            (a.isel(time=slice(2)), o.isel(time=slice(2)))
            if initialization == "observed"
            else ()
        )
        kwargs = (
            {}
            if initialization == "observed"
            else {initialization + "_start": "1966-01"}
        )
        expected_ar = trained.rollout(
            *states, co2=co2, months=3, members=2, seed=41, **kwargs
        )
        with open_rollout(output) as saved:
            xr.testing.assert_identical(saved.compute(), expected_ar.astype("float32"))
        assert first_ar.stat().st_mtime_ns == wrote
    # Two full years for the main AR climate analysis.
    run_script(
        "generate_ar_rollout",
        RUN_DIR=run,
        DEVICE="cpu",
        THREADS=1,
        MONTHS=24,
        MEMBERS=2,
        START="1966-01",
    )
    monkeypatch.setattr(
        sources, "open_canonical_state", lambda *args, **kwargs: (a, o, info["statics"])
    )
    for name, kind in (
        ("analyze_ae_reconstruction", "ae"),
        ("analyze_ar_rollout", "ar"),
    ):
        run_script(
            name,
            RUN_DIR=run,
            BASELINE=(1966, 1967),
            EARLY=(1966, 1966),
            LATE=(1967, 1967),
            MAKE_PLOTS=False,
        )
        with xr.open_dataset(run / f"analysis/{kind}/sst_maps.nc") as maps:
            assert maps.climatological_bias.isel(lon=0).isnull().all()
            assert "response_difference" in maps


def test_component_resume_preserves_optimizer_ema_rng_and_cache(
    experiment, run_script, monkeypatch
):
    from monthly_diffusion import checkpoints

    a, o, co2, config, params = experiment
    complete, resumed = (
        params["RUN_DIR"].parent / "complete",
        params["RUN_DIR"].parent / "resumed",
    )
    overrides = {**params, "EPOCHS": 2, "SPECIFICATIONS": [config.variables[0]]}
    run_script("train_atmosphere_autoencoder", **{**overrides, "RUN_DIR": complete})
    original = checkpoints.atomic_save

    def interrupt_after_epoch(value, path):
        original(value, path)
        if Path(path).name == "latest.pt" and value.get("epoch") == 1:
            raise RuntimeError("simulated interruption")

    with monkeypatch.context() as interrupted:
        interrupted.setattr(checkpoints, "atomic_save", interrupt_after_epoch)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            run_script(
                "train_atmosphere_autoencoder", **{**overrides, "RUN_DIR": resumed}
            )
    run_script(
        "train_atmosphere_autoencoder",
        **{**overrides, "RUN_DIR": resumed, "RESUME": True},
    )
    whole, _ = load_autoencoder(complete, "atmosphere", device="cpu")
    resumed_ae, _ = load_autoencoder(resumed, "atmosphere", device="cpu")
    for name, value in whole.state_dict().items():
        torch.testing.assert_close(value, resumed_ae.state_dict()[name], rtol=0, atol=0)
    for split in ("training", "validation"):
        torch.testing.assert_close(
            load_latents(complete, "atmosphere")[split],
            load_latents(resumed, "atmosphere")[split],
            rtol=0,
            atol=0,
        )
    fingerprint = file_digest(resumed / "atmosphere/model.pt")
    run_script(
        "train_atmosphere_autoencoder",
        **{**overrides, "RUN_DIR": resumed, "RESUME": True},
    )
    assert fingerprint == file_digest(resumed / "atmosphere/model.pt")
    with pytest.raises(ValueError, match="settings changed"):
        run_script(
            "train_atmosphere_autoencoder",
            **{**overrides, "RUN_DIR": resumed, "WIDTH": 12, "RESUME": True},
        )


def test_main_scripts_show_training_and_have_no_cli_or_config_dispatch():
    root = Path(__file__).resolve().parents[1] / "scripts"
    for path in root.glob("*.py"):
        source = path.read_text()
        tree = ast.parse(source)
        assert not any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for n in ast.walk(tree)
        ), path
        for removed in (
            "argparse",
            "ModelConfig",
            "TrainingConfig",
            "train_autoencoder(",
            "train_coupled_diffusion(",
        ):
            assert removed not in source, (path, removed)
        if path.name.startswith("train_"):
            assert "torch.optim.AdamW(" in source
            assert "loss.backward()" in source or 'terms["loss"].backward()' in source
            assert "for epoch in range(" in source


def test_diffusion_resume_and_changed_ae_detection(experiment, run_script, monkeypatch):
    import shutil
    from monthly_diffusion import checkpoints

    a, o, co2, config, params = experiment
    whole = params["RUN_DIR"]
    resumed = whole.parent / "resumed_diffusion"
    run_script(
        "train_atmosphere_autoencoder", **params, SPECIFICATIONS=[config.variables[0]]
    )
    run_script(
        "train_ocean_autoencoder", **params, SPECIFICATIONS=[config.variables[1]]
    )
    shutil.copytree(whole, resumed)
    diffusion = dict(
        DEVICE="cpu",
        THREADS=1,
        EPOCHS=2,
        WIDTH=8,
        DEPTH=1,
        HEADS=2,
        TIME_EMBEDDING_CHANNELS=8,
    )
    run_script("train_latent_diffusion", RUN_DIR=whole, **diffusion)
    save = checkpoints.atomic_save

    def interrupt(value, path):
        save(value, path)
        if Path(path).name == "latest.pt" and value.get("epoch") == 1:
            raise RuntimeError("interrupted diffusion")

    with monkeypatch.context() as interrupted:
        interrupted.setattr(checkpoints, "atomic_save", interrupt)
        with pytest.raises(RuntimeError, match="interrupted diffusion"):
            run_script("train_latent_diffusion", RUN_DIR=resumed, **diffusion)
    run_script("train_latent_diffusion", RUN_DIR=resumed, RESUME=True, **diffusion)
    expected, _ = load_diffusion(whole, device="cpu")
    actual, _ = load_diffusion(resumed, device="cpu")
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(value, actual.state_dict()[name], rtol=0, atol=0)
    checkpoint = resumed / "ocean/model.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["model"][next(iter(payload["model"]))] += 1
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="different ocean model.pt"):
        load_diffusion(resumed, device="cpu")
    with pytest.raises(ValueError, match="different AE"):
        load_latents(resumed, "ocean")
