"""Physical metrics and portable models, exercised through the editable scripts."""

import csv
import shutil

import numpy as np
import pytest
import torch
import xarray as xr

from monthly_diffusion import Autoencoder, Preprocessor, Variable, load_autoencoder
from monthly_diffusion.checkpoints import load_latents
from monthly_diffusion.preprocessing import select_variables
from monthly_diffusion.rollout import co2_values
from workflows.data.dataloader import ComponentStates
from workflows.data.sources import open_state_dataset
from workflows.run_files import save_history
from workflows.training.losses import reconstruction_mse


def test_reconstruction_normalizes_latitude_weights_for_each_variables_mask():
    # The second variable has only one observed cell. Its contribution must not
    # shrink with its support area, and masked predictions must not affect loss.
    target = torch.tensor(
        [
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[float("nan"), float("nan")], [0.0, float("nan")]],
            ]
        ]
    )
    prediction = torch.tensor(
        [[[[1.0, 1.0], [3.0, 3.0]], [[100.0, 100.0], [3.0, 100.0]]]], requires_grad=True
    )
    per_variable = reconstruction_mse(prediction, target, torch.tensor([0.0, 60.0]))
    torch.testing.assert_close(per_variable, torch.tensor([11 / 3, 9.0]))
    loss = per_variable.mean()
    assert float(loss.detach()) == pytest.approx(19 / 3)
    loss.backward()
    expected_gradient = torch.tensor(
        [[[[1 / 3, 1 / 3], [0.5, 0.5]], [[0.0, 0.0], [3.0, 0.0]]]]
    )
    torch.testing.assert_close(prediction.grad, expected_gradient)


@pytest.mark.parametrize("centering", ["annual_zonal", "gridpoint"])
def test_tensor_inverse_and_original_physical_targets(tiny_data, tmp_path, centering):
    _, ocean, statics, co2, _ = tiny_data
    fields = xr.Dataset(
        {
            "sst": ocean.sst,
            "sic": (ocean.sst / 25).clip(0, 1),
            "mld": ocean.sst * 2,
        }
    )
    fields.sic.values[:, :, 1] = 0
    fields.sic.values[:, :, 2] = 1
    fields.mld.values[:, :, 1] = 0
    variables = select_variables(["sst", "sic", "mld"], "ocean")
    preprocessor = Preprocessor(variables, centering=centering, ocean_support="sst")
    preprocessor.fit_component(fields, "ocean", statics, co2_values(co2, fields.time))
    encoded = preprocessor.encode(fields, "ocean")
    values = torch.tensor(np.stack([encoded[v.name].values for v in variables], axis=1))
    decoded = preprocessor.decode_tensor(values, "ocean")
    expected = preprocessor.decode(encoded, "ocean")
    np.testing.assert_allclose(
        decoded.numpy(),
        np.stack([expected[v.name].values for v in variables], axis=1),
        atol=1e-5,
        rtol=1e-5,
        equal_nan=True,
    )
    state, physical, _, _ = ComponentStates(fields, "ocean", preprocessor, co2)[0]
    assert physical[1, 0, 1] == 0 and physical[1, 0, 2] == 1
    assert physical[2, 0, 1] == 0
    # Even a perfect normalized prediction differs from the clipped raw endpoints.
    mse = reconstruction_mse(
        preprocessor.decode_tensor(state[None], "ocean"),
        physical[None],
        torch.tensor(statics.lat.values),
    )
    assert mse[1] > 0 and mse[2] > 0
    preprocessor.save(tmp_path / "preprocessing.pt")
    restored = Preprocessor.load(tmp_path / "preprocessing.pt")
    torch.testing.assert_close(
        restored.decode_tensor(values, "ocean"), decoded, equal_nan=True
    )


@pytest.mark.parametrize("centering", ["annual_zonal", "gridpoint"])
@pytest.mark.parametrize(
    "fields", ["temperature", "radiation_and_ice", "precipitation_and_constants"]
)
def test_scripts_log_physical_rmse_and_export_owned_state(
    tiny_data, tmp_path, run_script, centering, fields
):
    atmosphere, ocean, statics, co2, _ = tiny_data
    names = ("t2m", "sst")
    if fields == "radiation_and_ice":
        # These names normally select inverse-softplus/logit. The scripts must
        # override both, including zeros and the exact concentration endpoints.
        atmosphere = xr.Dataset({"ssrd": ((atmosphere.t2m - 280) * 100).clip(min=0)})
        ocean = xr.Dataset({"sic": ((ocean.sst - 20) / 2 + 0.5).clip(0, 1)})
        atmosphere.ssrd.attrs["units"] = "W m^-2"
        ocean.sic.attrs["units"] = "dimensionless"
        names = ("ssrd", "sic")
    if fields == "precipitation_and_constants":
        atmosphere = xr.Dataset({"mtpr": (atmosphere.t2m - 280).clip(min=0) / 86400})
        ocean = xr.Dataset({"sic": ((ocean.sst - 20) / 2 + 0.5).clip(0, 1)})
        ocean.sic.values[:, :, 1] = 0  # exclude from learning; restore zeros on decode
        ocean.sic.values[:, :, 2] = 1  # user-requested exception: still a target
        atmosphere.mtpr.attrs["units"] = "kg m^-2 s^-1"
        ocean.sic.attrs["units"] = "dimensionless"
        names = ("mtpr", "sic")
    source = tmp_path / "fields.nc"
    xr.merge([atmosphere, ocean, statics]).to_netcdf(source)
    co2_path = tmp_path / "co2.nc"
    co2.rename("co2").to_dataset().to_netcdf(co2_path)
    run = tmp_path / "run"
    parameters = dict(
        RUN_DIR=run,
        DEVICE="cpu",
        THREADS=1,
        WIDTH=8,
        LATENT_CHANNELS=4,
        MESH_LEVELS=(1, 0),
        REFINERS=(1, 1),
        EPOCHS=1,
        BATCH_SIZE=5,
        CENTERING=centering,
        MASK_CONSTANT_CELLS=centering == "gridpoint",
    )
    for component, name in zip(("atmosphere", "ocean"), names):
        if component == "atmosphere":
            inputs = dict(
                DATA=source,
                CO2=co2_path,
                TRAIN_PERIOD=slice("1966-01", "1966-12"),
                VALIDATION_PERIOD=slice("1967-01", "1967-12"),
            )
        else:
            inputs = {}
        # Select from VARIABLES, without supplying a hidden configuration/spec list.
        result = run_script(
            f"train_{component}_autoencoder", **parameters, **inputs, VARIABLES=[name]
        )
        directory = run / component
        network = load_autoencoder(run, component, device="cpu")
        assert network.encoder.mask_policy == network.decoder.mask_policy == "support"
        assert network.latent_normalizer.fitted
        assert network.preprocessor.centering == centering
        spec = network.preprocessor.variables[0]
        assert spec.transform == ("log1p" if name == "mtpr" else "identity")
        if name == "mtpr":
            assert spec.scale == 1 / 86400
        if centering == "gridpoint":
            stats = network.preprocessor.stats[name]
            assert stats["std"].dims == ("lat", "lon")
            raw = result["training_fields"][name].astype("float64")
            if name == "mtpr":
                raw = np.log1p(raw * 86400)
            xr.testing.assert_allclose(
                stats["center"].where(stats["mask"]),
                raw.mean("time").where(stats["mask"]),
            )
            xr.testing.assert_allclose(
                stats["std"].where(stats["mask"]),
                raw.fillna(0)
                .std("time")
                .clip(min=stats["std_floor"])
                .where(stats["mask"]),
            )
        assert [v.name for v in network.preprocessor.variables] == [name]
        validation = result["validation_fields"]
        xr.testing.assert_allclose(
            network.preprocessor.decode(
                network.preprocessor.encode(validation, component), component
            ),
            validation,
            atol=1e-5,
        )
        predictions = network.reconstruct(validation, co2=co2, batch_size=5)
        if (
            fields == "precipitation_and_constants"
            and component == "ocean"
            and centering == "gridpoint"
        ):
            assert not bool(stats["mask"][:, 1].any())
            assert bool(stats["mask"][:, 2].all())
            assert bool((predictions.sic[:, :, 1] == 0).all())
        # Independent area-weighted calculation over each month's valid cells.
        error = predictions[name] - validation[name]
        area = np.cos(np.deg2rad(validation.lat))
        expected_rmse = float(
            np.sqrt(
                (
                    (error**2 * area).sum(("lat", "lon"))
                    / area.broadcast_like(error)
                    .where(error.notnull())
                    .sum(("lat", "lon"))
                ).mean()
            )
        )
        with (directory / "history.csv").open() as stream:
            row = list(csv.DictReader(stream))[-1]
        assert float(row[f"val_rmse_{name}"]) == pytest.approx(expected_rmse, rel=2e-6)
        for split in ("train", "val"):
            assert np.isfinite(float(row[f"{split}_normalized_rmse_{name}"]))
            assert np.isfinite(float(row[f"{split}_rmse_{name}"]))
            # Each test script selects one field whose old preset weight is
            # below one. The scalar loss must now equal its normalized MSE.
            assert float(row[f"{split}_loss"]) == pytest.approx(
                float(row[f"{split}_normalized_rmse_{name}"]) ** 2, rel=2e-6
            )
        log = (directory / "training.log").read_text()
        assert "| batch " not in log and "physical RMSE train" in log
        assert f"{float(row['seconds']):.1f} s/epoch" in log
        assert f"{name} [{validation[name].attrs['units']}]" in log
        # Move only the exported file: setup/cache/run records are not required.
        portable = tmp_path / f"{component}.pt"
        shutil.copyfile(directory / "autoencoder.pt", portable)
        restored = Autoencoder.load(portable, device="cpu")
        xr.testing.assert_identical(
            restored.reconstruct(validation, co2=co2, batch_size=5), predictions
        )
        torch.testing.assert_close(
            restored.latent_normalizer.mean,
            network.latent_normalizer.mean,
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            restored.latent_normalizer.std,
            network.latent_normalizer.std,
            atol=0,
            rtol=0,
        )
        # Simulate a stop before export, then recover without training again.
        cache = load_latents(run, component)
        training_files = {
            filename: (directory / filename).read_bytes()
            for filename in ("best.pt", "latest.pt", "history.csv", "training.log")
        }
        for filename in ("model.pt", "latents.pt", "autoencoder.pt"):
            (directory / filename).unlink()
        run_script(
            "export_autoencoder",
            RUN_DIR=run,
            COMPONENT=component,
            DEVICE="cpu",
            THREADS=1,
            BATCH_SIZE=5,
        )
        recovered = load_autoencoder(run, component, device="cpu")
        xr.testing.assert_allclose(
            recovered.reconstruct(validation, co2=co2, batch_size=5), predictions
        )
        recovered_cache = load_latents(run, component)
        for split in ("training", "validation"):
            torch.testing.assert_close(recovered_cache[split], cache[split])
        for filename, content in training_files.items():
            assert (directory / filename).read_bytes() == content
        if component == "ocean":
            # Old ocean weights must not resume under the corrected mask operator.
            setup_path = directory / "setup.pt"
            original = setup_path.read_bytes()
            saved = torch.load(setup_path, weights_only=False)
            del saved["parameters"]["mask_policy"]
            torch.save(saved, setup_path)
            with pytest.raises(ValueError, match="architecture changed"):
                run_script(
                    "train_ocean_autoencoder",
                    **parameters,
                    VARIABLES=[name],
                    RESUME=True,
                )
            setup_path.write_bytes(original)


def test_subset_reader_and_custom_variables(tmp_path):
    dataset = xr.Dataset({"custom": ("time", [1, 2])}, coords={"time": [0, 1]})
    path = tmp_path / "custom.nc"
    dataset.to_netcdf(path)
    with open_state_dataset(path) as actual:
        xr.testing.assert_equal(actual, dataset)
    custom = Variable("custom", "atmosphere", "K", weight=0.2)
    assert select_variables(["t2m", custom], "atmosphere") == [
        select_variables(["t2m"], "atmosphere")[0],
        custom,
    ]
    with pytest.raises(ValueError, match="duplicate"):
        select_variables(["t2m", "t2m"], "atmosphere")


def test_old_history_accepts_new_metric_columns(tmp_path):
    history = [
        {"epoch": 1, "train_loss": 0.5},
        {"epoch": 2, "train_loss": 0.4, "train_rmse_t2m": 1.2},
    ]
    save_history(history, tmp_path / "history.csv")
    with (tmp_path / "history.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["train_rmse_t2m"] == ""
    assert float(rows[1]["train_rmse_t2m"]) == 1.2
