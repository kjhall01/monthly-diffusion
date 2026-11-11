import numpy as np
import pytest
import torch
import xarray as xr
from monthly_diffusion import ModelConfig, TrainingConfig, Variable

torch.set_num_threads(1)


@pytest.fixture
def tiny_data():
    lat = np.arange(-85, 90, 10.0)
    lon = np.arange(5, 360, 10.0)
    time = xr.date_range("1966-01-01", periods=24, freq="MS")
    coords = dict(time=time, lat=lat, lon=lon)
    rng = np.random.default_rng(4)
    shape = (24, len(lat), len(lon))
    a = xr.Dataset(
        {
            "t2m": (
                ("time", "lat", "lon"),
                (280 + rng.normal(size=shape)).astype("float32"),
            )
        },
        coords=coords,
    )
    o = xr.Dataset(
        {
            "sst": (
                ("time", "lat", "lon"),
                (20 + rng.normal(size=shape)).astype("float32"),
            )
        },
        coords=coords,
    )
    a.t2m.attrs["units"] = "K"
    o.sst.attrs["units"] = "degrees C"
    o.sst.values[:, :, 0] = np.nan
    static = xr.Dataset(
        {
            "phi_s": (("lat", "lon"), rng.normal(size=shape[1:]).astype("float32")),
            "lsm": (("lat", "lon"), np.zeros(shape[1:], dtype="float32")),
        },
        coords=dict(lat=lat, lon=lon),
    )
    co2time = xr.date_range("1966-01-01", periods=50, freq="MS")
    co2 = xr.DataArray(np.linspace(320, 350, 50), dims="time", coords={"time": co2time})
    train = TrainingConfig(batch_size=4, max_epochs=1, patience=5, rollout_steps=3)
    config = ModelConfig(
        variables=[
            Variable("t2m", "atmosphere", "K", weight=0.5),
            Variable("sst", "ocean", "degrees C", weight=0.5),
        ],
        ae_width=8,
        mesh_levels=(1, 0),
        refiners=(1, 1),
        transition_width=8,
        transition_depth=1,
        transition_heads=2,
        time_embedding_channels=8,
        autoencoder_training=train,
        diffusion_training=train,
    )
    return a, o, static, co2, config


@pytest.fixture
def run_script():
    """Run an actual script after editing only its top-level parameter assignments.

    Tests do this in memory; users edit the same assignments in their checkout.
    There is no test-only configuration hook in the production scripts.
    """
    import ast
    from pathlib import Path

    def execute(name, **values):
        path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
        tree = ast.parse(path.read_text())
        used = set()
        body = []
        for statement in tree.body:
            names = []
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        names.append(target.id)
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        names += [
                            item.id
                            for item in target.elts
                            if isinstance(item, ast.Name)
                        ]
            if len(names) == 1 and names[0] in values:
                statement.value = ast.Subscript(
                    value=ast.Name(id="test_values", ctx=ast.Load()),
                    slice=ast.Constant(names[0]),
                    ctx=ast.Load(),
                )
                used.add(names[0])
            body.append(statement)
            if len(names) > 1:
                for name in names:
                    if name in values:
                        body += ast.parse(f"{name} = test_values[{name!r}]").body
                        used.add(name)
        assert used == set(values), f"Unknown editable parameters: {set(values) - used}"
        tree.body = body
        ast.fix_missing_locations(tree)
        namespace = dict(__name__="__main__", __file__=str(path), test_values=values)
        import sys

        previous = sys.argv
        try:
            sys.argv = [str(path)]
            exec(compile(tree, str(path), "exec"), namespace)
        finally:
            sys.argv = previous
        return namespace

    return execute


@pytest.fixture
def train_small_run(monkeypatch, run_script):
    from monthly_diffusion.data import sources
    from monthly_diffusion import load_models

    def train(data, directory, *, device="cpu"):
        a, o, statics, co2, config = data
        monkeypatch.setattr(
            sources, "open_canonical_state", lambda *args, **kwargs: (a, o, statics)
        )
        monkeypatch.setattr(sources, "open_co2", lambda *args, **kwargs: co2)
        common = dict(
            RUN_DIR=directory,
            DEVICE=device,
            THREADS=1,
            EPOCHS=config.autoencoder_training.max_epochs,
            WIDTH=config.ae_width,
            LATENT_CHANNELS=config.latent_channels,
            MESH_LEVELS=config.mesh_levels,
            REFINERS=config.refiners,
            ROTATION=config.rotation,
            CO2_FILM_LEVELS=config.co2_film_levels,
            OCEAN_SUPPORT=config.ocean_support,
        )
        run_script(
            "train_atmosphere_autoencoder",
            **common,
            SPECIFICATIONS=[v for v in config.variables if v.component == "atmosphere"],
            TRAIN_PERIOD=slice(str(a.time.values[0])[:7], str(a.time.values[11])[:7]),
            VALIDATION_PERIOD=slice(
                str(a.time.values[12])[:7], str(a.time.values[-1])[:7]
            ),
        )
        run_script(
            "train_ocean_autoencoder",
            **common,
            MASK_AWARE=config.ocean_mask_aware,
            SPECIFICATIONS=[v for v in config.variables if v.component == "ocean"],
        )
        run_script(
            "train_latent_diffusion",
            RUN_DIR=directory,
            DEVICE=device,
            THREADS=1,
            EPOCHS=config.diffusion_training.max_epochs,
            WIDTH=config.transition_width,
            DEPTH=config.transition_depth,
            HEADS=config.transition_heads,
            TIME_EMBEDDING_CHANNELS=config.time_embedding_channels,
        )
        return load_models(directory, device=device)

    return train
