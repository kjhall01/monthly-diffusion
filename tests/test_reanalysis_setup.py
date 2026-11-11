from pathlib import Path
import yaml
from monthly_diffusion import ModelConfig

ROOT = Path(__file__).resolve().parents[1]


def test_reanalysis_config_matches_mask_aware_e3sm():
    era = ModelConfig.from_yaml(ROOT/"examples/model-era5-oras5.yaml")
    e3sm = ModelConfig.from_yaml(ROOT/"examples/model-ocean-mask-aware.yaml")
    assert era.to_dict() == e3sm.to_dict()
    assert era.ocean_mask_aware
    assert not era.co2_film_levels  # unchanged: all levels, not finest only
    assert (era.atmosphere_tau, era.ocean_tau) == (.5, 6.)
    assert era.autoencoder_training.seed == era.diffusion_training.seed == 20260716


def test_reanalysis_source_uses_same_forcing_not_same_physical_data():
    era = yaml.safe_load((ROOT/"examples/era5-oras5-local.yaml").read_text())
    e3sm = yaml.safe_load((ROOT/"examples/e3sm-local.yaml").read_text())
    assert era["state_zarr"] != e3sm["state_zarr"]
    assert era["co2_netcdf"] == e3sm["co2_netcdf"]
    assert era["co2_sha256"] == e3sm["co2_sha256"]
    assert era["expected_attrs"]["member_id"] == "era5_oras5"
    assert era["expected_attrs"]["product_release_id"] == "monthly-diffusion-data-era5-oras5-1958-2025-v1"


def test_launcher_uses_shared_training_entrypoint_and_separate_run():
    script = (ROOT/"scripts/recipes/train_era5_oras5.sh").read_text()
    assert "scripts/legacy/train_model.py examples/era5-oras5-local.yaml" in script
    assert "--config examples/model-era5-oras5.yaml" in script
    assert "--output runs/mdv1-era5-oras5" in script
    assert "--device mps --threads 4 --load-in-memory" in script
    assert '"$@"' in script  # resume/explicit user overrides forwarded
