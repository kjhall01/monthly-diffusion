from pathlib import Path
from time import perf_counter
import hashlib
import json
import sys

import torch
import yaml

PRODUCTION = Path("/Users/kyle/Dev/umd/frozen")
CHECKPOINT = PRODUCTION / "runs/mdv1-e3smv2-r1i1p1f1/model.pt"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output/umd-production"
BATCH_SIZE = 24
PERIOD = slice("1850-01", "2100-12")

# This checkpoint uses the frozen production package, including its preprocessor.
# The installed package and the current model's scripts are left as they are.
sys.path.insert(0, str(PRODUCTION))
from monthly_diffusion import MonthlyDiffusionV1
from monthly_diffusion.data.sources import open_canonical_state, open_co2

torch.set_num_threads(4)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = MonthlyDiffusionV1.load(CHECKPOINT, device=device)
model.autoencoder.eval().requires_grad_(False)

sources = yaml.safe_load((CHECKPOINT.parent / "data_sources.yaml").read_text())
atmosphere, ocean, statics = open_canonical_state(
    sources["state_zarr"], expected_attrs=sources["expected_attrs"],
)
co2 = open_co2(sources["co2_netcdf"], expected_sha256=sources["co2_sha256"])
atmosphere = atmosphere.sel(time=PERIOD)
ocean = ocean.sel(time=PERIOD)

for component in model.names:
    (OUTPUT_DIR / component / "ae_trajectory").mkdir(parents=True, exist_ok=True)

metadata = {
    "status": "running",
    "definition": "Encode each reference month, then decode with the same month and prescribed CO2",
    "checkpoint": str(CHECKPOINT),
    "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
    "production_source": str(PRODUCTION),
    "selected_autoencoder": model.stages["autoencoder"],
    "sources": sources,
    "period": [PERIOD.start, PERIOD.stop],
    "months": atmosphere.sizes["time"],
    "variables": model.names,
    "config": model.config.to_dict(),
    "device": device,
    "output_dtype": "float32",
    "compression": "lossless DEFLATE level 1 with byte shuffle",
    "latent_coordinates": "fixed_transition_normalized",
    "ocean_conventions": "sivol = SIC * SIT (zero on wet ice-free cells); ohc700_minus_ohc300 = ohc300_700",
}
(OUTPUT_DIR / "trajectory_metadata.json").write_text(json.dumps(metadata, indent=2))
log_file = (OUTPUT_DIR / "generation.log").open("w", encoding="utf-8", buffering=1)
print(f"Production AE epoch {model.stages['autoencoder']['epoch']} on {device}", flush=True)
print(f"Saving trajectory to {OUTPUT_DIR}", flush=True)
started = perf_counter()

with torch.no_grad():
    for start in range(0, atmosphere.sizes["time"], BATCH_SIZE):
        batch = slice(start, start + BATCH_SIZE)
        atmos_batch = atmosphere.isel(time=batch).compute()
        ocean_batch = ocean.isel(time=batch).compute()

        # Public APIs retain the production transforms, masks and conditioning.
        za, zo = model.encode(atmos_batch, ocean_batch, co2=co2)
        reconstructed_atmos, reconstructed_ocean = model.decode(za, zo, co2=co2)

        for component, reconstruction, latent in [
            ("atmosphere", reconstructed_atmos, za),
            ("ocean", reconstructed_ocean, zo),
        ]:
            output = OUTPUT_DIR / component / "ae_trajectory"
            reconstruction = reconstruction.astype("float32")
            reconstruction.attrs.update(
                checkpoint_sha256=metadata["checkpoint_sha256"],
                autoencoder_epoch=model.stages["autoencoder"]["epoch"],
                definition=metadata["definition"],
            )
            encoding = {name: {"dtype": "float32", "zlib": True, "complevel": 1, "shuffle": True}
                        for name in reconstruction.data_vars}
            target = output / f"reconstruction_{start:05d}.nc"
            temporary = target.with_suffix(".nc.partial")
            reconstruction.to_netcdf(temporary, engine="netcdf4", encoding=encoding)
            temporary.replace(target)

            latent.attrs["checkpoint_sha256"] = metadata["checkpoint_sha256"]
            target = output / f"latent_{start:05d}.nc"
            temporary = target.with_suffix(".nc.partial")
            latent.to_netcdf(temporary, engine="netcdf4")
            temporary.replace(target)

        line = (f"{str(atmos_batch.time.values[0])[:10]} to {str(atmos_batch.time.values[-1])[:10]} | "
                f"{start + atmos_batch.sizes['time']}/{atmosphere.sizes['time']} months | "
                f"{perf_counter() - started:.1f}s elapsed")
        print(line, flush=True)
        log_file.write(line + "\n")

metadata.update(status="complete", elapsed_seconds=perf_counter() - started)
(OUTPUT_DIR / "trajectory_metadata.json").write_text(json.dumps(metadata, indent=2))
log_file.close()
atmosphere.close()
ocean.close()
print("Production autoencoder trajectory complete.", flush=True)
