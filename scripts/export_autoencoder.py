"""Export the best completed AE epoch after stopping training.

Edit RUN_DIR and COMPONENT, then run this file. No optimizer steps are performed.
Existing selected weights and validated latent caches are reused.
"""

import sys
from pathlib import Path

import dask
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from monthly_diffusion.checkpoints import load_checkpoint, load_latents
from monthly_diffusion.loading import build_autoencoder
from monthly_diffusion.rollout import co2_values, month_features
from workflows.run_files import check_script_arguments, run_inputs
from workflows.data.sources import open_state_dataset, static_fields, open_co2
from workflows.training.checks import check_component_data
from workflows.training.files import select_best_weights, save_latent_cache

check_script_arguments()
RUN_DIR = Path(__file__).resolve().parent / "runs/test_gridpoint_log1p_mtpr_masked"
COMPONENT = "ocean"  # also works for "atmosphere"
DEVICE = "cpu"
THREADS = 2
BATCH_SIZE = 4

torch.set_num_threads(THREADS)
directory = RUN_DIR / COMPONENT
setup = load_checkpoint(directory / "setup.pt")
preprocessor = setup["preprocessor"]
names = [variable.name for variable in preprocessor.variables]
DATA, CO2, TRAIN_PERIOD, VALIDATION_PERIOD = run_inputs(RUN_DIR)
dataset = open_state_dataset(DATA)
fields = dataset[names]
training = fields.sel(time=TRAIN_PERIOD)
validation = fields.sel(time=VALIDATION_PERIOD)
statics = static_fields(dataset)
co2 = open_co2(CO2)
check_component_data(training, validation, statics, preprocessor.variables, co2, setup)
model = build_autoencoder(
    setup["parameters"],
    setup["statics"],
    len(names),
    torch.device(DEVICE),
    static_fields=preprocessor.static_tensor(setup["statics"]),
    preprocessor=preprocessor,
)
epoch = select_best_weights(directory, model)
print(f"Selected {COMPONENT} epoch {epoch}; no further training.", flush=True)

if (directory / "latents.pt").exists():
    model.latent_normalizer = load_latents(RUN_DIR, COMPONENT)["normalizer"]
else:
    encoded_splits = []
    with torch.no_grad():
        for label, fields in (("training", training), ("validation", validation)):
            months = month_features(fields.time).to(DEVICE)
            forcing = torch.as_tensor(
                (
                    (co2_values(co2, fields.time) - preprocessor.co2_mean)
                    / preprocessor.co2_std
                ).values,
                dtype=torch.float32,
                device=DEVICE,
            )[:, None]
            encoded_batches = []
            years = fields.time.dt.year.values
            # Load a year once, keeping memory bounded and avoiding repeated
            # reads of the same native storage chunks for individual months.
            for year in np.unique(years):
                indices = np.flatnonzero(years == year)
                with dask.config.set(scheduler="threads", num_workers=THREADS):
                    normalized = preprocessor.encode(
                        fields.isel(time=indices), COMPONENT
                    ).compute()
                state = torch.as_tensor(
                    np.stack(
                        [
                            normalized[name].transpose("time", "lat", "lon").values
                            for name in names
                        ],
                        axis=1,
                    ),
                    dtype=torch.float32,
                )
                for start in range(0, len(indices), BATCH_SIZE):
                    selected = indices[start : start + BATCH_SIZE]
                    latent = model.encode(
                        state[start : start + BATCH_SIZE].to(DEVICE),
                        months[selected],
                        forcing[selected],
                    )
                    encoded_batches.append(latent.cpu())
                if year % 10 == 0 or year == years[-1]:
                    print(f"Encoded {label} through {year}", flush=True)
            encoded_splits.append(torch.cat(encoded_batches))
    training_latents, validation_latents = encoded_splits
    model.latent_normalizer.fit(training_latents)
    save_latent_cache(
        directory,
        training_latents,
        validation_latents,
        model.latent_normalizer,
        training.time,
        validation.time,
    )
if not (directory / "autoencoder.pt").exists():
    model.save(directory / "autoencoder.pt")
print(
    f"Ready: selected model, latent cache, and portable AE in {directory}", flush=True
)
