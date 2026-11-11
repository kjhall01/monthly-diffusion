# Read and change the emulator

Start with the seven ordinary scripts in [scripts/README.md](../scripts/README.md).
Parameters are editable Python values at the top. Training scripts instantiate
actual Torch networks and contain their optimizer, batch/epoch loops, validation,
checkpoint selection and resume logic. There are no package training entry points.

## Reading order

1. **Atmosphere AE:** `scripts/train_atmosphere_autoencoder.py` creates the run,
   opens the shared source, fits one component's preprocessing, constructs one AE,
   trains it, selects its best EMA and writes its normalized latent cache.
2. **Ocean AE:** `scripts/train_ocean_autoencoder.py` reads the run's source and
   split record, then performs the same explicit steps for the ocean. Each script
   is independently readable; neither imports or executes the other script.
3. **Diffusion:** `scripts/train_latent_diffusion.py` loads the two selected latent
   caches, constructs the coupled transition, and trains it. The multi-step loss
   is network math in `components/coupled_latent_diffusion.py`; the training loop
   stays in the script. AEs are not allocated during diffusion training.
4. **Generation:** `scripts/generate_ae_reconstruction.py` loads both AEs and
   reconstructs one reference year at a time. `scripts/generate_ar_rollout.py`
   loads all three models, initializes history, resumes/spins up, then predicts,
   decodes and writes one month at a time.
5. **Analysis:** the two `scripts/analyze_*.py` files select the reference,
   generated fields, periods and variables. They calculate, save and plot the
   small time-series/map products, processing one physical variable at a time.

## Package layout

Paths below are relative to `monthly_diffusion/`.

| Location | Responsibility |
|---|---|
| `model_loading.py` | Load an actual AE or diffusion module, or assemble the physical inference interface from a run |
| `run_files.py` | Shared source/split record, file identities, selected checkpoint files, latent reads, CSV history |
| `model.py` | Physical xarray encode/decode, forecasting interface, combined checkpoint import/export; no training method |
| `modules/autoencoder.py` | Actual AE modules and finite-mask cosine-area reconstruction MSE |
| `modules/` | Mesh geometry, ECC, FiLM, embeddings, attention and DiT layers |
| `components/` | Diffusion objectives, component coupling, correlated noise and reverse-time sampling |
| `data/configs.py` | Physical variable definitions; legacy parameter records retained for checkpoint loading |
| `data/sources.py` | Read canonical sources and prescribed CO2 |
| `data/dataloader.py` | Physical/component batches, latent sequences, month features, boundary checks |
| `data/preprocessing.py` | Physical transforms, training-only statistics, fixed latent scaling |
| `training/ema.py` | Parameter EMA and temporary validation weights |
| `training/checks.py` | Validate component data and interrupted-run settings |
| `training/data.py`, `resume.py`, `latents.py` | Helpers used by the historical paired script |
| `inference/initialization.py` | Observed, Gaussian latent, encoded-noise and local-moment histories |
| `inference/moments.py` | Training-only local monthly field moments |
| `inference/forecast.py` | One monthly latent step, newest-first history, persistent noise and private RNG |
| `inference/spinup.py` | Discarded months and optional annual latent blocks |
| `inference/metadata.py`, `archive.py` | Calendar/forcing/model identity, atomic monthly writes, resume and extension |
| `inference/reconstruction.py` | Atomic annual physical/latent reconstruction archives |
| `checkpoints.py`, `random_state.py`, `devices.py` | Small file, RNG and device helpers |
| `analysis/trajectory.py`, `trajectory_plots.py` | Main evaluation calculations and figures |
| `analysis/` | Additional climate indices, regression, spectra and map calculations |
| `experiments/` | Helpers for the specialized scripts in `scripts/research/` |

## Compatibility and checks

The old `model.train()`, package AE/diffusion trainers and generic `fit()` loop
have been removed. Historical paired training remains an ordinary script under
`scripts/legacy/`, so its existing shell recipes can still be used. It retains
its original CLI and combined checkpoint layout; it is separate from the new
component workflow. Existing combined model checkpoints, forecast initialization
modes and archived rollout resume remain supported. Serialized preprocessing and
configuration classes retain their old module paths.

The new component layout allows independent AE widths/channels, with matching
latent mesh coordinates required by coupled diffusion. Every model records its
constructor parameters, preprocessing and selected inputs. Each AE initializes
and selects independently, so a fresh run need not reproduce the prior combined
initialization or paired checkpoint selection.

`tests/test_main_path.py` runs all seven scripts with small real networks, checks
that each training stage constructs only the networks it needs, validates the
model loaders, and compares interrupted training against uninterrupted training.
It also checks reconstruction before diffusion, unequal AE channels, combined
checkpoint export/reload, changed-cache rejection, and AR resume/extension for
all four initializers. Tests edit only the scripts' top-level parameter values
in memory; the production code contains no test configuration hooks.

The rest of the suite covers preprocessing/masks, stochastic initialization,
spin-up/restarts, scientific diagnostics and the historical paired CLI.
Run `python -m pytest tests -q`.
