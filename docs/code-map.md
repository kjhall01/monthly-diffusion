# Read and change the emulator

Start with the seven scripts listed in the [workflow guide](../scripts/README.md).
They run from start to finish as ordinary Python scripts. The steps, branches,
and loops are visible where they execute. Helpers live in the package and can
also be used from a notebook.

## Reading order

1. **Train the AEs:** `scripts/train_atmosphere_autoencoder.py` prepares training-only
   normalization and fits the atmosphere network. `train_ocean_autoencoder.py`
   fits the ocean network, freezes both selected AEs and creates the latent cache.
   `ComponentStates` in `data/dataloader.py` loads only the component being trained.
2. **Train diffusion:** `scripts/train_latent_diffusion.py` validates the frozen
   cache, builds chronological sequences and trains the coupled transition.
   `training/loop.py` contains the shared epoch/batch loop, EMA, validation,
   checkpoint selection, early stopping and interrupted-run resume.
3. **Encode and decode:** `model.py` translates physical xarray datasets into
   latent tensors and back. Networks are in `modules/` and `components/`.
   `scripts/generate_ae_reconstruction.py` reconstructs one reference year at a
   time, saving physical fields and normalized latents together.
4. **Roll out:** `scripts/generate_ar_rollout.py` initializes two history months,
   resumes the archive when requested, advances spin-up, then predicts, decodes
   and writes one month at a time. `inference/forecast.py` implements a single step.
5. **Analyze:** `scripts/analyze_ae_reconstruction.py` and `analyze_ar_rollout.py`
   show the source selection and variable loops. `analysis/trajectory.py` contains
   the mean/error/response calculations; `trajectory_plots.py` draws the figures.

## Package layout

Paths in this table are relative to `monthly_diffusion/`.

| Location | Responsibility |
|---|---|
| `model.py` | Model state, physical encode/decode, checkpoint save/load, convenience API |
| `data/configs.py` | Parameter dataclasses, canonical variables and units; no workflow dispatch |
| `data/sources.py` | Open pinned source products and prescribed CO2 |
| `data/dataloader.py` | Validate fields, dates and units; build physical samples and latent sequences |
| `data/preprocessing.py` | Training-only physical transforms, normalization and fixed latent scaling |
| `modules/` | Mesh geometry, ECC, encoder/decoder, FiLM, attention and DiT layers |
| `components/` | Diffusion losses, component coupling, correlated noise and reverse-time sampling |
| `training/autoencoder.py`, `single_component.py`, `coupled_training.py` | Stage objectives and trainers |
| `training/loop.py` | AdamW, validation, best checkpoint selection, patience and training resume |
| `training/ema.py` | Parameter moving average and temporary validation weights |
| `training/latents.py` | One-time encoding, training-only latent scaling and cache validation |
| `training/data.py`, `resume.py` | Check the supplied splits and restore an existing training run |
| `inference/initialization.py` | Observed, Gaussian latent, encoded field-noise and local-moment histories |
| `inference/moments.py` | Fit local monthly initialization moments on the saved training dates |
| `inference/forecast.py` | Monthly latent step, newest-first history, persistent noise and private RNG |
| `inference/spinup.py` | Discarded months and optional annual latent blocks |
| `inference/metadata.py` | Calendar/forcing coverage and forecast identity |
| `inference/archive.py` | Atomic monthly writes, opening results, exact resume and horizon extension |
| `inference/reconstruction.py` | Annual reconstruction archive writes, provenance and resume |
| `inference/rollout.py` | Convenience wrapper for notebook callers, including in-memory output |
| `checkpoints.py`, `random_state.py`, `devices.py` | Small shared file, RNG and timing helpers |
| `analysis/` | Climate indices, anomalies, regressions, spectra, maps and plotting |
| `experiments/` | Helpers for the matching research scripts, named by experiment |

Specialized experiments live in `scripts/research/`; helpers are in the matching
`monthly_diffusion.experiments` module. Core training and inference do not import
that research package. Historical shell launchers live in `scripts/recipes/`
and use the relocated paths. The former examples remain in `scripts/legacy/`.

## Compatibility and checks

Existing `MonthlyDiffusionV1` calls, saved checkpoints, latent caches, monthly
archives, initialization modes and resume/extend behavior remain supported.
`MonthlyDiffusionV1.py`, `forecast.py` and `training/common.py` retain the old
import paths; new code uses the named modules above. Serialized preprocessing
and configuration classes retain their original module paths.

`tests/test_main_path.py` runs the seven scripts, including interrupted ocean
cache creation, frozen-weight checks, reconstruction parity and AR resume/extension.
`tests/test_trajectory_analysis.py` checks known biases, RMSE, seasonal anomalies,
member variability, responses and plotting.

`tests/test_script_workflows.py` trains through both the legacy paired script and the
convenience API, compares selected weights and encoded fields exactly, then
compares forecasts for all four initialization modes, including CLI extension
and preservation of existing shards. Source opening alone is substituted with
tiny synthetic data; the numerical operations and file writes run for real.
Other tests cover interrupted training, spin-up, masks, forcing checks, and the
scientific calculations. Run them with `python -m pytest tests -q`.

The readability pass was also compared against the pre-refactor source in a
separate process: deterministic training weights, reconstruction, three forecast
initializers, archive extension and a real production checkpoint had identical
numerical fingerprints on CPU. This is a behavior-preservation check, not a new
assessment of climate skill or accelerator reproducibility.

The separate AE scripts intentionally select each component's best validation
epoch. They preserve the architecture, loss definitions and checkpoint format,
but newly trained weights need not match the historical combined-validation
selection. The legacy paired trainer and convenience API retain that behavior.
