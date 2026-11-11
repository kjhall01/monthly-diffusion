# Read and change the emulator

Start with `examples/train_model.py` and `examples/forecast.py`. They run from
start to finish as ordinary Python scripts. The steps, branches, and loops are
visible where they execute. Helpers live in the package and can also be used
from a notebook. The scientific equations and checkpoint formats are unchanged.

## Reading order

1. **Train:** `examples/train_model.py` opens the data, selects the two time
   periods, prepares normalization, trains the AE, freezes it, caches latents,
   and trains diffusion. Its Python parameter values are editable. `--config`
   optionally reads an existing YAML recipe; it does not select the workflow.
2. **Optimize:** `training/loop.py` contains the epoch and batch loops, validation,
   checkpoint selection, and early stopping. `training/autoencoder.py` and
   `training/coupled_training.py` supply their respective objectives.
3. **Encode and decode:** `model.py` holds the model state and translates between
   physical xarray datasets and normalized latent tensors. The neural networks
   themselves are ordinary Torch modules in `modules/` and `components/`.
4. **Roll out:** `examples/forecast.py` initializes two history months, resumes
   the archive when requested, advances spin-up, then predicts, decodes, and
   writes one month at a time. `inference/forecast.py` implements a single step.
5. **Analyze:** `scripts/diagnose_long_rollout.py` shows the physical diagnostics
   loop. `examples/diagnose.py` is a smaller example. Reusable scientific
   calculations are in `analysis/`.

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
| `inference/rollout.py` | Convenience wrapper for notebook callers, including in-memory output |
| `checkpoints.py`, `random_state.py`, `devices.py` | Small shared file, RNG and timing helpers |
| `analysis/` | Climate indices, anomalies, regressions, spectra, maps and plotting |
| `experiments/` | Helpers for the matching research scripts, named by experiment |

The `scripts/` directory contains executable workflows, including the historical
initialization experiments. Their functions now live in the matching
`monthly_diffusion.experiments` module. For example, import `fit_readout` from
`monthly_diffusion.experiments.diagnose_enso_initialization`. Core training and
forecasting do not import this research package. The shell launchers still use
the same script paths and command-line options.

## Compatibility and checks

Existing `MonthlyDiffusionV1` calls, saved checkpoints, latent caches, monthly
archives, initialization modes and resume/extend behavior remain supported.
`MonthlyDiffusionV1.py`, `forecast.py` and `training/common.py` retain the old
import paths; new code uses the named modules above. Serialized preprocessing
and configuration classes retain their original module paths.

`tests/test_script_workflows.py` trains through both the editable script and the
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
