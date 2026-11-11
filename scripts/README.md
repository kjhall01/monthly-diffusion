# The main path

Run these from the repository root after `python -m pip install -e '.[plots]'`.
Each is an ordinary top-to-bottom Python script. Edit the visible parameters and
steps directly; shared calculations live in `monthly_diffusion/`.

| Step | Script | Result |
|---|---|---|
| 1 | [train_atmosphere_autoencoder.py](train_atmosphere_autoencoder.py) | Select the atmosphere AE |
| 2 | [train_ocean_autoencoder.py](train_ocean_autoencoder.py) | Select the ocean AE, freeze both, cache normalized latents |
| 3 | [train_latent_diffusion.py](train_latent_diffusion.py) | Train the coupled transition on the fixed latent cache |
| 4 | [generate_ae_reconstruction.py](generate_ae_reconstruction.py) | Encode/decode the reference months; save fields and latents |
| 5 | [generate_ar_rollout.py](generate_ar_rollout.py) | Generate and archive autoregressive trajectories |
| 6 | [analyze_ae_reconstruction.py](analyze_ae_reconstruction.py) | Reconstruction errors, climatology and response maps |
| 7 | [analyze_ar_rollout.py](analyze_ar_rollout.py) | Climate trajectories, variability and response maps |

```bash
python scripts/train_atmosphere_autoencoder.py --run runs/e3sm --data examples/e3sm.yaml
python scripts/train_ocean_autoencoder.py --run runs/e3sm
python scripts/train_latent_diffusion.py --run runs/e3sm
python scripts/generate_ae_reconstruction.py --run runs/e3sm
python scripts/generate_ar_rollout.py --run runs/e3sm
python scripts/analyze_ae_reconstruction.py --run runs/e3sm
python scripts/analyze_ar_rollout.py --run runs/e3sm
```

Edit the source paths in `examples/e3sm.yaml` first. `--run` is the only path
needed after step 1: the scripts share `model.pt`, `data_sources.yaml`, and
`splits.yaml`. The default training period is 1966–2006, validation 1958–1965.
Change both periods in step 1; subsequent stages inherit them. Physical statistics
and latent scaling are fitted only on the training split. To reuse a model
recipe, pass `--config examples/model.yaml` to step 1; it is optional.

Training and generation accept `--device`, `--threads`, and `--resume`. Training
also accepts `--epochs`. Resume restores optimizer/EMA/RNG state for unfinished
training and skips selected stages. Choose a new run directory to retrain.
The two AEs select their best validation epochs independently. This intentionally
changes selection from the historical paired-AE trainer, so it does not promise
identical newly trained weights. The architecture, objectives, preprocessing,
optimizer, diffusion sampler, and checkpoint format are retained.

Step 2 finishes the latent cache, so step 4 can also run before step 3. AE
reconstruction writes one atomic `year-YYYY.zarr` at a time, including both physical
components and their fixed transition-normalized latents. Resume checks the
checkpoint file, source description, dates and forcing before keeping completed
years. If the checkpoint file changes, use a new reconstruction output directory.
The default period is January 1850–December 2100; use `--start` and `--end` for a
shorter reference. Missing requested months raise an error.

The default AR run generates five members for January 1850–December 2100 (3012
months). It uses the saved local-moment stochastic initializer, no discarded
spin-up, and the prescribed source CO2. If necessary, the first prescribed CO2
value supplies the two history months, November–December 1849; available earlier
forcing is retained. Use `--stochastic-start`, `--gaussian-start`,
`--encoded-noise-start`, or physical `--initialization` for another experiment.
An explicit start requires its own CO2 coverage or `--hold-first-co2-back-to`.
`--spinup-months`, CO2 controls, and `--resume --extend` remain available; see
`--help`. A resumed AR run must retain the same checkpoint, initialization,
members, seed and existing forcing. Existing combined production checkpoints can
skip the completed training stages and use `--checkpoint` / `--data` for generation.
Older checkpoints need saved local moments to use the default stochastic start;
[research/prepare_stochastic_initialization.py](research/prepare_stochastic_initialization.py)
creates a new export without changing neural weights.

## Analysis products

Each variable produces two PNG figures and two NetCDF files under
`run/analysis/ae/` or `run/analysis/ar/`. Both analyses save raw global means and
plot anomalies after subtracting each dataset/member's own 1966–2006 monthly
climatology. Trends are retained. Global means use cosine-latitude area weights
on common finite support; maps use Robinson projection with Cartopy coastlines.
Cartopy may download Natural Earth coastlines the first time it draws a map.

AE products include monthly global bias, spatial RMSE, a full-period RMSE map,
and the 1966–2006 climatological bias. AR products include per-member global
trajectories, climatological bias and temporal anomaly SD maps on the baseline
period. The SD is calculated per member before averaging variances. It includes
residual forced variability; it is not an ensemble internal-variability estimate.
Free-running AR members are not scored against the phase of the reference months.

Both save reference/generated response maps and their difference: **2070–2100
minus 1850–1880**, using equal-month annual means and inclusive endpoints. These
are period-difference estimates; with a single reference member, internal
variability remains. Generated ensemble means reduce, but do not remove, it.

Use `--variables t2m sst` for a smaller report, `--baseline FIRST LAST`, `--early`
and `--late` for other periods, or `--skip-forced-response` for a historical-only
source. Requested periods must have all complete years. `--no-plots` saves only
numerical products; `--input`, `--data`, and `--output` support existing archives.

## Other workflows

- [research/](research/): specialized diagnostics, ablations and initialization
  experiments formerly at the top of `scripts/`; filenames and CLI options stay
  the same, with `research/` added to the path.
- [recipes/](recipes/): the existing shell launchers, now pointing to the relocated
  scripts. They retain their original experiment settings and paired AE training.
- [legacy/](legacy/): the former `examples/train_model.py`, `forecast.py`, and
  `diagnose.py`, preserving the original paired-training workflow and interfaces.

The [package code map](../docs/code-map.md) describes where to change the model,
training loop, inference steps and scientific calculations.
