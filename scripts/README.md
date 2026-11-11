# One run, seven ordinary Python scripts

Edit the values near the top of each script, then run it from the repository
root. There are no command-line options, `main()` wrappers, or package trainers
in this path. The optimizer, batches, loss, validation, early stopping and
checkpoint writes are visible in each training script.

Set the same `RUN_DIR` in all seven files, for example `Path("runs/my-experiment")`.
In the first script, also set `DATA`, `CO2`, `TRAIN_PERIOD` and `VALIDATION_PERIOD`.
It creates the directory and records those source paths and periods once.
The other scripts read that record; they do not need duplicate data settings.

```bash
python scripts/train_atmosphere_autoencoder.py
python scripts/train_ocean_autoencoder.py
python scripts/train_latent_diffusion.py
python scripts/generate_ae_reconstruction.py
python scripts/generate_ar_rollout.py
python scripts/analyze_ae_reconstruction.py
python scripts/analyze_ar_rollout.py
```

Install first with `python -m pip install -e '.[test,plots]'`. The default source
paths are Kyle's canonical 1-degree E3SM Zarr and prescribed CO2 file; edit them
for another machine. `SPECIFICATIONS` selects the component's physical variables,
transforms and loss weights from the production definitions in
`monthly_diffusion/data/configs.py`. You can supply your own `Variable` records.

## What each stage does

1. **Atmosphere:** constructs only a `MeshHierarchyAutoencoder`, fits its own
   training-only preprocessing, trains it, selects the best validation EMA, and
   encodes its training/validation splits once.
2. **Ocean:** reads the shared source/split record, constructs only its own AE,
   and follows the same explicit training and caching steps. The atmosphere
   network is neither constructed nor updated.
3. **Diffusion:** reads the two fixed latent caches and constructs only
   `CoupledLatentDiffusion`. Its optimizer and multi-step rollout loss are in the
   script. The AEs are neither constructed nor updated.
4. **AE reconstruction:** loads both selected AEs and their preprocessing/scaling,
   then encodes and decodes the original chronology, one year at a time. This can
   also run immediately after step 2, before diffusion training.
5. **AR rollout:** loads the AEs and diffusion, initializes the two history months,
   advances any discarded spin-up, then predicts, decodes and archives each month.
6. **AE analysis:** measures reconstruction bias/RMSE, global means, baseline
   climatological bias, and late-minus-early responses.
7. **AR analysis:** compares global trajectories, climatology, baseline anomaly
   variability and response patterns. It does not score a free-running member
   against the phase of the reference's individual weather realizations.

Training defaults retain the production architecture, optimizer, loss definitions,
EMA and validation noise policy. Training uses 1966–2006 and validation 1958–1965.
Each AE initializes and selects its weights independently. Independent construction
can change the random initial weights relative to the previous combined model;
new training is not promised to reproduce those learned weights bit for bit.
AE widths/channels can differ; coupled diffusion requires the same latent mesh
coordinates. The saved constructor parameters are checked when loading.

## Directory layout

```text
runs/my-experiment/
    run.json                       # source paths and shared time periods
    atmosphere/
        setup.pt                   # preprocessing, statics, constructor values, local moments
        best.pt                    # lowest validation loss, using EMA weights
        latest.pt                  # online weights, optimizer, EMA, RNG, epoch and settings
        model.pt                   # selected AE, with its setup identity
        history.csv
        latents.pt                 # training/validation latents and fixed training-only scaling
    ocean/                         # same files for the ocean AE
    diffusion/
        setup.pt                   # constructor values and exact AE/cache identities
        best.pt
        latest.pt
        model.pt
        history.csv
    ae_reconstruction/
        reconstruction.json
        year-1850.zarr              # reconstructed physical fields and both latent arrays
        ...
    ar_rollout/                    # monthly fields, restart states and provenance
    analysis/
        ae/                        # two PNGs and two NetCDF files per variable
        ar/
```

Set `RESUME = True` to continue an interrupted stage. Completed selections and
valid caches are kept; unfinished training restores the optimizer, EMA and RNG.
Changing architecture, data, preprocessing or optimizer settings requires a new
run. An interrupted cache write can be completed without retraining the AE.
`EPOCHS` and `PATIENCE` can be adjusted for unfinished training. A completed stage
is deliberately not extended by changing `EPOCHS`: use a new run for a new fit.

## Loading from Python

```python
from monthly_diffusion import load_autoencoder, load_diffusion, load_models

atmosphere, saved = load_autoencoder("runs/my-experiment", "atmosphere", device="cpu")
ocean, saved = load_autoencoder("runs/my-experiment", "ocean", device="cpu")
diffusion, saved = load_diffusion("runs/my-experiment", device="cpu")

# Physical xarray encode/decode and forecast interface; no training occurs here.
models = load_models("runs/my-experiment", device="cpu")
aes = load_models("runs/my-experiment", device="cpu", diffusion=False)
```

Each individual loader returns the actual Torch module plus saved metadata
(including preprocessing/statics for an AE). The combined loader checks that the
models, preprocessing and caches belong together. It can also open older run
directories containing `model.pt` and `data_sources.yaml`. Existing combined
checkpoint files remain loadable with `MonthlyDiffusionV1.load(path)`.

The old `model.train()` and package trainer functions have been removed.
Training lives in the scripts. The historical paired CLI in `legacy/train_model.py`
now contains its own loops for existing recipe/resume workflows; it is not part
of the new directory layout. Start a new directory to use the new training path.

## Generation and analysis choices

Reconstruction defaults to January 1850–December 2100; edit `START` and `END`
for a shorter source. It uses the saved fixed latent scaling. Atmosphere and
ocean channels have distinct dimensions in the annual archive, allowing their
channel counts to differ. Reconstruction resume is bound to the AE weights and
preprocessing; subsequently training diffusion does not invalidate it.

AR defaults to five stochastic members for 3012 saved months, starting January
1850, without discarded spin-up. `INITIALIZATION` can be `stochastic`, `gaussian`,
`encoded_noise`, or `observed`. For observed initialization, the source must cover
the two months before `START`. The default prescribed CO2 prefix supplies the
history months back to November 1849 only when the source does not already cover
them. `CONSTANT_CO2`, `CONSTANT_INITIAL_CO2`, `SPINUP_MONTHS`, and
`SAVE_SPINUP_LATENTS` control other experiments. Spin-up counts from `START` and
is additional to the `MONTHS` saved months. Set `RESUME = EXTEND = True` to extend
a consistent existing AR archive; prior files and the forcing prefix are checked.

Both analyses save raw global means and plot anomalies after subtracting each
dataset/member's own 1966–2006 monthly climatology. Trends are retained. Spatial
means use cosine-latitude weights and common finite support. Maps use Robinson
projection and Cartopy coastlines; Natural Earth data may download on first use.

AE outputs include monthly global bias and spatial RMSE, full-period RMSE maps,
and 1966–2006 climatological bias. AR variability maps use baseline-period temporal
anomaly SD, calculated per member before averaging variances. This includes
residual forced variability and is not an ensemble internal-variability estimate.

Responses use **2070–2100 minus 1850–1880**, with equal-month annual means and
inclusive endpoints. A single reference trajectory retains internal variability.
Edit `BASELINE`, `EARLY`, `LATE`, `VARIABLES`, `SKIP_FORCED_RESPONSE` and `MAKE_PLOTS`
in the analysis scripts. Requested periods must have complete years.

`research/` retains specialized experiments; `recipes/` retains historical shell
launchers; `legacy/` retains the old CLI workflows. The
[code map](../docs/code-map.md) describes the reusable package modules.
