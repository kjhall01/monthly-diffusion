# One run, seven ordinary Python scripts

Edit `RUN_DIR` and the values beside each data read, network constructor and
optimizer, then run the script from the repository root. There are no command-line
options, `main()` wrappers, or package trainers in this path. The optimizer, batches, loss, validation, early stopping and
checkpoint writes are visible in each training script. `TrainingProgress` holds
only the epoch, step counts, best loss, patience counter and history. No settings
dictionary is passed into a trainer. `workflows/training/files.py` handles saved
records; it does not run training.

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

If you stop an AE before its export finishes, set `RUN_DIR` and `COMPONENT` in
`export_autoencoder.py`, then run `python scripts/export_autoencoder.py`. It
selects the best completed validation epoch, builds its training/validation
latent cache, and writes the portable AE without running any optimizer steps.
For changed AE weights, choose a new reconstruction `OUTPUT` and matching
analysis `INPUT`; archives from different weights cannot be resumed together.

## Autoencoders with log1p precipitation and constant-cell masks

Both AE training scripts, interrupted-export utility, AE reconstruction, and AE
analysis select `scripts/runs/test_gridpoint_log1p_mtpr_masked`. Both AEs start fresh.
MTPR uses `log1p(P / P0)`, with `P0 = 1 mm/day = 1/86400 kg/m2/s`. All other
variables use identity. Each gridpoint's mean/std is fitted across all training
months after the transform. SSRD uses ordinary all-month gridpoint normalization.
There is no month-specific masking or seasonal-climatology subtraction.

`MASK_CONSTANT_CELLS = True` excludes exact training constants from normalized
inputs and loss, except constant SIC=1 cells. Equal training minimum and maximum
detect zero std before the standard-deviation floor, avoiding floating-point
roundoff around a constant mean. Cells need at least two finite training values;
any nonzero variability is retained, including seasonal zeros. Missing individual
observations still stay missing. Masks remain fixed for validation and rollout.

Decoding fills excluded constant cells with their fitted physical values, so
ice-free ocean remains zero in global means. Those cells cannot evolve in a
rollout. Physical validation RMSE includes their observed values, so changes
outside the training period still count as errors. Genuine missing/land cells
remain NaN. Old preprocessors keep their previous behavior; changing masks or
transforms requires a new run. The standard-deviation floor retains its existing
5% calculation over available training cells, before constant cells are excluded.

Both encoder and decoder use actual CO2, without jitter. The corrected ocean
support masks, variable lists, latitude-weighted/equal-variable MSE, batch size
16, learning rate 0.001, weight decay 0.0001, and four-hour budget per AE are
unchanged. Training uses 1966–2006 and validation uses 1958–1965. Output values
are not clipped to physical bounds; physical RMSE reports those errors.

```bash
conda activate monthly-diffusion
caffeinate -i bash /Users/kyle/Dev/monthly-diffusion-production/scripts/train_masked_autoencoders.sh
```

This trains atmosphere and exports its model/cache, then trains and exports
ocean. It stops on failure. Console logs go to
`scripts/logs/test_gridpoint_log1p_mtpr_masked/`. Afterward, run
`python scripts/generate_ae_reconstruction.py` and
`python scripts/analyze_ae_reconstruction.py` from the repository root.

The previous `test_gridpoint_no_transforms` run is preserved. To reproduce that
experiment in another fresh directory, set `MASK_CONSTANT_CELLS = False` and use
identity for MTPR too. Shell launchers execute the current Python settings; their
filenames do not override transforms, masks, or `RUN_DIR`.

## CO2-conditioned diffusion with classifier-free guidance

`train_latent_diffusion.py` now selects `scripts/runs/test_gridpoint_co2_cfg`.
This run has copies of the selected atmosphere/ocean AEs and latent caches from
`test_gridpoint_no_co2` (epochs 441 and 217); `ae_source.json` records their source
and file hashes. The original run, including its CO2-free diffusion model, is
preserved. The AEs still ignore CO2. Only the diffusion network is trained here.

```bash
conda activate monthly-diffusion
caffeinate -i bash /Users/kyle/Dev/monthly-diffusion-production/scripts/train_co2_diffusion.sh
```

Omit `caffeinate -i` on other platforms. The launcher uses the active environment's
`python` and saves console output to `scripts/logs/test_gridpoint_co2_cfg/diffusion.log`.
The ordinary Python script exposes `CONDITION_ON_CO2 = True` and
`CO2_DROPOUT = 0.10` next to its constructor. All previous diffusion width, depth,
batch size, learning rate, weight decay, loss, EMA, patience and seed settings are
unchanged. Set `RESUME = True` to resume an interrupted fit.

For each forecast target month, its CO2 concentration in ppm is centered/scaled
using the mean/std of **training months only**, embedded with an MLP, and added
to the timestep/calendar vector driving FiLM in every DiT block. Each head learns
its own CO2 embedding and null embedding. The normalization, dropout probability,
conditioning flag and exact training/validation forcing values are saved.

Ten percent of training sequences replace the CO2 embedding with the learned
null embedding. This is condition dropout, not ordinary activation dropout or
zero ppm. The decision is shared by atmosphere/ocean, all forecast horizons, and
every denoising call within that sequence. Calendar and history remain present.
Validation uses the supplied CO2 without dropout and selects the best EMA as before.

`generate_ar_rollout.py` selects the new run and exposes `GUIDANCE_SCALE = 1.0`.
The same option is available as `model.rollout(..., guidance_scale=...)` and
`diffusion.sample(..., co2=ppm, guidance_scale=...)`. Guidance combines predictions
at each reverse step, using identical noisy states/history/calendar in both branches:

```python
v = v_without_co2 + guidance_scale * (v_with_co2 - v_without_co2)
```

Scale **0** selects the CO2-dropped branch; **1** is ordinary conditional sampling;
values **above 1** amplify the explicit CO2 conditioning. Both 0 and 1 need one
network evaluation per head/step; other scales need two. This follows
[classifier-free guidance](https://arxiv.org/abs/2207.12598), applied only to CO2:
the null branch is still conditioned on climate history and season. Guidance is
not a calibrated multiplier of physical warming. Forecast archives record the
scale and reject a different scale on resume. Old CO2-free checkpoints retain
their original sampler behavior and do not support non-unit guidance scales.

## Four-hour training budgets and the original sequential launcher

Each training script sets `TIME_LIMIT_HOURS = 4.0`, measured from before data/model
setup. Deadlines are checked at epoch and batch boundaries in training and
validation. At expiry, an incomplete epoch is discarded and the best completed
validation epoch is selected. Export and AE latent-cache generation can finish
afterward, so elapsed time can exceed four hours by the current batch and export.
Existing early stopping can finish sooner. If no epoch has completed validation,
the stage fails clearly instead of exporting an unvalidated model.

`train_no_co2.sh` remains the three-stage launcher (atmosphere, ocean, diffusion).
Before reusing it for a new experiment, align `RUN_DIR` in all three Python scripts;
for the original fully CO2-free ablation also set all three scripts'
`CONDITION_ON_CO2 = False`. The **current** diffusion defaults instead target the
separate CFG run described above; use `train_co2_diffusion.sh` for this experiment.
The two AE scripts select the separate no-transform experiment described above.
AEs and diffusion load their own saved switches;
old AE checkpoints without a switch retain their original CO2 conditioning.

Install first with `python -m pip install -e '.[workflows,test,plots]'`. The default source
paths are Kyle's canonical 1-degree E3SM Zarr and prescribed CO2 file; edit them
for another machine. Edit `VARIABLES` in each AE script to select its fields and
channel order. `select_variables` supplies the known units and transforms
from `monthly_diffusion/preprocessing.py`; entries may also be custom `Variable`
records. The current AE scripts explicitly replace every selected transform with
`identity` immediately below `VARIABLES`. The scripts call
`open_state_dataset(DATA)` visibly at the top. This
reader accepts NetCDF or Zarr, without requiring all production variables. You can
replace that line with your own `xr.open_dataset` or `xr.open_zarr` call.

## What each stage does

1. **Atmosphere:** fits its training-only preprocessing, then constructs an
   `Autoencoder` with the normalized static fields. Those fields are pooled once
   at construction and shared by encoder and decoder. The script trains the AE,
   selects the best validation EMA, and encodes its training/validation splits once.
2. **Ocean:** reads the shared source/split record, constructs only its own AE,
   and follows the same explicit training and caching steps. The atmosphere
   network is neither constructed nor updated.
3. **Diffusion:** reads the two fixed latent caches and constructs only
   `CoupledLatentDiffusion`. Its optimizer and training loop are in the script;
   the loss calculation is in `workflows/training/losses.py`. The AEs are frozen.
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

New AE training adds a residual channel MLP after each ECC refinement, on both
the encoder and decoder. `REFINER_MLP_RATIO = 2` expands from width to twice width
and back, matching the refactor branch. FiLM now acts only on spatial residual
updates; transfers and the output projection preserve an unnormalized path.
The optimizer, EMA and validation noise policy retain the production conventions.
Use a fresh `RUN_DIR` for this architecture. Loading an existing checkpoint keeps
its original FiLM placement. Training uses 1966–2006 and validation 1958–1965.
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
        latest.pt                  # online weights, optimizer, EMA, RNG and progress
        training_options.pt        # training values saved for resume checks
        model.pt                   # selected AE, with its setup identity
        history.csv                # loss and per-variable normalized/physical RMSE
        training.log               # epoch timing and metric summaries, including units
        autoencoder.pt             # portable AE + preprocessing + latent normalizer
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
run. For a pre-MLP AE run, set `REFINER_MLP_RATIO = 0` to retain its architecture;
old weight names, optimizer slots and EMA state are translated on load.
Unfinished AE runs that used the old variable-weighted objective cannot resume
under this new loss: use a new run directory so the old best scores and patience
counters are not mixed with the new objective. Completed AEs remain loadable.
An interrupted cache write can be completed without retraining the AE.
`EPOCHS` and `PATIENCE` can be adjusted for unfinished training. A completed stage
is deliberately not extended by changing `EPOCHS`: use a new run for a new fit.

## Training measurements

The AE scripts now select `CENTERING = "gridpoint"`: subtract each grid cell's
training-time mean and divide by its training-time population standard deviation.
Both statistics are fitted after the variable's transform, across all training
months together. There is no monthly climatology subtraction or detrending.
The fitted maps stay fixed during validation, reconstruction, and rollout, and
are included in saved models. Decoding restores the maps before undoing transforms.

`GRIDPOINT_STD_FLOOR = 0.05` bounds each local std below by 5% of that variable's
area-weighted RMS local temporal std (with an absolute minimum of `1e-6` in
transformed coordinates). This limits the influence of nearly constant cells.
Cells with fewer than two finite training observations are masked in both inputs
and loss targets, including for future data. Constant observed cells remain valid.

Use a **new `RUN_DIR`** for this option. Changing the mode or floor cannot resume
an existing fit. Set `CENTERING = "annual_zonal"` to use the previous zonal mean
and one pooled std per variable, or `"global"` for a global mean and std.
Old saved models continue to use their fitted statistics.

The package utility is equally direct:

```python
preprocessor = Preprocessor(variables, centering="gridpoint", gridpoint_std_floor=0.05)
```

Both AE scripts log normalized reconstruction loss and per-variable RMSE for training
and validation at the end of each epoch. All three training scripts report
`s/epoch` for training plus validation, using the same duration stored in the
`seconds` history column (before the final progress checkpoint write).
There are no batch progress messages. Epoch measurements
are written to `history.csv` and human-readable `training.log` in each component
folder. `train_rmse_NAME` / `val_rmse_NAME` are in the variable's physical units;
`train_normalized_rmse_NAME` / `val_normalized_rmse_NAME` use training coordinates.

Physical RMSE reverses the centering, scaling and nonlinear transforms, then
compares with the original physical targets. It uses cosine-latitude area weights
on valid cells, averages each month's MSE equally, and takes the square root only
after the full epoch. Different units are never added into one physical RMSE.
Training metrics follow online weights during the epoch; validation uses EMA
weights. Training and checkpoint selection use the same objective: latitude
weights normalized to sum to one over each sample/variable's valid cells, then an
equal mean across variables. There are no additional variable multipliers.
Preprocessing and latent statistics are fitted only on the training split.

## Loading from Python

```python
from monthly_diffusion import load_autoencoder, load_diffusion, load_models, load_model

atmosphere = load_autoencoder("runs/my-experiment", "atmosphere", device="cpu")
ocean = load_autoencoder("runs/my-experiment", "ocean", device="cpu")
diffusion = load_diffusion("runs/my-experiment", device="cpu")

# The AE owns its fitted data transforms and latent scaler.
reconstructed = atmosphere.reconstruct(atmospheric_fields, co2=co2)
atmosphere.save("portable-atmosphere.pt")
restored = load_autoencoder("portable-atmosphere.pt")

# Physical xarray encode/decode and forecast interface; no training occurs here.
models = load_models("runs/my-experiment", device="cpu")
aes = load_models("runs/my-experiment", device="cpu", diffusion=False)
```

Each individual loader returns the actual Torch module. For saved-file details,
pass `return_metadata=True` to receive `(module, metadata)`. An AE exposes
`.preprocessor` and `.latent_normalizer`; its tensor forward/encode/decode retain
the training convention (normalized fields and raw latents). Its `.reconstruct()`
method takes physical xarray data and returns physical data. The combined loader checks that the
models, preprocessing and caches belong together. It can also open older run
directories containing `model.pt` and `data_sources.yaml`. Existing combined
checkpoint files remain loadable with `load_model(path)`.

Model save/load and preprocessing utilities are included in the installed package.
Training, analysis and archive helpers belong to the checkout.
The scripts add the checkout to their import path when invoked directly.
Old import paths under `monthly_diffusion.data`, `.training`, `.analysis`,
`.modules`, and `.components` have been removed. Use the paths in the code map.

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

Both analysis scripts read one variable and one calendar year at a time. A first
pass accumulates the baseline monthly means (and fixed shared AR support); a
second pass accumulates global series, errors, variability and response maps.
Only reduced series, maps and the current year's fields stay in memory. Native
storage chunks are read directly, so the calculation no longer constructs one
Dask graph for all gridded anomalies over 1850–2100. Each variable prints a start
message and elapsed seconds when its NetCDF files and figures are saved.
The same masks, latitude weights, member handling and annual-response definitions
are retained; running sums use float64 to limit accumulation roundoff.

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

## WeatherBench2 atmosphere autoencoder

`train_weatherbench_autoencoder.py` is a separate, plain training script. It uses
our same mesh `(4, 3, 2)`, width 64, 32 latent channels, `(4, 6, 6)` refiners,
residual-only FiLM and residual channel MLPs. The calendar input has four values:
sine/cosine of fractional year and UTC time of day. CO2 conditioning is disabled.
The spatial positions already identify longitude; the clock input is UTC.

Defaults select the public 1.5-degree, six-hourly ERA5 store and 71 channels:
2m temperature, 10m u/v wind, mean sea-level pressure, total-column water vapour,
and six-hour precipitation; plus geopotential, temperature, u/v wind, and specific
humidity at all 13 pressure levels. Pressure-level fields use the supplied ERA5
values/availability, including its extrapolated values below terrain; this setup
does not introduce a new below-ground mask. Edit the variable lists and levels
at the top of the file. Six-hour precipitation is accumulated water depth in
metres, not the monthly experiment's precipitation rate.

The model subtracts the supplied **1990–2017 day-of-year/hour climatology**, then
fits each cell's remaining mean/std on **1990–2017 training samples**. Validation
uses **2018–2019**. Changing the climatology requires updating its declared year
range too. All weather transforms are identity, including precipitation: these
physical anomalies may be negative. Exact raw training constants are excluded
before the std floor; physical decoding restores those constants. Seasonal
variability stays valid. Climatology is added back for physical RMSE and inference.
The std floor is 5% of each channel's typical residual std, with a tiny
`1e-12` numerical guard so upper-level humidity is not suppressed by an absolute
`1e-6 kg/kg` floor. The loss uses cosine-latitude weights normalized within each
variable's valid cells, then averages the variables equally.

The input datasets are streamed from GCS using anonymous `xarray.open_zarr`
reads. There is **no raw, normalized, climatology or latent dataset cache on disk**.
Each batch keeps source time chunks intact; chunks are shuffled within calendar
blocks and the calendar-block order changes each epoch. This mixes years while
reusing a bounded two-block climatology buffer in RAM (about 2 GB for 71 channels).
One physical batch is prefetched. Cloud bandwidth can limit throughput, and every
epoch reads its training data again. A full statistics pass precedes training;
its progress is logged every 100 batches. A stats pass interrupted before setup
is saved must restart. The four-hour training/validation budget begins afterward;
export finishes outside that budget. On this much larger dataset, raise
`TIME_LIMIT_HOURS` if a complete epoch plus validation cannot fit.

Run in the existing environment (the optional `weatherbench` dependency group
adds `gcsfs`, Zarr and Dask if needed):

```bash
conda activate monthly-diffusion
caffeinate -i python /Users/kyle/Dev/monthly-diffusion-production/scripts/train_weatherbench_autoencoder.py
```

Outputs go to `scripts/runs/weatherbench2_anomalies/atmosphere/`: fitted setup,
best/latest checkpoints, physical-unit training/validation histories, and
`autoencoder.pt`. Latent normalization is fitted on a fixed random subset of 128
training batches after selecting the best weights; only its channel moments are
saved. `RESUME = True` continues an interrupted training run. An interrupted final
export can also be completed by rerunning this script with `RESUME = True`.
The monthly `export_autoencoder.py` does not handle the six-hourly weather run.

```python
from monthly_diffusion import Autoencoder, load_autoencoder

model = load_autoencoder("scripts/runs/weatherbench2_anomalies", "atmosphere")
# Or: model = Autoencoder.load("scripts/runs/weatherbench2_anomalies/atmosphere/autoencoder.pt")
reconstructed = model.reconstruct(fields, batch_size=16)
```

Here `fields` uses the flattened names from `select_weather_fields`, with
`time/lat/lon` coordinates. The saved preprocessor retains a lazy reference to the
published climatology. Physical reconstruction therefore needs network access;
the multi-gigabyte climatology is not embedded in the checkpoint. Existing monthly
run defaults and checkpoints retain their original calendar behavior.

Sources: [WeatherBench2 data guide](https://weatherbench2.readthedocs.io/en/latest/data-guide.html),
[climatology implementation](https://github.com/google-research/weatherbench2/blob/main/weatherbench2/utils.py),
and [xarray Zarr reads](https://docs.xarray.dev/en/stable/user-guide/io.html#zarr).
