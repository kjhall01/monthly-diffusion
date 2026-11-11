# Monthly Diffusion

A standalone, readable implementation of spherical climate autoencoders and
coupled monthly latent diffusion. This branch contains the standalone production emulator imported from
`umd/frozen`. It imports no prototypes, has no implicit parent-repository file
dependencies, and does not download data. The exact imported source hashes are
recorded in [production-import.json](docs/production-import.json). Those hashes
describe the initial import, before the readability changes on this branch.

The `production/emulator-v1` branch contains the model code, training and forecast
entry points, analysis tools, example configurations, and tests. Trained
checkpoints and generated data/results are separate local artifacts. The
`*-local.yaml` examples retain the original workstation paths; edit them for
your machine or start from `examples/e3sm.yaml` / `examples/era5_oras5.yaml`.

This is a **fresh-training implementation**, not a checkpoint converter. The
architecture comes from the epoch-150 1° zonal-mean autoencoder and the corrected
four-block, 1M-per-component, six-month ocean-noise transition. Staged AE-only
training is new; it does not guarantee the same learned representation or skill
as the co-trained epoch-150 checkpoint.

## Install and test

```bash
git clone --branch production/emulator-v1 https://github.com/kjhall01/monthly-diffusion.git monthly-diffusion-production
cd monthly-diffusion-production
python -m pip install -e '.[test,plots]'
python -m pytest
```

Python 3.11+, PyTorch, NumPy/SciPy, xarray, cftime, and Zarr 2 are required.
Cartopy/Matplotlib are optional plotting dependencies. The repository retains its existing [MIT license](LICENSE).

## Public API

For learning or changing the workflow, start with the scripts:

- [`examples/train_model.py`](examples/train_model.py) shows the parameter values,
  data split, normalization, AE training, latent caching, and diffusion training
  in execution order. Edit the Python values directly, or supply `--config` to
  reproduce an existing YAML recipe. The default parameter values are unchanged.
- [`examples/forecast.py`](examples/forecast.py) shows initialization, archive
  resume, spin-up, and the monthly predict/decode/save loop.
- [`scripts/diagnose_long_rollout.py`](scripts/diagnose_long_rollout.py) shows
  archive selection, anomaly fitting, index extraction, and the variable loop.

These are executable scripts with no `main()` wrappers or locally defined
helpers. Import reusable calculations from `monthly_diffusion`, rather than
importing a script. The [code map](docs/code-map.md) gives the reading order and
the location of each helper. The convenience API below remains available for
existing notebooks; the scripts call the individual operations directly.

```python
from monthly_diffusion import MonthlyDiffusionV1

# Each component is an xr.Dataset(time, lat, lon) with physical-unit fields.
# statics is an xr.Dataset(lat, lon) containing phi_s and lsm.
# co2 is a monthly xr.DataArray(time), in ppm, covering both splits.
model = MonthlyDiffusionV1(statics=statics, config='examples/model.yaml')
model.train(
    train_atmos, train_ocean,
    validation=(val_atmos, val_ocean), co2=co2,
    directory='runs/e3sm', resume=False,
)

za, zo = model.encode(atmos, ocean, co2=co2)
reconstructed_atmos, reconstructed_ocean = model.decode(za, zo, co2=co2)
za, absent = model.encode(atmos=atmos, co2=co2)  # absent is None

forecast = model.initialize(history_atmos, history_ocean,
                            co2=future_co2, members=5, seed=42)
za_next, zo_next = next(forecast)  # time, member, node, channel
atmos_next, ocean_next = model.decode(za_next, zo_next, co2=future_co2)

fields = model.rollout(history_atmos, history_ocean, co2=future_co2,
                       months=120, members=5, seed=42,
                       path='runs/e3sm/ssp370', resume=True)
model.save('runs/e3sm/export.pt')
loaded = MonthlyDiffusionV1.load('runs/e3sm/export.pt')
```

The wrapper's `.train()` means **fit**; atomic PyTorch modules retain normal
`.train()`/`.eval()` semantics. Keyword constructor arguments override YAML.
`model.autoencoder` and `model.transition` expose the ordinary PyTorch modules.
The default device prefers MPS, then CUDA, then CPU; `device='cpu'` is explicit.

`encode` returns fixed, transition-normalized latents; `decode` reverses that
normalization. Do not mix raw encoder outputs with the public decoder API.
Physical inputs are chronological. Initialization uses their **last two adjacent
months**. Internally all transition histories are **current, previous**; after a
prediction they become **prediction, old current**, with no skipped months.
Coupled training/forecasting requires both components. Encoding and decoding may
omit either component and return `None` for it. The high-level physical API uses
a regular lat/lon grid; atomic spherical operators can use arbitrary node sets.

A scalar CO2 value means a genuinely constant prescribed pathway. A DataArray
must cover every requested month. There is no extrapolation, smoothing, invented
seasonal cycle, CO2 retrieval, or unit conversion inside inference. Calendars are
preserved, matching CO2 by year/month. Forecasts contain **future months only**,
not copies of their initialization states.

### Stochastic initialization without observed states

```python
model = MonthlyDiffusionV1.load('runs/my-run/model.pt')
forecast = model.initialize(co2=284.27, stochastic_start='1850-01',
                            members=5, seed=20260716)
za, zo = next(forecast)  # January 1850 prediction; subsequent months follow normally

# Same initializer, with compressed monthly output and exact resume:
model.rollout(co2=284.27, stochastic_start='1850-01', months=120,
              members=5, seed=20260716, path='runs/my-run/stochastic', resume=True)
```

This is the **local-moment-matched noise + one-month cycling** initializer:

1. For each variable, history month, and grid cell, draw `mean + std * epsilon`
   in the checkpoint's already transformed, centered, scaled input space.
   Moments are fitted on its exact training dates, not validation/future data.
   Cells with fewer than two finite monthly samples remain unavailable;
   zero standard deviation remains zero. Ocean support and variable masks apply.
2. Encode the two preceding months, in **current, previous** order, using
   checkpoint statics and those months' calendar/CO₂ conditioning.
3. Predict the first requested month, decode **all** fields, re-encode, and use
   the resulting latents in the current history slot. Previous history and
   atmosphere/ocean AR(1) noise are retained, with no RNG reset.
4. Continue normal latent-only autoregression. There is no continuing physical
   projection or additional field-noise injection.

The iterator returns each month's **predicted** latents, including the first
month, exactly as in the cycling experiment. Only its internal first-month
history is replaced by the re-encoded state. No month is skipped. Optional
`spinup_months` discards explicitly requested generated months, including that
first cycled month; saved spin-up latents are the predictions, with the
post-cycle history preserved in restart states. This is a heuristic initializer,
not a guarantee of equilibrium or realistic variability in every mode.

Supply **neither** physical component with `stochastic_start`; observed-history
initialization and the older `gaussian_start` / `encoded_noise_start` options
are unchanged. A CO₂ trajectory must cover both history months and generated
months; a scalar, as above, is constant. No backward extrapolation is automatic.

New `.train()` runs save the initialization moments automatically. For older
checkpoints, fit once and save a **new** export (no neural weights change):

```python
model.fit_stochastic_initialization(atmos, ocean)  # exact saved training dates selected internally
model.save('runs/my-run/model-stochastic-init.pt')
```

Or, from `frozen/`, use:

```bash
python scripts/prepare_stochastic_initialization.py runs/my-run/model.pt data.yaml --output runs/my-run/model-stochastic-init.pt
python examples/forecast.py runs/my-run/model-stochastic-init.pt data.yaml --stochastic-start 1850-01 --constant-co2 284.27 --months 120 --output runs/my-run/stochastic --resume
```

At 1°, the saved monthly local moments add approximately 0.25 GB. Once exported,
stochastic inference needs only the model and prescribed CO₂—not the training
archive or analysis scripts. Rollout provenance records the moment digest,
training dates, initialization forcing, seed, and one-cycle policy; resume does
not repeat a completed cycle.

## Default experiment

| Item | Setting |
|---|---|
| AE hierarchy | 1° grid → M4/2562 → M3/642 → M2/162 |
| Rotation | fixed Rz(11°) Ry(23°) Rx(17°), both components |
| AE width / refiners | 102; single-hop 6/6/4 per side |
| Latents | 162 nodes × 4 channels per component |
| AE trainable parameters | 3,017,193, both components combined |
| Transition | four DiT blocks, width 132, four heads, FFN ratio 1.55 |
| Transition parameters | 994,520 each; 1,989,040 combined |
| Total trainable parameters | 5,006,233 |
| Coupling | synchronous noisy-state coupling; no cross-component gradients |
| Terminal noise | independent AR(1), atmosphere 0.5 months, ocean 6 months |
| Diffusion target | full next-state v-prediction, not residual diffusion |
| Rollout objective | three equally weighted steps; loss multiplier 0.01 |
| Sampler | DPM-Solver++ 2M midpoint; 999, 674, 373, 120, 29, 6, 0 |

Grid→mesh uses degree-bucketed ECC; mesh refiners and up/down paths use padded
ECC. The radius/barycentric connectivity and learned math are unchanged. Empty
radius targets raise an error, rather than being silently assigned neighbors.
Encoder/decoder conditioning is shared **within a component**, not across
atmosphere/ocean. See [architecture.md](docs/architecture.md) for equations.

### Optional mask-aware ocean ECC

The default is intentionally unchanged: missing ocean inputs are zero-filled at
the encoder boundary and the ordinary geometric ECC denominator is retained.
For an ocean-connectivity ablation, set

```yaml
ocean_mask_aware: true
```

This preserves the same mesh and learned parameter shapes, but the ocean branch
uses the train-fitted finite-SST support mask to weight ECC messages (falling
back to continuous `1 - lsm` support for custom ocean variable sets). Mean
aggregation is renormalized over supported sources, and unsupported mesh
targets are gated after each ocean ECC. The atmosphere branch is unaffected.
This is a retraining ablation; old checkpoints with the default `false` setting
must not be interpreted as mask-aware models.

## Data and preprocessing

For the corrected ERA5/ORAS5 retrain, run
`bash frozen/scripts/train_era5_oras5_sst_support.sh`. Its separate output is
`runs/mdv1-era5-oras5-sst-support/`. The only configuration change is
`ocean_support: sst`: all ocean fields use the fixed training-SST support before
normalization and loss calculation. Additional missing cells remain missing;
decoded output and diagnostic references use the same saved mask. This does not
change LSM/topography, source products, or existing checkpoints. See the
[run guide](docs/era5-oras5-run.md#consistent-sst-support-retrain) for its matching
rollout/diagnostic command. Training and rollout are separate commands; neither
deletes the previous run.

For the matched, fresh 1° mask-aware ERA5/ORAS5 training run, use
`bash frozen/scripts/train_era5_oras5.sh` from the parent checkout. See
[era5-oras5-run.md](docs/era5-oras5-run.md) for its pinned data, unchanged model
settings, train/validation/test periods, resume command and smoke test.
After training, `bash frozen/scripts/run_era5_oras5_long_rollout.sh` runs the
five-member 1850–2100 SSP3-7.0 and constant-CO2 workflows, after discarded
1845–1849 constant-CO2 spin-up from independent Gaussian latent histories.
Use `--trajectory-only` when space permits only one archive. The same guide
documents training-period reanalysis comparisons and model-only future maps.

For the full memory-bounded long-rollout evaluation (held-out references,
teleconnections, spectra, distributions, ACF, phase locking, global trajectories,
and decoder gains), see [long-rollout-diagnostics.md](docs/long-rollout-diagnostics.md).

The MDv1 preset explicitly defines 16 atmospheric and nine ocean fields. Units
come from xarray and must agree with the preset (notation aliases are accepted).
For a custom variable set, pass `Variable(...)` records; the code does not infer
physical transforms or loss weights from variable names or arbitrary metadata.

1. SIC: guarded logit, epsilon 1e-4. Positive fields: guarded inverse-softplus
   with epsilon 1e-6 in scaled units. Others, including OLR and SST: identity.
2. Subtract one **annual zonal latitude profile**, fitted on training time ×
   longitude after transforming. No monthly climatology or detrending.
3. Divide by one pooled population **standard deviation**, not variance, of the
   residuals. These statistics are unweighted; valid physical zeros count.
4. Preserve per-variable missing values for reconstruction masks. Only encoder
   inputs are zero-filled. Default decoded masks are the union of finite training
   support; they are fixed availability masks, not predictions of missingness.
5. Normalize raw CO2 with its training mean/std. Topography uses global scalar
   mean/std; LSM remains a fraction. Static tensor order is `[phi_s, lsm]`.

Inverse-softplus fields/scales: sfcWind 5, TCWV 10, MTPR 1e-5, SSRD 100, MLD 5,
D20 50, sea-ice equivalent thickness 0.1, in their native physical units. SIC is
bounded by inverse sigmoid. Physical zero is approximated by the epsilon guard;
it is not discarded from statistics. `sivol` means **SIC × SIT in metres**, not
cell-integrated volume. The final-product adapter preserves missing land and
sets wet ice-free equivalent thickness to zero.

Reconstruction is cosine-latitude weighted MSE over each field's finite cells.
The 14 core atmospheric variables share weight 0.5; stresses have 0.05 each;
ocean variables have 0.05 each except SIC/equivalent thickness at 0.025 each.
Total weight is one. There is no special ice loss.

## Staged training and persistence

1. Fit physical preprocessing using only the supplied training dataset.
2. Train only the AE with reconstruction, selecting its best EMA validation.
3. Freeze that AE. Encode training/validation once. Fit fixed, per-channel latent
   mean/std on **training time × nodes only**, and cache standardized latents.
4. Train diffusion on the cached sequences. Own sampled-history gradients remain
   live through the seven-step sampler; cross-component histories/noisy states
   are detached by default. No diffusion gradients reach the frozen AE.

Each stage uses AdamW (lr 0.001, weight decay 0.0001, epsilon 1e-6), batch 4,
float32, no clipping, no AMP, no accumulation. EMA decay is 0.995, updated after
every optimizer step with the inherited warmup. Validation is every epoch and
uses EMA weights; transition validation replays seeds 271828 and 314159. Best
absolute validation determines `best.pt`; five stale epochs with min_delta 1e-4
determine stopping. There is no minimum-epoch rule; the maximum is 1000 by default.
For coupled diffusion, the gradient objective is multiplied by `loss_weight`
(0.01), but checkpoint selection and stale-patience use the unscaled
`val_transition` objective, so `min_delta` is independent of that balancing
multiplier.

`config.yaml`, `model.pt`, `latents.pt`, stage `latest.pt`/`best.pt`, and separate
`history.csv` files record settings, preprocessing, statics, selected weights,
optimizer/EMA/RNG state, and component/horizon losses. Interrupted stages resume;
completed stages are skipped. Changing an already-completed stage's epoch limit
does not silently retrain it. Use a fresh run for a new experiment. Checkpoints
use trusted Python serialization: **do not load untrusted `.pt` files**.

Streaming forecasts write compressed, chunked monthly Zarr shards, each with
the associated latent/history/AR-noise/RNG state. Completed shards are reused on
resume; incomplete `.partial` shards are overwritten. A manifest rejects changed
weights, initial latents, seeds, configuration, or forcing. Read lazily with
`monthly_diffusion.open_rollout(path)`, not `xr.open_zarr(path)` at the shard root.
This makes writes/resume explicit and avoids rewriting an entire long trajectory.

## Examples and reusable analysis

Edit the paths in [e3sm.yaml](examples/e3sm.yaml) or
[era5_oras5.yaml](examples/era5_oras5.yaml), then:

```bash
python examples/train_model.py examples/e3sm.yaml --output runs/e3sm
# After interruption:
python examples/train_model.py examples/e3sm.yaml --output runs/e3sm --resume
python examples/forecast.py runs/e3sm/model.pt examples/e3sm.yaml --output runs/e3sm/ssp370 --months 3010 --members 5
```

The example training script uses 1966–2006 for training, 1958–1965 for validation,
and never passes the 2007–2014 test set to training. Source YAMLs carry release
locks and the checksum of the existing annual-interpolated CO2 product. They do
not include data. ERA5 future states are projections, not future observations.

`analysis` provides existing ENSO/PDO/NAO/QBO definitions, area-weighted means,
complete seasonal/annual means, finite-ensemble-corrected internal variability,
forced-response maps, standardized-index regression teleconnections, Welch PSD,
ACF and plotting. Reference PDO loadings must be supplied for model comparisons.
There are no automatic member, period, regridding, or truth substitutions. If a
forecast starts in March 1850, 1850 is incomplete: use an explicitly chosen
complete-year response window (e.g. 1851–1880), or explicitly supply/document
initialization months rather than silently treating ten months as a full year.

See [the code map](docs/code-map.md), [validation report](docs/validation.md), and
[source hashes](docs/source-provenance.json). No real training has been launched
as part of this extraction.

### Decadal SST variability

From the **parent repository root**, run:

```bash
PYTHONPATH=frozen OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  python frozen/scripts/diagnose_decadal_sst.py frozen/examples/decadal-sst.yaml
# Redraw retained results without reopening the monthly archives:
PYTHONPATH=frozen python frozen/scripts/diagnose_decadal_sst.py \
  frozen/examples/decadal-sst.yaml --plots-only
```

The [configuration](examples/decadal-sst.yaml) specifies the four existing SST
archives, periods, basin boxes, filter, and EOF count. This diagnostic performs
no model inference and changes no training or forecast products. The reusable
calculations live in [analysis/decadal.py](monthly_diffusion/analysis/decadal.py).

Definitions and interpretation:

- Equal-month annual means require all 12 months. March-initialized rollout
  years 1850/1958 are omitted, not padded with observations. Only SST is read,
  at most one year at a time; small annual caches are reused on subsequent runs.
- The 1° fields are reduced **for this analysis only** to wet-area-weighted 5°
  means. All datasets use a common fixed ocean mask: intersect the two
  training-SST supports and archived decoder supports. Coarse cells must remain
  finite in every source/year. Retained wet area, including coastal fractions,
  weights every EOF, projection, and regional summary. Land is never zero SST.
- Ensemble internal anomalies are `sqrt(N/(N-1)) * (SST - ensemble_mean)` at
  the exact same time, calculated separately for each ensemble. All five E3SM
  members are included, including the training member. Member period means are
  removed, but **not linear trends**; no monthly climatology is subtracted again.
  The finite-member correction assumes independent realizations and does not
  make the five-member mean an exact forced response.
- A normalized 21-point annual Lanczos filter has a nominal 10-year cutoff.
  Filter each member and analysis interval separately, discard ten years at
  both ends, and remove each filtered member's remaining constant mean. No
  reflected endpoints, member concatenation, or borrowing future data into the
  historical EOF fit. `filter_response.png` shows the finite filter's rolloff;
  ten years is not a sharp spectral boundary or a ten-year running average.
- Historical inputs 1851–2014 retain **1861–2004**; future 2015–2100 retains
  **2025–2090**; matched recent 1959–2025 retains **1969–2015**. The annual and
  filtered variance denominators use the identical retained years. Historical
  and future covariance fits are not pooled. Recent results straddle 2015 and
  are labeled as recent, not purely CMIP-historical.
- Covariance EOFs use square-root wet-area weighting before SVD, pooled
  member-years, with **no gridpoint variance standardization**. Global, Pacific
  (60°S–65°N, 120–280°E), and North Atlantic (0–70°N, 280°E–20°E) are explicit
  rectangular ocean boxes, not a basin-connectivity atlas. The leading three
  historical E3SM EOFs form a common basis for every projection. EOF patterns
  have wet-area RMS one; PCs retain units of °C. Maps show patterns multiplied
  by a PC standard deviation. PCs are not independently standardized per model.
- Independent EOFs are also retained. Match the leading three by maximum
  absolute weighted pattern cosine, one-to-one, allowing rank changes and sign
  flips. These are descriptive statistical patterns, **not identified physical
  PDO/AMV modes**. Nearly tied eigenvalues can mix/rotate modes. Comparison
  against the fixed reference basis and independent fits answers different
  questions: shared-mode amplitude versus where each model puts its variance.
- `lowpass_rms_std_degC = sqrt(area_mean(mean_member(var_time(lowpass))))`.
  The low-pass fraction divides that **variance**, not standard deviation, by
  the corresponding annual variance. Member PCs and amplitude statistics are
  saved. Welch spectra use unfiltered annual PCs, 64-year segments or the
  available interval if shorter, per-member estimates, and constant segment
  detrending. Spectral member ranges are **not confidence intervals**. Short
  records and few low-frequency cycles limit interpretation; a smooth PC alone
  does not prove an oscillation. Free-running trajectories need not align in phase.
- ORAS5 is a single realization. The observational sensitivity removes a
  separate gridpoint linear trend from **every** trajectory, reference and
  emulator alike, without ensemble subtraction or finite-member correction.
  It is not an exact forced/internal decomposition: nonlinear forced signals
  can remain, genuine slow variability can be removed, and only 47 years remain
  after edge trimming. Keep it separate from the internal-ensemble comparison.

Outputs under `analysis/decadal_sst/` include publication PNG/PDF maps, common-PC
trajectories/spectra, independent EOFs, variance summaries, all annual/anomaly/
filtered fields and projection products in compressed NetCDF, CSV statistics,
filter coefficients, masks, exact coverage, configuration, source/checkpoint
identities, and source-code hashes. Raw archives are untouched. Cached annual
inputs are validated against source metadata and support; source payloads are
not rehashed in full. Configuration/source changes require a new output folder.

## Matching 5-degree run

`examples/e3sm-5deg-local.yaml` and `scripts/train_e3sm_5deg.sh` use the same
preprocessing, split, seed, optimizer, and staged workflow on the independently
generated canonical 5-degree E3SMv2 store. Because the 5-degree grid does not
provide a connected M4 grid-to-mesh stencil, this run uses
`examples/model-5deg.yaml`, which drops only M4 and retains the M3→M2 hierarchy;
the 1-degree model and `examples/model.yaml` remain unchanged.
The output directory is `runs/mdv1-e3smv2-r1i1p1f1-5deg-m3m2`.

The source YAML points to the production 5-degree path. Validate it before
training; the frozen workflow does not create a 5-degree store by reducing the
1-degree store at runtime.

```bash
cd /Users/kyle/Dev/umd/frozen
MD_PYTHON=/Users/kyle/anaconda3/envs/umd/bin/python \
  /Users/kyle/anaconda3/envs/umd/bin/python scripts/validate_sources.py \
  examples/e3sm-5deg-local.yaml \
  --state-zarr /Users/kyle/Dev/monthly-diffusion-data/data/e3smv2-five-member-1850-2100/regular_latlon_5deg/e3smv2_monthly_5deg_r1i1p1f1_185001-210012.zarr \
  --co2-netcdf /Users/kyle/Dev/monthly-diffusion-data/data/co2-forcing-v1/products/co2_monthly_e3sm_ssp370_185001-210012.nc \
  --output runs/mdv1-e3smv2-r1i1p1f1-5deg/source-validation.json
MD_PYTHON=/Users/kyle/anaconda3/envs/umd/bin/python \
  bash scripts/train_e3sm_5deg.sh
```
