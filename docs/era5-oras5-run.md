# Matched mask-aware ERA5/ORAS5 run

## Consistent SST-support retrain

The new, separate run is `runs/mdv1-era5-oras5-sst-support`. Launch from `umd/`:

```bash
bash frozen/scripts/train_era5_oras5_sst_support.sh
# After interruption, append --resume.
```

This uses `examples/model-era5-oras5-sst-support.yaml`: the **only configuration
change** from the original reanalysis run is `ocean_support: sst`. All nine ocean
variables are restricted to the fixed mask of finite SST in any **training**
month (1966–2006), before fitting transforms' centering/scaling statistics.
Original field-specific missing values inside the mask remain missing. No
physical zeros are added, genuine ice-free zeros remain valid, and LSM/topography
and the source Zarr are unchanged. Missing normalized inputs are zero-filled
only at the encoder boundary; reconstruction loss ignores those targets.

Both stages start fresh with the same architecture, seeds, splits, noise,
optimizer and stopping criteria. Old checkpoints retain their per-variable
support; they are not silently migrated, overwritten or resumed here. This
correction changes the fitted statistics and therefore requires new AE weights,
latent caches and diffusion weights, not a continuation of the old model.

After training, the matched rollout/diagnostic workflow is:

```bash
bash frozen/scripts/run_era5_oras5_sst_support_long_rollout.sh --trajectory-only
# Or omit --trajectory-only to include the constant-CO2 control.
```

Initialization, dates and sampler are unchanged from the original workflow
below (Gaussian 1845 start, five-year latent-only spin-up, five members,
1850–2100 retained). Initialization is deliberately **not** another ablation.
Forecasts save `ocean_support.nc` and its checksum in `forecast.json`. Streaming
diagnostics automatically apply that exact mask to **both** reference and model
fields before climatologies, global means, teleconnections and index analysis.
The public `model.preprocessor.mask_ocean(raw_ocean)` method applies the same
rule for additional xarray-based evaluation. Additional source NaNs are retained.

No old rollout is automatically deleted. Training fits within the current free
space; the rollout script separately checks space and refuses to start when
insufficient. Existing ERA5 checkpoints, histories, diagnostics and rollouts
remain in `runs/mdv1-era5-oras5` for comparison.

## Encoded-noise initialization with 50-year spin-up

For the selected **SST-support** checkpoint, from `umd/` in the `umd` environment:

```bash
bash frozen/scripts/run_era5_oras5_sst_support_encoded_noise_rollout.sh
# The same command resumes an interrupted spin-up or decoded rollout.
# To run/resume only the diagnostics:
bash frozen/scripts/run_era5_oras5_sst_support_encoded_noise_rollout.sh --diagnostics-only
```

This does not retrain the model or alter the observed-initialized historical
archive. It requires MPS and uses five members, seed 20260716, the unchanged
seven-step sampler, and the trained independent 0.5/6-month AR(1) noises.

1. Draw independent N(0,1) **preprocessed fields**, not physical-unit white
   noise and not direct latent noise. The coordinate is after physical
   transforms, annual-zonal centering, and global scaling. Apply the checkpoint's
   train-fitted per-variable validity masks and fixed SST ocean support;
   unavailable inputs remain NaN until the existing encoder boundary handles
   them. No reanalysis values enter the random initial state.
2. Encode November/December 1799 with their actual calendar-month features,
   checkpoint statics, and **284.266571 ppm**, the January-1850 prescribed CO2
   value. Each component/member/history month has independent field noise.
   PCG64 seeds are `seed+1000` for atmosphere and `seed+1001` for ocean,
   separate from the diffusion RNG. Apply the fitted latent normalization once.
   Transition histories are always **[current December, previous November]**.
3. Advance **January 1800–December 1849 (600 months)**, holding the CO2
   coordinate at that same value. Save every monthly atmosphere/ocean latent
   in `spinup_latents/year-0001.pt` through `year-0050.pt`. Each atomic annual
   file includes dates, arrays shaped `[time, member, node, channel]` in fixed
   transition-normalized coordinates, and its final history/noise/RNG restart
   state. No physical fields are decoded in spin-up. An interrupted partial
   year is replayed from the last complete block, using the same RNG.
4. Without resetting histories, AR(1) noise, or RNG, continue through
   **January 1850–December 2100 (3012 months)**. Decode and retain all 25
   prognostic variables with the bit-for-bit original historical/SSP3-7.0
   monthly CO2 product. Every monthly physical shard also retains its latent
   and restart state. An identity/forcing mismatch refuses resume.
5. Run the existing bounded-memory diagnostic package: indices, distributions,
   spectra, lag ACF, phase locking, teleconnections, climatology, trajectories,
   future change maps, and same-latent decoder-gain probes. The configuration is
   `examples/diagnostics-era5-oras5-sst-support-encoded-noise.yaml`. Indices and
   teleconnections compare **1966–2006** with reanalysis, using the same
   per-trajectory monthly-intercept/linear-trend anomalies as the historical
   baseline. Spin-up never enters those diagnostics.

The transition does **not** consume CO2. Thus “constant-CO2 spin-up” means
constant forcing when the random fields are encoded and a held forcing record;
it does not add a new forcing input to latent dynamics. Fifty years is a chosen
burn-in, not a guarantee of stationarity; retaining all latent trajectories
allows that assumption to be checked without rerunning inference.

Inspect the saved spin-up without decoding or running any forecasts:

```bash
python frozen/scripts/diagnose_spinup_latents.py
# Extend into the already-saved decoded rollout, without reading physical fields:
python frozen/scripts/diagnose_spinup_latents.py --through-year 1920
```

This writes `spinup_diagnostics/` beside the archive: all-member atmospheric
and oceanic ENSO readout trajectories, ten-year mean/spread summaries,
early-versus-late distributions/spectra/ACF, and per-channel raw latent
distributions (PNG/PDF plus NetCDF/CSV). It reuses the already fitted
training-only ENSO ridge probes and verifies checkpoint/cache identities.
These are latent projections in Celsius-equivalent units, not decoded SST.
Only the fixed encoded-training calendar-month climatology is subtracted;
no spin-up trend, fitted spin-up seasonality or ensemble mean is removed.
PSD and ACF subtract window/segment means but do not linearly detrend.
Persistent offsets remain visible in the trajectory, distribution and window
mean panels. Similar pooled latent distributions do not establish recovery
of the correct temporal dynamics or prove a stationary attractor.
The `--through-year` version extracts each current latent from its monthly
`forecast_state.pt` and checks that its previous slot equals the preceding
saved current state, including the spin-up boundary. It writes a separate
`latent_diagnostics_through1920/` directory, preserves the original spin-up
plots, and marks 1850 in the extended trajectories. The final year is included
in trajectories and the last-20-year comparison; ten-year summaries use only
complete decades (through 1919 when the endpoint is December 1920).

Output, separate from all previous rollouts:

```text
frozen/runs/mdv1-era5-oras5-sst-support/long_rollout/
  ssp370_1850_2100_5member_encoded_noise_spinup1800/
    forecast.json, initial_state.pt, ocean_support.nc
    spinup_latents/year-*.pt
    month-*.zarr
    diagnostics/
```

The launcher budgets about **70 GiB** (22 MiB per decoded month, 5 GiB for
diagnostics/headroom, 0.1 GiB for latent spin-up) and never deletes anything.
On 2026-09-30, with explicit permission, only the two old experimental
`scripts/mdv1-era5-oras5/long_rollout/` SSP3-7.0 and constant-initial-CO2
`rollout.zarr` directories were deleted (approximately 64.8 GiB combined).
Their sibling diagnostics, plots, restart/checkpoint files, forcing and seed
archives were preserved. Raw fields require regeneration to recover; these
were not moved to Trash. Free space immediately afterward was about 130 GiB.

## Original run

From the `umd/` checkout, with the `umd` environment active:

```bash
bash frozen/scripts/train_era5_oras5.sh
```

After an interruption:

```bash
bash frozen/scripts/train_era5_oras5.sh --resume
```

This uses the same `examples/train_model.py` and `monthly_diffusion` package as
the selected E3SM run, now named `runs/mdv1-e3smv2-r1i1p1f1`. It starts from
fresh seeded weights, **not** the E3SM checkpoint. Output is
`frozen/runs/mdv1-era5-oras5/`. It trains both stages, but does not automatically
start forecasts or diagnostics. The launcher explicitly requires MPS, uses
four CPU threads and unbuffered logging, and refuses silent CPU fallback.

## What is held fixed

`examples/model-era5-oras5.yaml` is an explicit copy of the selected E3SM run's
resolved configuration. The empty `co2_film_levels` list explicitly preserves
CO2 conditioning at every level; this is **not** the finest-mesh-only ablation.

- 1° grid, rotated M4/M3/M2 hierarchy, refiners 6/6/4, AE width 102.
- Four latent channels on 162 nodes for each component.
- Ocean-mask-aware encoder **and** decoder; unchanged atmosphere branch.
- Four DiT blocks per component, width 132, four heads, FFN ratio 1.55.
- Coupled denoising (Jacobi), no cross-component gradients, three equally
  weighted rollout steps and the corrected current/previous history order.
- Independent atmospheric/oceanic AR(1) noise with 0.5/6-month timescales.
- 3,017,193 autoencoder + 1,989,040 transition = 5,006,233 parameters.
- Identical 25 variables, variable weights, transform scales and epsilon guards:
  logit SIC, inverse-softplus positive fields, identity elsewhere.
- Annual-zonal centering after transforms; one global residual standard
  deviation per variable. ERA5/ORAS5 statistics and masks are fitted afresh.
- Same raw annual-interpolated prescribed CO2 file as E3SM, centered/scaled on
  the training period. No replacement with a seasonally varying observed CO2
  series. This is a controlled source-data comparison, not a change in forcing.
- Same AdamW settings, batch size 4, seed 20260716, EMA 0.995 and validation
  seeds 271828/314159. Reconstruction validation needs only the first seed.
- Each stage has a 1000-epoch cap and five-stale-epoch patience, min_delta 1e-4.
  Absolute best EMA validation selects the weights. Diffusion patience and
  selection use the **unscaled** transition objective, despite the retained
  0.01 multiplier in the training objective.

Training is reconstruction-only first, then fixed train-only latent statistics
and a cached-latent diffusion stage. There is no new co-training or transfer
learning. Matching seeds does not imply identical trajectories across datasets.

## Data and splits

`examples/era5-oras5-local.yaml` points to the version-tracked final archive:

```text
/Users/kyle/Dev/monthly-diffusion-data/data/era5-oras5-1958-2025/regular_latlon_1deg/era5_oras5_monthly_1deg_195801-202512.zarr
```

The adapter checks release/source-lock/pipeline/finalization/support-policy
attributes, and checks the CO2 file SHA-256. No download, regridding, or implicit
unit conversion happens in training. The same adapter derives equivalent ice
thickness from SIC×SIT and renames the OHC band. Reanalysis statics and finite
SST support, not E3SM statics or masks, are supplied to the new model.

| Split | Inclusive years | Monthly states |
|---|---|---:|
| Training | 1966–2006 | 492 |
| Validation | 1958–1965 | 96 |
| Held-out test | 2007–2014 | 96 |

Only training/validation fields are loaded into CPU RAM (roughly 3.6 GiB before
model/optimizer memory). Batches then move to MPS. The 2007–2025 fields are not
used for preprocessing fits or either trainer. Source manifests, resolved
config, per-stage histories, best/latest checkpoints, cached latents and the
final self-contained `model.pt` are retained beside the run.

## Setup check

The following **CPU smoke test**, not a training run, uses one year of real
reanalysis to check the source, exact configuration match, calendar/units,
transforms, finite masked reconstruction gradients, and the three-step
diffusion loss/backward pass at the real architecture and batch size:

```bash
python frozen/scripts/check_training_setup.py \
  frozen/examples/era5-oras5-local.yaml \
  --config frozen/examples/model-era5-oras5.yaml \
  --reference-config frozen/runs/mdv1-e3smv2-r1i1p1f1/config.yaml \
  --output frozen/analysis/era5_oras5_setup/smoke_test.json
```

The pilot weights/statistics are discarded; no training checkpoint is created.
Actual training fits the full 1966–2006 period. This test is not an MPS speed
benchmark or evidence that the model has trained.

## Long rollouts and diagnostics

After both training stages finish, run this from the `umd/` checkout with the
`umd` environment active:

```bash
bash frozen/scripts/run_era5_oras5_long_rollout.sh
# One archive at a time if space is limited:
bash frozen/scripts/run_era5_oras5_long_rollout.sh --trajectory-only
bash frozen/scripts/run_era5_oras5_long_rollout.sh --control-only
```

This runs five-member SSP3-7.0 and constant-initial-CO2 trajectories sequentially,
with diagnostics after each. It requires MPS for inference, never falls back to
CPU, and uses the selected completed `model.pt`. Re-running the same command
resumes complete monthly shards and retained diagnostic products. A different
checkpoint, forcing or initialization is rejected instead of being mixed into
an existing archive. Nothing is deleted, and no training is started.

Both initialize the two required latent histories with independent standard
normal draws in **fixed transition-normalized coordinates**, not raw encoder
coordinates. Draws are independent across atmosphere/ocean, both history slots,
members, nodes and channels. No reanalysis state is encoded for initialization.
November/December 1844 are the history dates; January 1845 is the first advance.
NumPy PCG64 generates the initial histories separately from the Torch diffusion
RNG, both seeded with 20260716. The trained AR(1) terminal-noise processes and
seven-level sampler are unchanged; only the initial histories are white noise.

The first 60 advances (January 1845–December 1849) are latent-only spin-up at
**284.266571 ppm**, the prescribed January 1850 CO2 value. They are never decoded
or written as physical fields and never enter diagnostics. Initial histories
and an annually updated spin-up restart state are retained for reproducibility.
This is an explicit spin-up assumption, not a reconstruction of 1845 CO2 or a
guarantee that five years removes every transient.

The retained archive contains all **3012 months from January 1850 through
December 2100**. Its prescribed historical/SSP3-7.0 CO2 values are bit-for-bit
unchanged from the pinned product; the product itself is not edited. The control
keeps 284.266571 ppm throughout. Both runs start from identical Gaussian draws
and use the same diffusion seed, so their latent trajectories are paired:
the transition does not take CO2. This control isolates decoder conditioning,
not a dynamically forced ocean adjustment.

Archives and diagnostics are saved below:

```text
frozen/runs/mdv1-era5-oras5/long_rollout/
  ssp370_1850_2100_5member_gaussian_spinup1845/
    forecast.json, initial_state.pt, spinup_state.pt, month-*.zarr, diagnostics/
  constant_initial_co2_1850_2100_5member_gaussian_spinup1845/
    forecast.json, initial_state.pt, spinup_state.pt, month-*.zarr, diagnostics/
```

Each shard retains all 25 decoded variables and the latent/noise/RNG restart
state. Inference does not open all completed shards into one large Xarray graph
at exit. Diagnostics read at most 12 months of one variable at a time, with
separate bounded training-only anomaly fits; they do not load the full physical
archive into RAM.

For diagnostics alone:

```bash
bash frozen/scripts/diagnose_era5_oras5_long_rollout.sh
# Or only one completed archive:
bash frozen/scripts/diagnose_era5_oras5_long_rollout.sh --trajectory-only
bash frozen/scripts/diagnose_era5_oras5_long_rollout.sh --control-only
```

The editable specification is `examples/diagnostics-era5-oras5.yaml`. It reuses
the package's streaming evaluation and plotting code, with these distinctions
from the held-out E3SM ensemble comparison:

- Climatology maps compare the 1966–2006 annual means in physical units.
- ENSO, PDO, NAO and QBO indices and teleconnections use **only 1966–2006** in
  both reanalysis and emulator. Each trajectory independently fits twelve
  calendar-month intercepts and one linear trend at every valid grid point.
  The residual defines its anomaly. Reanalysis is one trajectory, not a
  fabricated observational ensemble; neither side uses ensemble-mean
  subtraction or finite-ensemble variance inflation in this comparison.
- This is a matched seasonal/trend-anomaly comparison, not proof that the
  forced response has been exactly removed. The fit can remove some real
  low-frequency variability. It is also an in-training-period comparison,
  not a held-out forecast-skill score.
- The PDO EOF is fitted on reanalysis training-period North Pacific SST
  anomalies and held fixed for emulator projections. Other index definitions
  are unchanged. Teleconnections regress physical-unit anomalies on each
  dataset's independently standardized index: units per +1 SD event.
- Index plots include spectra, distributions, lag ACF, seasonal phase locking,
  member amplitudes and ten-year samples. Welch segments are 240 months for
  this shorter 492-month comparison. No post-2006 values enter these fits.
- Global-mean trajectories show reanalysis for 1958–2025 and model output for
  1850–2100, in physical units with area weighting and ocean masks. Every saved
  year is complete; the spin-up years are absent from these calculations.
- Future maps show **model-only 2070–2100 minus 1966–2006** changes. There is no
  observed future map, pattern-scaling baseline or future error panel. For the
  constant control these are late-minus-training changes, not forced warming.
- Same-latent CO2-response and ENSO decoder-gain probes use training latents
  and the matched checkpoint. They appear in the SSP directory only because
  repeating checkpoint-only probes for the control would be redundant.

Anomaly coefficients, index series, regression maps and counts, climatology
and response maps, global means, spectral/ACF statistics and figures are all
retained. PNG and PDF figures are produced. The diagnostic-only script supports
`--stage physical`, `--stage plots`, `--stage decoder`, and `--wait`; use the
trajectory-only option for decoder probes.

Allow approximately 57 GiB per five-member archive, or 114 GiB for both, before
diagnostics and headroom. This scales the actual two-member reanalysis pilot;
compression will differ. The launcher conservatively budgets 22 MiB per missing
monthly shard plus 5 GiB of headroom and refuses insufficient disk space before
starting. With roughly 88 GiB free, use `--trajectory-only` first; running both
requires additional space. The script never removes a completed archive.

The runner refuses an incomplete training checkpoint. Do not retrain into or
replace `model.pt` while a forecast/diagnostic workflow is running; use a new
archive for any newly selected checkpoint.

## Space prepared on 2026-09-29

With explicit user authorization, only the following old raw archive was
removed (about 57 GiB):

```text
scripts/mdv1-e3smv2-r1i1p1f1/interim_long_rollout/epoch150/ssp370_1850_2100_5member/rollout.zarr
```

Its sibling `diagnostics/` directory retained all 223 PNG/PDF/CSV/Markdown
files counted before deletion, along with its other retained products. Its
restart checkpoint, seed states, forcing archive and progress record remain.
The renamed `frozen/runs/mdv1-e3smv2-r1i1p1f1` was not touched. Free space rose
from approximately 33 to 90 GiB. The raw archive is not in trash; reproducing
it requires rerunning inference, not resuming from its final restart state.
