# ENSO dynamical-realism diagnostics

Run from the `umd` repository root:

```bash
PYTHONPATH=frozen OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  python frozen/scripts/research/diagnose_enso_dynamics.py frozen/examples/enso-dynamics.yaml
```

`--stage calculate` retains numerical products without plots. `--stage plots`
regenerates figures from those products, without reading the large archives.
Completed variable products are reused. No inference, retraining, or changes to
the archived monthly forecasts are performed. The output directory is
`frozen/analysis/enso_dynamics_1958_2025/`.

## Sources and comparable anomalies

The sources are (1) the existing five-member ocean-mask-aware **E3SM-trained**
Monthly Diffusion rollout, (2) four held-out E3SMv2 members r2--r5, and (3) the
ERA5/ORAS5 reanalysis dataset. The ERA5-trained emulator is not included here.
Canonical 1-degree source fields and archive checkpoint identities are retained
in `data_provenance.json`. OHC here is **upper 300-m OHC per horizontal area**.

Monthly data span **1958--2025**. For the primary comparison, each trajectory
and each grid cell is independently fitted to twelve monthly intercepts and
one common linear time slope, jointly, over that whole interval. Subtract this
fit to define anomalies. Apply the identical procedure to all three sources.
This is retrospective climate analysis, **not training-only calibration for a
forecast-skill test**. Removing a linear trend is an approximation to removing
forced change; it is not knowledge of the true forced response.

This deliberately differs from the earlier E3SM-only teleconnections, which
subtract the exact monthly ensemble mean. A single reanalysis trajectory
cannot undergo that operation. For both ensembles we also retain the previous
`sqrt(N/(N-1)) * (field - same-month ensemble mean)` convention as the
`ensemble_residual` sensitivity. Main figures use `trajectory` anomalies.
Neither version applies index standardization; slopes remain per **+1°C**.

Ocean support is fixed from SST finite in every 1966--2006 training month: r1
for the emulator, the intersection of the four reference members for E3SM, and
reanalysis itself for ERA5/ORAS5. This support governs SST, OHC, D20, and SSH.
Variable-specific remaining NaNs remain NaN. They are never replaced by zero.
Zonal stress is an atmospheric field, positive eastward/downward, in N m⁻².

## Seasonal regression maps

There are **66 DJF anchors, 1959--2024**, allowing preceding/following months
inside the common archive interval. The seasons associated with January year Y
are SON(Y-1), December(Y-1)--February(Y), MAM(Y), and JJA(Y). Each field season
is regressed on the **same DJF Niño3.4 SST anomaly**, pooled across member ×
winter with an intercept. There are nominally 330 MD, 264 E3SM, and 66 reanalysis
cases; per-cell valid counts are saved. Seasons require all three months finite.
Maps and map metrics further require at least **90% of nominal member-winter
cases** at each cell. The unfiltered `slope`, `count`, `sufficient_support`, and
filtered `supported_slope` are all retained. This is especially important for
D20, which is undefined where the 20°C isotherm does not exist: a slope based
on two rare available winters is not a meaningful teleconnection estimate.
Gray cells are land, missing data, or inadequate coverage, not zero response.

Niño3.4 is the existing cosine-latitude-weighted SST mean over 5°S--5°N,
170°W--120°W. Seasonal averages use native no-leap days for E3SM/emulator and
Gregorian days for reanalysis. Canonical Gregorian E3SM timestamp labels do not
introduce leap days into the original model calendar.

Four Pacific-focused 4×4 figures (`lagged_ohc300`, `lagged_d20`, `lagged_tau_x`,
`lagged_ssh`) have rows MD, E3SM, reanalysis, and MD minus E3SM; columns are the
four seasons. The geographic domain is 30°S--30°N, 120°E--80°W. Shared signed
scales allow amplitude comparison. OHC displays GJ m⁻² per °C, D20 m per °C,
stress N m⁻² per °C, and SSH cm per °C. Saved calculations remain in native units.
`map_metrics.csv` retains area-weighted RMSE and centered pattern correlations
on common valid support for both anomaly methods. E3SM-vs-reanalysis differences
are climate-model differences, not errors attributable to the emulator.

## Hovmöller and recharge/wind diagnostics

`equatorial_lagged_regressions` shows longitude versus lag (-12 to +12 months)
for SST, zonal stress, D20, SSH, and OHC300, averaged over 5°S--5°N. Every lag
uses the same raw DJF Niño3.4 predictor as the maps. **Lag zero is January**;
negative lags precede January and positive lags follow it. Spatial averaging
uses latitude area weights and excludes missing cells. January fields, unlike
the DJF maps, are not three-month-averaged.

`recharge_wind_lag_correlations` compares monthly regional field anomalies at
time t+lag with Niño3.4 at time t. Negative lag therefore means the field leads
Niño3.4. Ocean regional means cover 5°S--5°N, 120°E--80°W; stress covers
5°S--5°N, 160°E--150°W. Lags never cross member boundaries. Basin-mean D20/OHC
are recharge proxies, **not an actual warm-water-volume budget or proof of
causal feedback**. Apparent eastward features in these composites do not by
themselves identify individual Kelvin waves.

## ENSO events and E/C diversity

Events are defined separately for each member from a centered, equal-month,
three-month mean of its detrended Niño3.4 anomaly. A warm episode is a contiguous
run **at or above +0.5°C**, and a cold episode is a contiguous run **at or below
-0.5°C**. Include **all durations**, as requested: no five-season persistence
criterion. Do not bridge neutral gaps, combine different members, or count a
long episode repeatedly. The event peak is the signed extreme of the smoothed
Niño3.4 within the run; ties select the earliest month.

Every event is saved in `events.csv`, including start/end/peak dates, duration,
member identity, intensity, E/C, and boundary flags. Plots use uncensored events
with peak years 1959--2024 and complete ±12-month field windows. Events touching
the available smoothed record boundary or an interior gap remain in the table
but are excluded from the displayed distributions/composites. Duration is the
number of consecutive qualifying *centers of overlapping three-month windows*.

Use the same existing Niño-box approximations from Takahashi et al. (2011):

```text
E = Niño1+2 - 0.5 × Niño4
C = 1.7 × Niño4 - 0.1 × Niño1+2
```

Niño4 covers 5°S--5°N, 160°E--150°W; Niño1+2 covers 10°S--0°, 90°W--80°W.
All box anomalies are in °C and unstandardized. These are the paper's fixed
SST-box **approximations**, not its normalized-PC indices and not necessarily
uncorrelated in these datasets. Evaluate both E and C in the SAME three-month
window as each event's Niño3.4 peak, not at their separately optimized peaks.
[Takahashi et al., equations 3--4](https://repositorio.igp.gob.pe/bitstreams/9d0e9a71-0685-4b60-9557-c76340810046/download).

`ec_event_distribution` has one dot per episode, separate warm/cold marginal
densities, and combined 50%/90% joint KDE contours (Scott bandwidth). Each
source's joint density integrates to one; marginal densities are normalized
within each phase. These are descriptive smooth distributions, not evidence
for distinct clusters or confidence regions. Counts and rates per decade per
member are retained in `event_summary.csv`.

`event_sst_d20_composites` shows equal-event-weight SST evolution around the
episode peak separately for warm and cold events, with D20 contours. SST colors
are in **°C**, D20 contours in **m**, not divided by event amplitude. Positive D20
means deeper thermocline; negative contours are dashed. Distinct events may
have overlapping ±12-month windows, so those composites are not independent
samples and no independent-grid-cell uncertainty claim is made.

This is an **ONI-like threshold definition**, not official ONI or RONI. It uses
our explicit detrended climatology and includes short episodes. Actual RONI
also subtracts tropical-mean SST anomalies and rescales variance, which is not
done here. [NOAA's RONI definition](https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/).

## What to look for, and limitations

- Do western/central Pacific stress changes accompany SST changes with the
  appropriate sign and timing?
- Does subsurface heat/depth lead surface warming, then discharge or reverse?
- Does the warm-event thermocline response propagate and decay similarly, and
  does cold-event behavior have the corresponding asymmetry?
- Is the E/C range, warm/cold balance, episode duration, and peak timing similar,
  rather than just Niño3.4 variance?

The ensembles provide more trajectories than the single reanalysis. Their
events are not date-matched to observed events. Sampling uncertainty is largest
for rare eastern-Pacific extremes, and model/reference epochs share forcing.
No significance stippling or naive independent-cell bootstrap is supplied.
Differences in reanalysis assimilation, SST/OHC/D20 definitions, and mean-state
climatology remain relevant; matching units and anomaly conventions does not
make all diagnostics physically identical. These plots test dynamical
consistency, not prove that the emulator learned the underlying causal physics.

All figures are saved as PNG and PDF; native seasonal maps, counts, monthly
indices and equatorial means, regression Hovmöllers, event composites, anomaly
fits, masks, event catalogs, code hashes, and source/checkpoint metadata remain
available. Only small derived products are added; existing forecast archives
and previous figures are untouched.

## Historical-period ENSO diversity only, all available E3SM members

To extend the event distribution without repeating the lagged-dynamics plots:

```bash
PYTHONPATH=frozen python frozen/scripts/research/diagnose_enso_diversity.py frozen/examples/enso-diversity-historical.yaml
```

This writes `frozen/analysis/enso_diversity_historical_1850_2014/` and includes
**all five locally available E3SMv2 members, including training member r1**,
and all five members of the existing ocean-mask-aware emulator archive.
It does not download additional realizations or generate forecasts. ERA5/ORAS5
does not cover 1850–2014 and is left in the original common-period analysis.
The previous analysis and all lagged-dynamics plots are retained unchanged.

The E/C formulas, cosine-latitude-weighted boxes, fixed training-SST support,
and per-trajectory joint monthly-intercept/linear-trend anomaly definition
are unchanged. The anomaly coefficients are refitted over the historical
analysis interval, not borrowed from the previous 1958–2025 fit. Only regional
SST means need be loaded: with fixed finite support, the spatial average and
the linear anomaly fit commute. The utility explicitly checks this support;
tests verify the equivalence. No ensemble-mean subtraction or standardization
is added. Raw box means, fitted coefficients, anomalies, support masks, event
catalogs, per-member counts, summary statistics, KDEs, and provenance are saved.

All threshold-crossing episodes are still allowed. Because this is a diversity
analysis, the ±12-month field window needed for the earlier lagged composites
is **not** required. Only boundary-censored events are excluded, and their
records remain in `events.csv`. E3SM has January 1850–December 2014; the emulator
archive has March 1850–December 2014, since January/February were initialization
history, not generated forecasts. No missing months are synthesized. Rates use
actual available member-month exposure. Each plotted point has equal event
weight; KDEs are descriptive, not confidence regions.

Use `--plots-only` to regenerate the diversity PNG/PDF from the saved catalogs.
Changing the configuration requires a new output directory.
