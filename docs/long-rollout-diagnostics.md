# Long-rollout diagnostics

From the parent `umd` directory, after activating the environment:

```bash
bash frozen/scripts/recipes/diagnose_ocean_mask_aware_long_rollout.sh
```

This analyzes the existing forecast. It runs no new forecasts. The combined
`run_ocean_mask_aware_long_rollout.sh` now also calls this full suite after its
forecast stage. The editable experiment is
`examples/diagnostics-ocean-mask-aware.yaml`; paths there are relative to
`frozen/`, unless absolute. Other checkpoints/archives can use a copied YAML.

All analysis imports come from `monthly_diffusion` in this directory. There
are no imports of experimental `umd` code or legacy prototype runners.

## Inputs and definitions

- Model: five completed stochastic members of the mask-aware 1° run. Reference:
  four held-out E3SMv2 members r2–r5. The training member is excluded from all
  physical reference comparisons.
- Evaluation: January 1851–December 2100. The forecast starts March 1850, so
  1850 is excluded from complete-year statistics. Early response: 1851–1880;
  late: 2070–2100; training climatology: 1966–2006. Endpoints are inclusive.
- Internal fields: `sqrt(N/(N-1)) * (x - mean_member(x))` at the exact month/year,
  independently within reference/model ensembles. No further monthly
  climatology subtraction. Missing ocean cells remain missing; the ensemble
  mean requires common finite member support at each cell/date.
- Indices: the package's Niño3.4 SST, Azores-minus-Iceland pressure, equatorial
  U50, and North Pacific SST EOF projection. A single area-weighted PDO loading
  is fitted on held-out training-period internal SST and used for both ensembles.
  It uses the package's existing EOF/projection normalization, not a standardized
  unit-variance PDO index. These definitions match `analysis/indices.py`.
- Teleconnections: OLS with an intercept on internal fields, pooled over members
  and months. Each ensemble's index is standardized independently. Coefficients
  retain physical units per +1 SD event. Streaming sufficient statistics give
  the same result as `analysis.climate.teleconnection`, including missing cells.
- Spectrum: per-member Welch PSD, Hann windows, 600 monthly samples, 50% overlap,
  constant detrend. ACF: per-member linear-detrended normalized autocovariance,
  through 120 months. Lines are ensemble means, envelopes are member min/max,
  not confidence intervals. No members are concatenated along the time axis.
- Distributions: KDE pooled over internal indices. Seasonal phase locking:
  pooled member/time SD separately by calendar month. These statistics use the
  full evaluation period; early/training/late amplitudes are saved separately.
- Ten-year examples: 1980–1989, one E3SM and one MD member. These are independent
  uninitialized realizations; event timing agreement is not expected.
- Global trajectories: cosine-latitude mean on each variable's finite support,
  then equal-month annual means. Lines show ensemble means in physical units;
  shading is member range. Training years are highlighted. NetCDFs retain the
  underlying monthly means for every member.
- Forced response/climatology: equal-month complete-year means. Spatial maps
  stay at native 1°. MTPR is displayed in mm/day, MSLP in hPa, OHC in GJ/m²;
  retained arrays use native source units. This suite has no pattern-scaling fit.

## Decoder probe

`decoder_gains/` uses the selected model and its digest-checked cached training
latents. It regresses each component's monthly-anomalized normalized latents
onto the detrended, monthly-anomalized standardized training-member Niño3.4
index. This is a regression direction, not a factorized or uniquely identified
physical latent mode.

For 36 base states (all months of 1966, 1986, 2006), it evaluates

$$
\left\langle D(z+\tfrac12\beta_{\rm ENSO},m,C)
            -D(z-\tfrac12\beta_{\rm ENSO},m,C)\right\rangle.
$$

Latents, month and statics stay identical across CO₂ levels: early-period mean,
350, 450, 600, late-period mean ppm. This measures decoder expression of the same
one-SD latent perturbation, including inverse preprocessing. RMS ratios and
Niño3.4 response are reported. This is not a forecast variance estimate.
The unperturbed paired early/late decodes also give a same-latent response map,
compared to held-out E3SM. That decoder-only response is distinct from the
full rollout's response.

## Memory, progress and persistence

`analysis/archives.py` reads one variable in at most 12-month blocks by default.
All members for those dates are present so ensemble subtraction is exact.
There is no Dask graph spanning the global rollout. The only longer spatial
array is the training-period North Pacific SST subset for the fixed PDO EOF.
The direct Zarr reader applies the same OHC rename and SIC×SIT derivation as
the training source adapter, after slicing.

`full_suite/configuration.json` binds cached results to configuration, forecast
checkpoint identity and reference members. Per-variable NetCDFs are written
atomically; rerunning reuses completed files and repeats only an interrupted
variable. A process lock prevents duplicate diagnostic jobs in the same output.
Progress messages report processed months and measured peak process RSS.
Use `--wait` to queue a continuation behind an already-running diagnostic stage;
without it, a duplicate invocation exits immediately. Completed calculations
are reused, although figures are rendered again.

Outputs include `reference/` and `model/` numerical products, `indices/`,
`teleconnections/`, `forced_response/`, `climatology/`, `time_series/`, and
`decoder_gains/` PNG/PDF figures. The old partial files in the parent analysis
directory are not read or reused. No raw rollout files are deleted.

Stages can also be dispatched explicitly:

```bash
bash frozen/scripts/recipes/diagnose_ocean_mask_aware_long_rollout.sh --stage physical
bash frozen/scripts/recipes/diagnose_ocean_mask_aware_long_rollout.sh --stage plots
bash frozen/scripts/recipes/diagnose_ocean_mask_aware_long_rollout.sh --stage decoder --device mps
```

The decoder probe defaults to the package's MPS/CUDA/CPU device choice; an
explicit device can be supplied. It neither retrains nor updates the checkpoint.
