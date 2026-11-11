# DJF-anchored seasonal ENSO teleconnections

Run from the repository root:

```bash
PYTHONPATH=frozen python frozen/scripts/plot_lagged_enso.py frozen/examples/lagged-enso-e3sm.yaml
```

This reads existing physical monthly forecasts, not a new rollout. The default
is the renamed ocean-mask-aware E3SMv2 model, five emulator members, and four
held-out E3SMv2 members (r2--r5; excludes training member r1). DJF anchor years
1851--2100 supply 250 complete winters: 1,250 emulator and 1,000 reference cases.
Each member is treated as its own time sequence, never matched across ensembles.

For a winter labeled Y, field seasons are SON(Y-1), December(Y-1)--February(Y),
MAM(Y), and JJA(Y). Every panel regresses its seasonal field on the **same DJF
Niño3.4 index**, not on that field season's ENSO index. Values are OLS slopes with
an intercept, pooling member × winter equally, in **physical units per +1°C
DJF Niño3.4**. Neither index is standardized. Positive slopes indicate a positive
field anomaly associated with a warmer DJF Niño3.4; they do not establish causality.

Within each ensemble, remove the exact time-matched monthly ensemble mean and
multiply by sqrt(N/(N-1)). This is applied to fields and SST before the regional
index and seasonal means. No second monthly-climatology subtraction, linear
detrending, or spatial centering is performed. Niño3.4 uses cosine-latitude area
weighting over 5°S--5°N, 170°W--120°W. Seasonal averaging uses native E3SM no-leap
day weights; canonical Gregorian month labels do not introduce February 29.
Missing member/month field values remain unavailable; valid regression counts
are saved. Regressions pool historical and SSP3-7.0 regimes and can hide changes
in coupling with forcing. Shared-ensemble-mean subtraction removes the common
forced/seasonal component, but does not make cases statistically independent.

The two 3×4 figures place Monthly Diffusion v1.0, E3SMv2, and MD minus E3SMv2
down the rows and seasons across columns. Z500 is displayed as geopotential
height (native geopotential divided by standard gravity 9.80665 m s⁻²), in m/°C.
Precipitation is displayed in mm day⁻¹/°C (native kg m⁻² s⁻¹ multiplied by 86400).
NetCDF maps retain native field units per °C. A single symmetric color scale
applies to all twelve panels of each figure. Bounds are rounded upward from
the pooled 99.5th percentile of absolute slopes/differences; colorbar arrows
indicate saturation. Exact bounds and clipped fractions are in `display.json`.

Outputs include high-resolution PNG and PDF figures, native regression maps,
map differences, sample counts, raw winter Niño3.4 indices, spatial comparison
metrics, and configuration/checkpoint/script identities. An existing output
with changed configuration is rejected; completed ensemble maps are reusable.
To examine another period without overwriting the full-run results:

```bash
PYTHONPATH=frozen python frozen/scripts/plot_lagged_enso.py frozen/examples/lagged-enso-e3sm.yaml \
  --start-year 1966 --end-year 2006 --output frozen/analysis/lagged_enso_e3sm_training
```

Suggested caption: Seasonal atmospheric responses to DJF ENSO in Monthly
Diffusion v1.0 and E3SMv2. Columns progress from the preceding SON through DJF
to the following MAM and JJA. Maps show regression slopes per +1°C DJF Niño3.4
over 1851--2100 after separate, finite-ensemble-corrected removal of the exact
monthly ensemble mean. Bottom-row differences are Monthly Diffusion minus
E3SMv2; shared color scales allow direct comparison of amplitudes. Arrows mark
values exceeding the plotted range. No significance stippling is shown.
