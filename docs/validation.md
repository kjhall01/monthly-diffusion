# Validation and resource canary — 2026-09-28

No real training or production rollout was launched. Existing source code,
checkpoints, datasets, figures, and running experiments were not modified.

## Numerical checks

The optional source-parity tests load corresponding operator weights from
`mdv1_e3smv2_r1i1p1f1_frozenae150_dit4_1m_ordered_ocean6` into the extracted
modules on tiny fixtures. They check:

- AE raw latents, reconstructions, input gradients and parameter gradients;
- paired v-prediction losses;
- seven-step paired sampler outputs and gradients with fixed supplied noise;
- the full corrected three-horizon objective and parameter gradients, including
  AR(1) noise and generated-history updates.

The tests passed. Fixed-noise sampler outputs/gradients matched bitwise on the CPU
fixture; AE/three-horizon checks allow documented float32 tolerances. These are
operator equivalence checks, not a claim that retraining yields identical weights.
Direct bucketed/padded/scatter ECC output/gradient checks also passed on CPU and MPS.

Recorded test runs: 19 passed / 3 MPS-only skipped with source parity enabled;
the independently installed wheel passed 16 / skipped 6 (three MPS-only and three
optional source-checkout tests) from `/private/tmp`, with no `umd` module imported.
Separate MPS runs passed all five ECC/count tests and all three workflow tests.

Tests cover train-only zonal statistics, guarded transform round trips, fixed
latent statistics, valid-cell masks, canonical parameter counts and conditioner
sharing, chronological windows/newest-first concatenations, cross-gradient
detachment, generated-atmosphere reuse in Gauss–Seidel, cftime months, incomplete
forcing rejection, area weighting, complete-period response selection, finite-N
internal residuals and regression maps. Tiny staged training/checkpoint reload,
in-memory-versus-streamed forecasting, and iterator resume passed on CPU and MPS.
Optimizer/RNG training resume was compared to uninterrupted training on a small
deterministic fixture. The independently installed wheel was tested outside the
parent repository; source-parity tests are optional, not a runtime dependency.

## Actual source-data smoke checks

`scripts/validate_sources.py` was run on **January–December 1966** from each pinned
1° source product. Release attributes and the prescribed CO2 product checksum
were checked, and all 25 model fields were passed through fit/transform/inverse.
Records: `validation/source_e3sm.json`, `validation/source_era5_oras5.json`.

No infinities were produced; support/missing values were retained. Round-trip
differences reflect float32 precision and the documented guards: SIC endpoints
move by 1e-4; positive zeros are represented by scale × 1e-6. OHC absolute
round-trip errors can be thousands of J/m² because the fields are O(1e10) J/m²;
that is float32-scale error, not a unit conversion. This validates the adapter
and numerics on a real subset, not training convergence or climate skill.

## Short MPS compute measurements

Environment: macOS 26.5.2 arm64; PyTorch 2.5.1; NumPy 2.4.6; xarray 2026.7.0.
Float32, 1° grid, batch four, canonical 3,017,193 AE + 1,989,040 transition
parameters. Separate processes, one warmup and two measured batches per mode.

| Workload | Mean seconds/batch | Training compute/epoch | End-of-run MPS driver reservation |
|---|---:|---:|---:|
| AE-only reconstruction | 0.84 | 103 s / 492 states | 13.3 GiB |
| Matched co-training workload | 1.98 | 241 s / 488 windows | 17.3 GiB |
| Cached three-step diffusion | 0.57 | 69 s / 488 windows | 2.45 GiB |
| Encode both components only | 0.154 | 19 s / 492 states | 2.19 GiB |

JSON records in `validation/canary_*.json` retain individual measurements and
process RSS. MPS driver reservation includes cached allocator/driver resources:
**it is not live-tensor usage or a rigorously sampled peak-memory measurement**.
These short timings exclude source I/O, preprocessing, EMA updates, validation,
checkpoint serialization and thermal/contention effects. The matched joint
canary reproduces the main encoder/decoder/rollout computations, not the complete
historical training script. Treat epoch estimates as compute-only ballparks.

AE-only was about **2.36× faster per batch** than the matched co-training workload.
That does not imply 2.36× less total training time: there are now two sequential
training stages, and their convergence lengths are unknown. One-time encoding of
492 train + 96 validation months is approximately 23 seconds of device compute,
plus data I/O. Their two-component float32 latent cache is approximately 2.9 MiB
before serialization overhead, compared with approximately 3.55 GiB of decoded
25-variable physical states for those months.

For forecasts, 25 fields × 180 × 360 × five members × 3010 months is approximately
90.8 GiB **uncompressed**, before metadata/state. Compression depends on the
trained fields. The implementation writes one month/member/full-1°-field chunk
(about 253 KiB) rather than many tiny spatial chunks; it does not promise that a
full archive fits available disk. Check free space before launching one.

## Remaining limitations

- Fresh staged training has not been climate-validated. No claim of retaining
  epoch-150 ENSO or forced-response skill is made before training/evaluation.
- High-level data/decoder outputs use the prepared regular grid. Arbitrary-node
  geometry/ECC primitives are available; arbitrary-grid high-level decoding is
  not silently inferred from missing normalization/statics.
- Source record checks do not hash all training data chunks. Use the separately
  versioned data-pipeline products; supplied xarray datasets with no provenance
  attributes cannot provide that provenance automatically.
- No checkpoint migration, distributed training, CRPS training, full old
  diagnostics CLI, data downloading, nested Git initialization, or publishing was
  added. Those are outside this implementation.

To rerun source parity from this original checkout:

```bash
MONTHLY_DIFFUSION_SOURCE_ROOT=/path/to/umd python -m pytest tests/test_source_parity.py
```

To reproduce a canary after installing the standalone package:

```bash
python scripts/resource_canary.py --mode autoencoder --output validation/canary_autoencoder.json
python scripts/resource_canary.py --mode joint --output validation/canary_joint.json
python scripts/resource_canary.py --mode diffusion --output validation/canary_diffusion.json
python scripts/resource_canary.py --mode encode --output validation/canary_encode.json
```
