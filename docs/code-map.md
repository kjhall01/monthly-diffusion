# Walk through the code

The scientific operations live in ordinary modules and explicit loops. There is
no plugin registry, callback framework, prototype import, or hidden source-data
download. Private geometry files retain the tested low-level algorithms rather
than duplicating them into a giant public file.

| Step | File / entry point | What to inspect |
|---|---|---|
| Resolve choices | `data/configs.py`, `ModelConfig` | Preset fields, units, transforms, loss weights, architecture and stage settings |
| Open final products | `data/sources.py` | Source locks, explicit OHC rename and equivalent-thickness derivation; no regridding |
| Validate data | `data/dataloader.py` | Adjacent monthly dates, exact grid match, units and required CO2 months |
| Fit physical statistics | `data/preprocessing.py`, `Preprocessor.fit` | Train-only transformed zonal profile and pooled residual SD |
| Build physical batches | `data/dataloader.py`, `PhysicalStates` | Lazy loading; NaNs preserved; sin/cos calendar, scalar CO2 normalization |
| Build geometry | `modules/mesh.py` | Fixed rotation, radius down maps, three-vertex up maps, explicit graph dictionaries |
| See edge features | `modules/_geometry/geometry.py` | Directed geographic feature construction |
| Compare ECC implementations | `modules/ecc.py`, `_ecc_padded.py`, `_ecc_reference.py` | Same parameters/equation; bucketed, padded, scatter reference |
| Inspect conditioning | `modules/film.py` | Node×channel statics, channel-only month, separate post-SiLU CO2 vector |
| Encode/decode | `modules/autoencoder.py` | Input projection, ECC/refinement loops, shared per-level conditioners, latent projection |
| Fit AE only | `training/autoencoder.py` | Masked area-weighted per-variable reconstruction |
| Select/freeze/cache | `MonthlyDiffusionV1.py`, `.train` | Best EMA AE selection, fresh fixed latent stats, one-time encoded cache |
| DiT backbone | `modules/spherical_dit.py`, `attention.py` | Self/cross attention, LayerNorm, FFN and v-prediction head |
| Single diffusion | `components/latent_diffusion.py` | Forward noise, v target, differentiable sampler |
| Paired diffusion | `components/coupled_latent_diffusion.py` | Context concatenation, equal component/horizon losses, explicit history updates |
| Coupling/noise | `components/coupling.py` | Jacobi/atmosphere-first choices and stationary AR(1) recurrence |
| Reverse-time numerics | `components/sampling.py` | Inherited DPM-Solver++ 2M equations and synchronous paired updates |
| Optimize/select/resume | `training/common.py` | AdamW, EMA-per-step, fixed validation draws, independent best/patience counters |
| Forecast | `forecast.py`, `Forecast.__next__` | Newest-first histories, no skipped months, private RNG and persistent noise |
| Stream/resume | `forecast.py`, `rollout` | Atomic monthly data+state shards and identity checks |
| Climate diagnostics | `analysis/` | Existing index definitions, complete seasonal means, internal residuals and regressions |

`examples/train_model.py` shows the only split selection in the example workflow.
The package itself does not choose train/validation/test periods. `examples/forecast.py`
shows how to stream SSP or scalar-CO2 forecasts. The atomic trainers accept Torch
datasets, so custom experiments need not use the high-level wrapper.

For the regular tests, start with `tests/test_workflow.py`. For equation-level
checks, see `test_operators.py`, `test_coupling.py`, `test_preprocessing.py`, and
`test_source_parity.py`. The last is optional and needs the original source
checkout; nothing in the installed runtime depends on it.
