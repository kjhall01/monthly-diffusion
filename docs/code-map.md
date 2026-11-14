# Read and change the emulator

Start with the seven ordinary scripts in [scripts/README.md](../scripts/README.md).
They contain editable parameter values and the training/generation/analysis steps.
The installed package contains architectures, preprocessing, model I/O and monthly iteration.

## Architecture package

| File | What to change here |
|---|---|
| `monthly_diffusion/autoencoder.py` | Encoder, decoder, shared static/month/CO2 FiLM, grid position embedding, ECC and channel MLP refinement |
| `monthly_diffusion/diffusion.py` | DiT blocks, optional CO2 FiLM and CFG, component heads, noise schedule, AR(1) noise, coupling, complete sampling loops and numerical solver updates |
| `monthly_diffusion/preprocessing.py` | Variable definitions, physical transforms, fitted statistics and `LatentNormalizer` |
| `monthly_diffusion/loading.py` | Save/load individual networks or assemble a `ClimateModel` from a run directory |
| `monthly_diffusion/model.py` | Physical xarray encode/decode and in-memory forecasts around the networks |
| `monthly_diffusion/checkpoints.py` | File records, atomic checkpoint writes and historical weight-name translation |
| `monthly_diffusion/rollout.py` | Initial history creation, monthly forecast iteration, private RNG and in-memory forecast state |
| `monthly_diffusion/mesh.py` | Fixed-orientation hierarchy and maps between physical grid and meshes |
| `monthly_diffusion/ecc.py` | `BucketedECC` and the public imports for `PaddedECC` and the test reference `ScatterECC` |
| `monthly_diffusion/_geometry/` | Graph records, geographic features, connectivity and geometry construction |

For diffusion, read `CoupledLatentDiffusion.denoise()`, then
`denoise_paired_with_timesteps()`, then `_advance()` and `_dpmpp2m_update()`.
They are all in **diffusion.py**. The single-component sampler is in that file too.
`denoise()` supports gradients; `sample()` calls the same solver without gradients.

ECC means edge-conditioned convolution. Each implementation has one name:
`BucketedECC` groups neighbors by count; `PaddedECC` uses one padded table;
`ScatterECC` is the numerical reference used by tests. The first two are used
in the model and have identical learned parameterization. Import them from
`monthly_diffusion.ecc`; there are no additional aliases for these classes.

Models accept explicit keyword arguments such as `width=132`, `depth=4` and
`latent_channels=4`. Geometry is built once and passed directly to the AE.
No configuration object is constructed or passed between neural modules.
The geometry record has named `grid`, `meshes`, `down_maps`, `up_maps` and
`neighbor_maps` fields. There is no separate conditioning context object.

For the AE, start with `Encoder.forward()` and `Decoder.forward()` in
**autoencoder.py**. Static fields are pooled once in the constructor; forward
walks the meshes using those fixed fields. Only per-sample masks need new pooling.
`FiLM.modulation()` computes conditioning once per level and pass;
`FiLM.forward()` applies it to normalized residual inputs. `RefinementBlock`
contains both spatial mixing and the residual channel MLP. The decoder shares
`encoder.film`; the rest of the encoder/decoder machinery belongs to each class.
Transfers and the final readout operate directly on unnormalized features.

## External workflow code

`workflows/` is part of the checkout and is deliberately excluded from the wheel.
Installed model loading, preprocessing and in-memory inference work without it.
`ClimateModel.rollout(path=...)` uses its archive helpers when saving a forecast to disk.

| Location | Responsibility |
|---|---|
| `scripts/train_*` | Optimizer, epochs, validation, EMA selection and interrupted training resume |
| `workflows/run_files.py` | Create the shared run directory and write histories/selection summaries |
| `workflows/legacy_model.py`, `legacy_records.py` | Historical paired model construction and saved config records |
| `workflows/data/` | Source reads and training datasets |
| `workflows/training/` | Loss calculations, EMA, validation checks and file I/O; no training loops |
| `workflows/forecast.py` | Connect forecast iteration to decoding, spin-up and optional archives |
| `workflows/inference/` | Archive metadata, disk resume/extension and reconstruction output |
| `workflows/analysis/` | Metrics, climate diagnostics and figures |
| `workflows/experiments/` | Helpers for specialized scripts in `scripts/research/` |

The main scripts construct the actual Torch modules. Their loops do not delegate
to a trainer or orchestrator. They share a run directory containing preprocessing,
selected weights, cached latents, histories, forecasts and evaluation products.

## Existing runs and imports

New AEs use `refiner_mlp_ratio=2`: each ECC residual is followed by
`h = h + mlp(norm(h))`. This is an intentional architecture change and needs
a fresh run. New AEs also use `residual_film_only=True`: transfer and output paths
bypass FiLM. Saved files without this flag use the historical path when loaded.
To continue historical training explicitly, construct the AE with
`residual_film_only=False` and its original `REFINER_MLP_RATIO` (zero for pre-MLP
files); the main scripts otherwise require a new run directory.

Existing checkpoint weights, optimizer/EMA states and serialized preprocessing
are translated by the package loaders. Unused historical mesh-position weights
are kept only as checkpoint data so existing archive identities remain valid.
Old AEs load without adding an untrained MLP. Their encode/decode, stochastic
initialization and diffusion sampling behavior is preserved.

Python import paths intentionally change: architecture imports use the flat
package, and workflow imports use `workflows`. There are no forwarding modules
under the old package paths. For example:

```python
from monthly_diffusion import Autoencoder, CoupledLatentDiffusion, Forecast
from monthly_diffusion import load_models
from workflows.inference.archive import open_rollout
```

The historical paired CLI and its shell recipes remain under `scripts/legacy/`
and `scripts/recipes/`. They use external helpers and do not add configuration
support to the architecture package. New experiments use the seven main scripts.

Run `python -m pytest`. The suite exercises actual small-network training,
independent components, interrupted training, loading, reconstruction, every
initializer, rollout resume/extension and the retained scientific diagnostics.
