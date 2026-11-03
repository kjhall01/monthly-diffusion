# MonthlyDiffusion architecture v1

A working additive package derived from the uploaded November 2 production
snapshot. Implementations live here: there are no imports into the old modules
package. No existing production file is replaced. This is not a full production
migration or a trained climate model.

## Read in this order

1. `examples/architecture_walkthrough.py`: geometry → encoder → decoder → denoiser.
2. `geometry/hierarchy.py`: the explicit neighborhood recipe.
3. `autoencoder.py`: network construction and forward paths.
4. `layers/conditioning.py`: where static, month, and CO2 terms enter.
5. `denoiser.py`: network with externally supplied diffusion time and context.
6. `coupled.py`: atmosphere/ocean ownership and explicit paired calls.

Paths 2–6 are relative to `monthly_diffusion/architecture/`.

## Package map

| File | Responsibility |
| --- | --- |
| `geometry/graph.py` | SphericalGraph, EdgeSet, SphericalMap |
| `geometry/mesh.py` | Regular grids, refined icospheres, Fibonacci nodes |
| `geometry/connectivity.py` | Adjacency, radius, barycentric neighborhood builders |
| `geometry/features.py` | Coordinates and directed geometric edge features |
| `geometry/hierarchy.py` | Hierarchy assembly, rotation, static edge augmentation |
| `layers/ecc.py` | Default degree-bucketed ECC |
| `layers/_ecc_reference.py` | Gather/scatter reference implementation |
| `layers/_ecc_padded.py` | Alternative padded implementation |
| `layers/_geometry_runtime.py` | Private device-local graph layouts |
| `layers/attention.py` | Attention, feedforward, vector FiLM |
| `layers/conditioning.py` | Factorized static/month/CO2 conditioning |
| `layers/joint_conditioning.py` | Joint conditioning alternative |
| `layers/contextual_conditioning.py` | Contextual conditioning alternatives |
| `layers/embeddings.py` | Position embedding |
| `autoencoder.py` | Encoder, decoder, autoencoder, hierarchy context helpers |
| `denoiser.py` | SphericalDiT, DenoiserConfig, transformer blocks |
| `coupled.py` | Component autoencoders and paired denoiser calls |

Geometry decides who communicates; layers decide how; networks arrange layers.
Training, normalization, sampling and data I/O are outside architecture.

## Connectivity and execution

COO `sender[E]`, `receiver[E]`, `edge_attr[E,D]` are the canonical directed
connectivity. Each map records source/target meshes and their potentially different
node counts. Do not modify graph tensors in place: runtime caches assume fixed
geometry. Construction is CPU-only and non-differentiable.

The retained hierarchy uses radius neighborhoods downwards, barycentric connections
upwards, and face adjacency within meshes. Read its explicit radius arguments;
there is no policy registry. Geometric interpolation weights and learned edge
features have different meanings: inspect map construction before assuming ECC
performs fixed interpolation. StaticPool currently uses incoming-neighbor means.

COO is not a forced execution format. Default ECC builds private degree-bucketed
layouts and uses gather/multiply/sum. Reference and padded implementations remain
available. No performance improvement is claimed; profile on actual hardware first.

## Intentional changes

- Encoder/decoder conditioning is **independent by default**, initialized with equal
  weights but separate storage. Fixed geometry is shared. Set
  `share_conditioning=True` explicitly for the old ownership pattern.
- Denoiser owns no beta schedule or sampler. DenoiserConfig has no schedule fields.
  `t` must be supplied and may be real-valued. The caller owns the consistent time
  convention and whether targets are epsilon, velocity or x0.
- CoupledDenoiser receives already assembled component contexts; it does not select
  Jacobi/Gauss-Seidel coupling or AR history. CoupledAutoencoder is a container of
  independent physical encoders/decoders, not an exchange mechanism.
- Reconstruction loss and preprocessing are not part of architecture.

Conditioning variants and detailed geometry algorithms are retained for this first
version. The autoencoder helpers remain in one readable file rather than being
split across many tiny files. Existing training scripts are not automatically
rewired. Whole-object production checkpoint compatibility is not promised.

## Tensor contracts

| Tensor | Shape/meaning |
| --- | --- |
| Physical state | `[B,C,latitude,longitude]`, already transformed/normalized |
| Statics | `[B,S,latitude,longitude]`, already normalized |
| AE month | `[B,2]`, production month harmonics |
| AE CO2 | `[B,1]`, standardized CO2, **not raw ppm** |
| Latent | `[B,mesh_nodes,latent_channels]` |
| Denoiser context | `[B,mesh_nodes,condition_channels]` |
| Denoiser `f` | `[B,n_forcings]`, generic conditioning, not necessarily CO2 |
| Denoiser `t` | scalar or `[B]`, supplied diffusion time |

For ocean latent-only decoding, call `autoencoder.set_ocean_support(mask)` with
training-fitted support. This binds both independent context paths. Per-variable
physical masks and loss masks remain the caller's responsibility. Keep the
AutoencoderConfig mesh levels/refiners consistent with the hierarchy supplied.

## Try it

From the repository root, with Python 3.11+:

```bash
python -m pip install -r requirements-architecture.txt
python -m examples.architecture_walkthrough
python -m pytest tests/architecture -q
```

Keep the PyTorch build appropriate to your CUDA/MPS machine. The walkthrough uses
synthetic states: it demonstrates construction, shapes and gradients, not climate
skill. No climate data or production weights are required.

The patch does not replace root `monthly_diffusion/__init__.py` or packaging
configuration. On old v0.9 it works as a namespace package from repository root.
Use module execution above, not direct execution of the example file. Wheel/package
discovery needs an intentional update during the overall migration. If applied to
a newer checkout, its existing root initializer may require its normal dependencies.

## New GitHub branch

Use a fresh clone so local work in your current checkout is unaffected. Save the
ZIP as `~/Downloads/monthly-diffusion-architecture-v1.zip` (adjust if renamed).
Use a fresh temporary extraction directory if the one below already exists.

```bash
git clone https://github.com/kjhall01/monthly-diffusion.git monthly-diffusion-architecture
cd monthly-diffusion-architecture
git switch -c refactor/architecture-v1 origin/main
unzip -n ~/Downloads/monthly-diffusion-architecture-v1.zip -d /tmp/md-architecture-v1
cp -R /tmp/md-architecture-v1/monthly-diffusion-architecture-v1/. .
python -m pip install -r requirements-architecture.txt
python -m examples.architecture_walkthrough
python -m pytest tests/architecture -q
git status --short
git add monthly_diffusion/architecture examples/architecture_walkthrough.py tests/architecture requirements-architecture.txt ARCHITECTURE_V1.md ARCHITECTURE_PROVENANCE.json ARCHITECTURE_VALIDATION.md
git diff --cached --stat
git commit -m "Add explicit geometry and model architecture package"
git push -u origin refactor/architecture-v1
```

Review the staged files before committing. Nothing merges into main automatically.
This branch adds architecture to old main; it does not import the full November
snapshot or claim its source commit is present in GitHub. Import that baseline in
its own commit later if desired. Never interpret omissions from the supplied
production ZIP as instructions to delete files.

## Next steps

Add the readable autoencoder experiment; then migrate preprocessing and training
call sites explicitly. Add diffusion schedules and rollout outside architecture.
Rebuild identical geometry/configuration for new-network state_dict loading.
Original production checkpoint migration needs its own work; changing imports
alone is not sufficient, especially with independent conditioning.
