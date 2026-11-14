# Architecture and scientific conventions

## Physical and latent coordinates

For variable $v$, the new scripts' `CENTERING = "gridpoint"` preprocessing is

$$
y_v(t,\phi,\lambda)=\frac{T_v[x_v(t,\phi,\lambda)]-\mu_v(\phi,\lambda)}{\widetilde\sigma_v(\phi,\lambda)}.
$$

$T_v$ is the configured identity, guarded logit, or guarded inverse-softplus.
$\mu_v$ and $\sigma_v$ are the time mean and population standard deviation at
each grid point, fitted using training months only. They include the seasonal
cycle: no month-specific centering or detrending is performed. The fitted maps
are fixed for all subsequent periods. The inverse is
$x_v=T_v^{-1}(\widetilde\sigma_v y_v+\mu_v)$.

For stable scaling in nearly constant cells, let
$s_v=\sqrt{\sum_i a_i\sigma_{v,i}^2/\sum_i a_i}$ over cells with at least two
finite training samples, with $a_i=\cos\phi_i$. The saved scale is
$\widetilde\sigma_{v,i}=\max(\sigma_{v,i}, f s_v, 10^{-6})$, where
`GRIDPOINT_STD_FLOOR` sets $f$ (default 0.05). Cells with fewer than two samples
are masked during training and inference. The fitted scale maps and actual
floors are saved with the preprocessor, AE, and combined model.

The loss remains the equal mean across samples and variables of
$\sum_i a_i m_i(\hat y_i-y_i)^2/\sum_i a_i m_i$, where $m_i$ marks valid targets.
Latitude appears once in this spatial average; local statistics are temporal.
The area-weighted reference $s_v$ only determines the floor for small local stds.

The package default and historical checkpoints retain `annual_zonal`: a
training-time/longitude mean at each latitude followed by one unweighted pooled
residual std per variable. `global` uses one mean and std per variable.
Switching modes or changing the gridpoint floor requires a fresh training run.

Latent normalization is a different operation: after selecting/fixing the AE,
fit a scalar mean and population standard deviation for each latent channel over
training time and nodes. These statistics are immutable during diffusion training
and inference. Public `encode`/`decode` apply/reverse this latent normalization;
the atomic encoder/decoder operate on raw latent coordinates.

## Edge-conditioned convolution

Let $h_i\in\mathbb R^H$ be source-node features and $e_{ij}$ the fixed geometric
features for directed edge $i\to j$. Each layer computes

$$
p_i=W_s h_i+b_s,\qquad k_{ij}=\mathrm{MLP}(e_{ij}),
$$
$$
u_j=W_o\left[\frac{1}{|\mathcal N(j)|}\sum_{i\in\mathcal N(j)}
k_{ij}\odot p_i\right]+b_o.
$$

The kernel is a **per-channel vector**, not an $H\times H$ matrix. On same-mesh
refiners the layer returns $h_j+u_j$, where the residual bypasses preconditioning.
Cross-mesh layers have no target-feature residual. Averaging is by incoming edge
count, not by cell area or barycentric weights. Barycentric maps choose three
source vertices; learned ECC kernels still determine their contributions.

`_geometry/geometry.py::compute_edge_features` defines the inherited geometric
edge vector. LSM/topography do not enter the edge kernel. Position enters a
learned Cartesian embedding at the grid encoder input. Grid-to-mesh radius is
0.6 times the maximum receiving-mesh edge length; same-mesh refiners use the
original single-hop adjacency, without adding self-loop edges.

Bucketed and padded implementations reorganize incoming edges but preserve this
equation and parameterization. They are algebraically equivalent, not a promise
of bitwise equality for all floating-point reduction orders.

The class names describe those layouts: `BucketedECC` groups destinations by
incoming neighbor count; `PaddedECC` pads every destination to one maximum count.
Production uses the former for grid-to-mesh and the latter for the remaining
transfers and refiners. `ScatterECC` is the edge-list reference used in numerical
tests. All three are available from `monthly_diffusion.ecc` under those names.

## Conditioning

For $h\in\mathbb R^{B\times N\times H}$:

$$
h'=\operatorname{SiLU}\left[
\operatorname{LN}_H(h)\odot(1+\gamma_s+\gamma_m)+\beta_s+\beta_m
\right]+c\,w_C,
\qquad c=\frac{C-\overline C_{\rm train}}{\sigma_{C,\rm train}}.
$$

- $\operatorname{LN}_H$ normalizes hidden channels independently at each batch/node;
  it has no learned affine parameters here.
- Static heads receive normalized topography and unchanged LSM, shape $1\times N\times2$,
  and output $\gamma_s,\beta_s\in\mathbb R^{1\times N\times H}$, broadcast over samples.
- Month uses $[\sin(2\pi(m-1)/12),\cos(2\pi(m-1)/12)]$, producing
  $\gamma_m,\beta_m\in\mathbb R^{B\times H}$, broadcast over nodes.
- $w_C\in\mathbb R^H$ is a learned, zero-initialized channel-only vector.
  The CO2 shift is **outside SiLU**, not a multiplicative gain.

Each component owns separate heads for each hierarchy level. Encoder and decoder
share these heads within that component. The encoder pools normalized static
fields over the actual down-map neighbors once during construction. Each FiLM
layer stores its pooled fields in a buffer; the decoder shares those same layers
and fields. A standalone decoder prepares its own fields at construction.
These derived buffers move with the model and are rebuilt from the saved run's
preprocessing when loading. Learned static/month/CO2 projections run once per
mesh level in each encoder/decoder pass and are reused by that level's refiners.
They retain their gradient graph and are recomputed next pass, so their weights
continue to train. Per-sample validity masks are
pooled during forward because their support can vary. Position is not an
additional FiLM input.

New AEs use `mask_policy="support"`. The first mesh is wet wherever any incoming
grid cell is wet, matching the refactor branch. Coarser meshes average that
support. These fractions weight source neighbors, with division by their actual
total weight even when it is below one. At the destination, the mask is binary:
a dry node is zeroed and a wet node keeps its full feature amplitude, including
its skip connection. Repeated refinement therefore cannot shrink a skip just
because its node is partly wet. Encoder and decoder use the same rule.

The policy is saved with the AE. Checkpoints without it load as `legacy`, keeping
their original fractional target scaling and mean denominators exactly. Training
refuses to resume those weights under the new policy; use a fresh ocean run.

Only the grid input has a learned position embedding. The unused historical
mesh-position heads are removed from the architecture. The external checkpoint
reader retains their old tensors solely to preserve existing archive identities.

New AEs apply FiLM only to same-mesh residual updates. The down/up transfers,
latent projections, and output projection receive unnormalized features:

```python
h = transfer(h, None, mapping)  # affine in h for fixed geometry/masks
```

Within a mesh, the spatial and channel residuals are:

```python
h = spatial(film(h, month, co2), h, mapping)
h = h + mlp(norm(h))
```

Here `norm` is an affine LayerNorm and `mlp` is
`Linear(width, 2 * width) → SiLU → Linear(2 * width, width)`, matching the refactor
branch’s extra MLP. Ocean nodes outside the propagated support remain zero after
the MLP. Both encoder and decoder use this block. Older checkpoints construct
`refiner_mlp_ratio=0`, retaining their original spatial-only refinement.

There is no LayerNorm, FiLM, or activation on the main transfer/readout path.
With residual updates disabled, the complete AE is affine in its input (with
fixed validity masks); biases and the fixed-grid position embedding explain
the constant term. ECC edge-kernel MLPs depend on geometry, not on the state.
The model as a whole remains nonlinear because of its residual updates.

Saved constructor metadata records `residual_film_only=True` for new AEs.
Files lacking that field load with the historical transfer/output FiLM behavior.
This is an architecture change: use a fresh run directory to train the new path.
The grid-level FiLM weights remain allocated for checkpoint compatibility but
are never evaluated by new models.

## AE performance audit (November 2026)

The old path evaluated grid-sized FiLM twice per reconstruction and repeated
the same learned static/month/CO2 projections for every refiner. New models
remove those grid evaluations and project shared conditioning once per mesh
level in each encoder/decoder pass. For refiners `(2, 4, 8)`, this reduces
conditioning-head evaluations from 35 to 6 per reconstruction; static heads
also run on a single copy of geography rather than duplicating it per sample.

Measured locally with PyTorch 2.5.1, CPU, four threads, a 180×360 grid, 16 input
channels, width 128, latent width 32, M4→M3→M2, and refiners `(2, 4, 8)`:

| Forward + backward | Historical FiLM path | New residual-only path |
|---|---:|---:|
| Batch 1, median of 5 steps | 467 ms | 340 ms |
| Batch 4, median of 10 steps | 1,370 ms | 908 ms |

The batch-4 comparison is about **34% less step time**. These measurements use
synthetic normalized fields and exclude optimizer/EMA, input preparation,
physical metrics, validation, and checkpoint writes. They measure the combined
architecture change and conditioning reuse. The large grid/mesh transfers still
account for substantial compute; their kernels depend on learned weights and
cannot be cached across training steps.

A separate CPU audit using four real atmosphere months (January–April 1966)
with fitted preprocessing and already-loaded fields measured median batch
preparation at 63 ms, normalized MSE at 3 ms, and inverse preprocessing plus
physical-space MSE at 9 ms. Thus physical RMSE was a small cost in this sample;
the xarray sample preparation adds more overhead even with `LOAD_IN_MEMORY=True`.
The four-month source read took 250 ms once; this is not a whole-dataset I/O
benchmark. GPU host transfers/synchronization may change these proportions.

Run `python scripts/research/benchmark_autoencoder.py` to repeat the model-only
comparison on the training device. Edit its ordinary top-level values as needed.
It synchronizes device work before timing, warms up both models, and starts
them from identical random weights. MPS/CUDA were unavailable during this audit;
the CPU speedup is not a measured GPU or full-epoch speedup.

## Transition

Each component is a full-attention DiT with separate parameters. Each block has
self-attention, history cross-attention, channel conditioning, and an SiLU FFN,
with the inherited LayerNorm/residual placements. Token identities are learned
vectors on the 162-node latent mesh; there is no geographic edge-conditioned
attention in the transition.

Both heads see their own current/previous states and the other component's
current/previous states. In the default coupled-denoising path they also see the
other component's contemporaneous noisy target. Updates are synchronous:
neither denoiser sees the other's newly updated state within that diffusion step.
Calendar contains current, previous and target month harmonics. Historical
transitions have no CO2 input. The new `condition_on_co2=True` option embeds the
target month's CO2 concentration, standardized using training-only mean/std,
and adds it to the timestep/calendar embedding that drives every block's FiLM.
Both heads have their own CO2 MLP and learned null embedding. The AEs can remain
CO2-free; the transition then carries the explicit forcing dependence.

The training script uses 10% CO2-conditioning dropout: one decision per sample,
shared across both heads, forecast horizons and solver steps in the training
rollout. Dropping CO2 substitutes the learned null embedding while retaining
calendar and state history. Validation keeps CO2 present. During sampling,
`guidance_scale` blends the two v predictions as
`v_null + guidance_scale * (v_conditional - v_null)` before the existing solver
update. Scale 1 is ordinary conditional sampling, scale 0 uses the CO2-dropped
branch, and larger scales amplify conditioning. This is
[classifier-free guidance](https://arxiv.org/abs/2207.12598) for CO2 alone.
The scale is stored in forecast provenance and checked on restart. All model
conditioning, dropout and normalization choices survive checkpoint save/load.

For normalized target latent $z$ and independent training noise $\epsilon$,

$$
z_t=\alpha_t z+\sigma_t\epsilon,\qquad
v_t=\alpha_t\epsilon-\sigma_t z,
$$

with 1000 linearly spaced betas from $10^{-4}$ to $0.02$ and
$\alpha_t^2+\sigma_t^2=1$. Training minimizes mean squared v-prediction error,
averaged equally over atmosphere/ocean and the three forecast horizons, then
multiplied by 0.01. Diffusion times are drawn uniformly during the score loss;
they are **not restricted to the seven sampling levels**.

Intermediate forecast contexts are differentiably sampled with DPM-Solver++
2M midpoint on `[999,674,373,120,29,6,0]`. Own generated-history gradients remain
live across months; default cross-component gradients are detached. Frozen AE
encodings/targets do not carry gradients. Histories always remain newest-first.

Terminal sampling noise is independently persistent in each component:

$$
\eta_{k,t}=a_k\eta_{k,t-1}+\sqrt{1-a_k^2}\,\xi_{k,t},\qquad
a_k=e^{-1/\tau_k},\quad \xi_{k,t}\sim\mathcal N(0,I),
$$

with stationary initial draw, $\tau_A=0.5$ and $\tau_O=6$ months. This temporal
correlation governs terminal forecast noise, **not** the independent noising of
training targets in the v-loss.

The optional atmosphere-first monthly Gauss–Seidel formulation first samples a
complete atmosphere, then conditions ocean loss/sampling on **that same generated
atmosphere**, not the observed future atmosphere. `cross_gradients=True` permits
ocean loss to update that atmospheric sampling path. This option is implemented
but is not the canonical default.

## Changes relative to the source experiment

Preserved: ECC learned projections and kernels, physical transforms, hierarchy/refiners,
conditioner sharing, diffusion equations, sampler, corrected history order,
terminal noise, and default no-cross-gradient coupling.

Deliberate changes: wet support masks that preserve skip amplitudes;
FiLM restricted to residual updates with an unnormalized
transfer/readout path; the residual channel MLP; AE-only first-stage training;
fixed latent statistics fitted
after selecting that AE; stage-specific EMA/checkpoint histories; fresh weights;
explicit xarray inputs and persistent iterator state. Eight frozen legacy ocean
noise compatibility parameters are omitted: they never controlled the selected
independent-noise formulation. Numerical parity tests transfer module weights
only for verification; old checkpoint metadata selects the historical AE path.
