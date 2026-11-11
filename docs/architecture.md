# Architecture and scientific conventions

## Physical and latent coordinates

For variable $v$, preprocessing is

$$
y_v(t,\phi,\lambda)=\frac{T_v[x_v(t,\phi,\lambda)]-\mu_v(\phi)}{\sigma_v}.
$$

$T_v$ is the configured identity, guarded logit, or guarded inverse-softplus.
$\mu_v(\phi)$ is the training-period mean over time and longitude in transformed
coordinates. $\sigma_v$ is the **population standard deviation of the residual**, pooled
over all finite training cells/times without latitude weighting. It is one scalar
per variable, not a latitude-dependent scale. The inverse is
$x_v=T_v^{-1}(\sigma_v y_v+\mu_v)$.

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
- Static heads receive normalized topography and unchanged LSM, shape $B\times N\times2$,
  and output $\gamma_s,\beta_s\in\mathbb R^{B\times N\times H}$.
- Month uses $[\sin(2\pi(m-1)/12),\cos(2\pi(m-1)/12)]$, producing
  $\gamma_m,\beta_m\in\mathbb R^{B\times H}$, broadcast over nodes.
- $w_C\in\mathbb R^H$ is a learned, zero-initialized channel-only vector.
  The CO2 shift is **outside SiLU**, not a multiplicative gain.

Each component owns separate heads for each hierarchy level. Encoder and decoder
share these heads within that component. Static fields are pooled over the
actual down-map neighbors. Modulation terms are reused within a forward pass,
not cached across optimizer steps. Position is not an additional FiLM input.

The inherited implementation owns position heads at every mesh level, although
the active ECC path uses only the grid input embedding: ECC cross-map layers do
not consume their target position seeds. These unused heads are retained in the
parameter count for exact architectural parity, but their redundant forward
evaluations have been removed. This is documented rather than silently changing
the model while extracting it.

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
Calendar contains current, previous and target month harmonics. **No CO2 or
CO2 increment is supplied to the transition.**

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

Preserved: active forward operators, masks/transforms, hierarchy/refiners,
conditioner sharing, diffusion equations, sampler, corrected history order,
terminal noise, and default no-cross-gradient coupling.

Deliberate changes: AE-only first-stage training; fixed latent statistics fitted
after selecting that AE; stage-specific EMA/checkpoint histories; fresh weights;
explicit xarray inputs and persistent iterator state. Eight frozen legacy ocean
noise compatibility parameters are omitted: they never controlled the selected
independent-noise formulation. Numerical parity tests transfer module weights
only for verification; the package does not provide a legacy checkpoint importer.
