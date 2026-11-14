"""Spherical autoencoders with FiLM, spatial mixing and channel mixing.

Read Encoder.forward(), then Decoder.forward(). Each walks its meshes directly.
Physical fields use [batch, channel, latitude, longitude]; mesh features use
[batch, node, channel]. FiLM is shared by encoder and decoder at each resolution.
Supply normalized statics [2, latitude, longitude] once at construction; batches
then need only state, month and CO2. Static pooling never runs in forward().
Transfers and projections carry unnormalized features. FiLM and LayerNorm act
only inside residual updates; disabling those updates leaves an affine network.
"""

import torch
from torch import nn
from torch.nn import functional as F
from .ecc import BucketedECC, PaddedECC
from .preprocessing import LatentNormalizer


class FiLM(nn.Module):
    """Condition one mesh on geography, cyclic calendar features and optional CO2."""

    def __init__(self, width, statics, *, include_co2=True, calendar="monthly"):
        super().__init__()
        self.include_co2 = include_co2
        if calendar not in ("monthly", "day_hour"):
            raise ValueError("calendar must be monthly or day_hour")
        calendar_channels = 2 if calendar == "monthly" else 4
        # Fixed, already-pooled geography. Rebuilt from the run's preprocessing
        # when loading; moves with .to(device/dtype) but is not a learned weight.
        self.register_buffer(
            "static_fields", statics.detach().clone(), persistent=False
        )
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.static = nn.Sequential(
            nn.Linear(2, width), nn.SiLU(), nn.Linear(width, 2 * width)
        )
        self.month = nn.Sequential(
            nn.Linear(calendar_channels, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.month_affine = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        # Keep the parameter layout even when CO2 is disabled, so switching it
        # off leaves every other initial weight unchanged for the same seed.
        self.co2_shift = nn.Linear(1, width, bias=False)
        # Start with ordinary normalized features; learn the conditioning offsets.
        nn.init.zeros_(self.static[-1].weight)
        nn.init.zeros_(self.static[-1].bias)
        nn.init.zeros_(self.month_affine[-1].weight)
        nn.init.zeros_(self.month_affine[-1].bias)
        nn.init.zeros_(self.co2_shift.weight)

    def modulation(self, month, co2, *, batch_statics=False):
        """Compute learned conditioning once per level, within this forward pass.

        Geography is identical across the batch. Its learned projection can be
        broadcast, then reused by every refiner at this level. Keep its graph:
        caching it across optimizer steps would use stale weights/gradients.
        """
        statics = self.static_fields
        if batch_statics:  # reproduce the exact arithmetic of old checkpoints
            statics = statics.expand(month.shape[0], -1, -1)
        static_scale, static_shift = self.static(statics).chunk(2, dim=-1)
        month_scale, month_shift = self.month_affine(self.month(month)).chunk(2, dim=-1)
        scale = 1 + static_scale + month_scale[:, None, :]
        co2_shift = self.co2_shift(co2)[:, None, :] if self.include_co2 else None
        return scale, static_shift, month_shift[:, None, :], co2_shift

    def forward(self, features, month, co2, *, modulation=None):
        if modulation is None:
            modulation = self.modulation(month, co2, batch_statics=True)
        scale, static_shift, month_shift, co2_shift = modulation
        features = self.norm(features)
        features = features * scale
        features = features + static_shift + month_shift
        features = F.silu(features)
        # Preserve the production convention: CO2 is added AFTER SiLU.
        if co2_shift is not None:
            features = features + co2_shift
        return features


class MeanPool(nn.Module):
    """Average fixed geographic fields from one grid/mesh onto the next."""

    def __init__(self, mapping):
        super().__init__()
        self.n_target = mapping.n_target
        self.register_buffer("sender", mapping.edge_set.sender.clone())
        self.register_buffer("receiver", mapping.edge_set.receiver.clone())
        counts = torch.bincount(self.receiver, minlength=self.n_target)
        self.register_buffer("count", counts.float().clamp_min(1))

    def forward(self, values):
        result = values.new_zeros(values.shape[0], self.n_target, values.shape[-1])
        index = self.receiver[None, :, None].expand(
            values.shape[0], -1, values.shape[-1]
        )
        result.scatter_add_(1, index, values[:, self.sender])
        return result / self.count.to(values)[None, :, None]


class RefinementBlock(nn.Module):
    """Mix neighboring nodes with ECC, then mix channels with a residual MLP."""

    def __init__(self, width, edge_channels, mlp_ratio=2, *, mask_policy="support"):
        super().__init__()
        self.spatial = PaddedECC(
            width,
            width,
            edge_channels,
            width,
            residual=True,
            aggregation="mean",
            mask_policy=mask_policy,
        )
        self.mlp = None
        # A zero ratio is only needed to load older AEs that had no MLP.
        if mlp_ratio > 0:
            self.norm = nn.LayerNorm(width)
            self.mlp = nn.Sequential(
                nn.Linear(width, int(width * mlp_ratio)),
                nn.SiLU(),
                nn.Linear(int(width * mlp_ratio), width),
            )

    def forward(self, features, conditioned_features, mapping, node_mask=None):
        # ECC already adds the original features as its residual connection.
        features = self.spatial(
            conditioned_features,
            features,
            mapping,
            source_mask=node_mask,
            target_mask=node_mask,
        )
        if self.mlp is not None:
            features = features + self.mlp(self.norm(features))
            if node_mask is not None:
                features = features * (node_mask > 0).unsqueeze(-1)
        return features


class Encoder(nn.Module):
    """Physical grid -> fine mesh -> coarse mesh -> latent channels."""

    def __init__(
        self,
        channels,
        geometry,
        *,
        statics,
        width=102,
        latent_channels=4,
        refiners=(6, 6, 4),
        co2_film_levels=(),
        condition_on_co2=True,
        calendar="monthly",
        refiner_mlp_ratio=2,
        residual_film_only=True,
        mask_policy="support",
    ):
        super().__init__()
        if len(refiners) != len(geometry.meshes):
            raise ValueError("Supply one refiner count for each mesh")
        if refiner_mlp_ratio < 0 or (0 < refiner_mlp_ratio < 1 / width):
            raise ValueError("MLP width must be positive, or ratio zero for an old AE")
        self.geometry = geometry
        self.width, self.latent_channels = width, latent_channels
        self.refiner_counts = tuple(refiners)
        self.refiner_mlp_ratio = refiner_mlp_ratio
        # False exists only to reproduce models trained before the residual fix.
        self.residual_film_only = residual_film_only
        # 'legacy' exists only to reproduce the masks used by older checkpoints.
        self.mask_policy = mask_policy
        self.co2_film_levels = tuple(co2_film_levels)
        self.condition_on_co2 = bool(condition_on_co2)
        self.calendar = calendar
        n_levels = len(geometry.meshes) + 1  # physical grid plus each mesh
        if any(i < 0 or i >= n_levels for i in co2_film_levels):
            raise ValueError("CO2 FiLM level is outside the hierarchy")
        if (
            statics.ndim != 3
            or statics.shape[0] != 2
            or statics[0].numel() != geometry.grid.n_nodes
        ):
            raise ValueError(
                "statics must contain normalized topography and land mask: [2, latitude, longitude]"
            )
        self.film = nn.ModuleList()
        # Keep grid FiLM (index 0) for old checkpoint weight names. New models
        # condition mesh residuals only, so never evaluate this grid-sized MLP.
        self.static_pools = nn.ModuleList([MeanPool(m) for m in geometry.down_maps])
        # Geography is fixed for the run. Pool it once, before any batches arrive.
        fields = statics.detach().cpu().permute(1, 2, 0).flatten(0, 1)[None]
        for level in range(n_levels):
            if level > 0:
                fields = self.static_pools[level - 1](fields)
            self.film.append(
                FiLM(
                    width,
                    fields,
                    calendar=calendar,
                    include_co2=condition_on_co2
                    and (not co2_film_levels or level in co2_film_levels),
                )
            )
        self.position = nn.Sequential(
            nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.register_buffer("grid_xyz", geometry.grid.xyz)
        self.input_projection = nn.Linear(channels, width)
        self.latent_projection = nn.Linear(width, latent_channels)
        self.down = nn.ModuleList()
        self.refiners = nn.ModuleList()
        for level, mapping in enumerate(geometry.down_maps):
            convolution = BucketedECC if level == 0 else PaddedECC
            self.down.append(
                convolution(
                    width,
                    width,
                    mapping.edge_set.edge_attr_dim,
                    width,
                    residual=False,
                    aggregation="mean",
                    mask_policy=mask_policy,
                )
            )
            edges = geometry.neighbor_maps[level].edge_set.edge_attr_dim
            self.refiners.append(
                nn.ModuleList(
                    [
                        RefinementBlock(
                            width, edges, refiner_mlp_ratio, mask_policy=mask_policy
                        )
                        for _ in range(refiners[level])
                    ]
                )
            )

    def forward(self, state, month, co2, *, node_mask=None):
        # Validity can vary between samples; unlike geography, these masks are dynamic.
        masks = [None] * len(self.film)
        if node_mask is not None:
            masks[0] = node_mask.flatten(1).to(state.dtype)
            for level, pool in enumerate(self.static_pools):
                masks[level + 1] = pool(masks[level][..., None])[..., 0]
                if level == 0 and self.mask_policy == "support":
                    # Any wet grid cell activates a finest-mesh node, matching
                    # the refactor branch. Coarser fractions weight neighbors.
                    masks[level + 1] = (masks[level + 1] > 0).to(state.dtype)

        # Missing inputs cannot contribute NaNs to convolution.
        features = torch.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        features = features.permute(0, 2, 3, 1).flatten(1, 2)
        features = self.input_projection(features)
        features = features + self.position(self.grid_xyz.to(features))[None]

        for level, convolution in enumerate(self.down):
            if not self.residual_film_only:
                features = self.film[level](features, month, co2)
            features = convolution(
                features,
                None,
                self.geometry.down_maps[level],
                source_mask=masks[level],
                target_mask=masks[level + 1],
            )
            film = self.film[level + 1]
            modulation = None
            if self.residual_film_only and self.refiners[level]:
                modulation = film.modulation(month, co2)
            for block in self.refiners[level]:
                conditioned = film(features, month, co2, modulation=modulation)
                features = block(
                    features,
                    conditioned,
                    self.geometry.neighbor_maps[level],
                    masks[level + 1],
                )
        return self.latent_projection(features)


class Decoder(nn.Module):
    """Latent channels -> coarse mesh -> fine mesh -> physical grid."""

    def __init__(
        self,
        channels,
        geometry,
        grid_shape,
        *,
        statics=None,
        width=102,
        latent_channels=4,
        refiners=(6, 6, 4),
        co2_film_levels=(),
        condition_on_co2=True,
        calendar="monthly",
        refiner_mlp_ratio=2,
        residual_film_only=True,
        film=None,
        mask_policy="support",
    ):
        super().__init__()
        if len(refiners) != len(geometry.meshes):
            raise ValueError("Supply one refiner count for each mesh")
        if refiner_mlp_ratio < 0 or (0 < refiner_mlp_ratio < 1 / width):
            raise ValueError("MLP width must be positive, or ratio zero for an old AE")
        self.geometry = geometry
        self.residual_film_only = residual_film_only
        self.mask_policy = mask_policy
        self.grid_shape, self.channels = tuple(grid_shape), channels
        self.co2_film_levels = tuple(co2_film_levels)
        self.condition_on_co2 = bool(condition_on_co2)
        self.calendar = calendar
        n_levels = len(geometry.meshes) + 1
        if any(i < 0 or i >= n_levels for i in co2_film_levels):
            raise ValueError("CO2 FiLM level is outside the hierarchy")
        # Sharing encoder.film also reuses its pooled geography; no second pooling.
        self.film = film if film is not None else nn.ModuleList()
        self.static_pools = nn.ModuleList([MeanPool(m) for m in geometry.down_maps])
        if film is None:
            if statics is None or tuple(statics.shape) != (2, *self.grid_shape):
                raise ValueError(
                    "Supply statics with shape [2, latitude, longitude], or share encoder.film"
                )
            fields = statics.detach().cpu().permute(1, 2, 0).flatten(0, 1)[None]
            for level in range(n_levels):
                if level > 0:
                    fields = self.static_pools[level - 1](fields)
                self.film.append(
                    FiLM(
                        width,
                        fields,
                        calendar=calendar,
                        include_co2=condition_on_co2
                        and (not co2_film_levels or level in co2_film_levels),
                    )
                )
        self.latent_expand = nn.Linear(latent_channels, width)
        self.output_projection = nn.Linear(width, channels)
        self.up = nn.ModuleList()
        self.refiners = nn.ModuleList()
        for level in reversed(range(len(geometry.meshes))):
            mapping = geometry.up_maps[level]
            self.up.append(
                PaddedECC(
                    width,
                    width,
                    mapping.edge_set.edge_attr_dim,
                    width,
                    residual=False,
                    aggregation="mean",
                    mask_policy=mask_policy,
                )
            )
            edges = geometry.neighbor_maps[level].edge_set.edge_attr_dim
            self.refiners.append(
                nn.ModuleList(
                    [
                        RefinementBlock(
                            width, edges, refiner_mlp_ratio, mask_policy=mask_policy
                        )
                        for _ in range(refiners[level])
                    ]
                )
            )

    def forward(self, latent, month, co2, *, node_mask=None):
        masks = [None] * len(self.film)
        if node_mask is not None:
            masks[0] = node_mask.flatten(1).to(latent.dtype)
            for level, pool in enumerate(self.static_pools):
                masks[level + 1] = pool(masks[level][..., None])[..., 0]
                if level == 0 and self.mask_policy == "support":
                    # Use exactly the same wet support as the encoder.
                    masks[level + 1] = (masks[level + 1] > 0).to(latent.dtype)

        features = self.latent_expand(latent)
        for step, convolution in enumerate(self.up):
            level = len(self.up) - step - 1  # walk from coarsest mesh to finest
            film = self.film[level + 1]
            modulation = None
            if self.residual_film_only and self.refiners[step]:
                modulation = film.modulation(month, co2)
            for block in self.refiners[step]:
                conditioned = film(features, month, co2, modulation=modulation)
                features = block(
                    features,
                    conditioned,
                    self.geometry.neighbor_maps[level],
                    masks[level + 1],
                )
            if not self.residual_film_only:
                features = film(features, month, co2)
            features = convolution(
                features,
                None,
                self.geometry.up_maps[level],
                source_mask=masks[level + 1],
                target_mask=masks[level],
            )
        if not self.residual_film_only:
            features = self.film[0](features, month, co2)
        fields = self.output_projection(features)
        return fields.reshape(latent.shape[0], *self.grid_shape, self.channels).permute(
            0, 3, 1, 2
        )


class Autoencoder(nn.Module):
    """An encoder and decoder with shared FiLM and separate spatial/MLP weights."""

    def __init__(
        self,
        channels,
        geometry,
        grid_shape,
        *,
        statics,
        preprocessor=None,
        latent_normalizer=None,
        width=102,
        latent_channels=4,
        refiners=(6, 6, 4),
        co2_film_levels=(),
        condition_on_co2=True,
        calendar="monthly",
        refiner_mlp_ratio=2,
        residual_film_only=True,
        mask_aware=False,
        mask_policy="support",
    ):
        super().__init__()
        self.preprocessor = preprocessor
        self.latent_normalizer = (
            latent_normalizer if latent_normalizer is not None else LatentNormalizer()
        )
        if preprocessor is not None:
            if (
                len(preprocessor.variables) != channels
                or len({v.component for v in preprocessor.variables}) != 1
            ):
                raise ValueError(
                    "AE preprocessing must describe exactly this component's channels"
                )
        self.encoder = Encoder(
            channels,
            geometry,
            statics=statics,
            width=width,
            latent_channels=latent_channels,
            refiners=refiners,
            co2_film_levels=co2_film_levels,
            condition_on_co2=condition_on_co2,
            calendar=calendar,
            refiner_mlp_ratio=refiner_mlp_ratio,
            residual_film_only=residual_film_only,
            mask_policy=mask_policy,
        )
        self.decoder = Decoder(
            channels,
            geometry,
            grid_shape,
            width=width,
            latent_channels=latent_channels,
            refiners=refiners,
            co2_film_levels=co2_film_levels,
            condition_on_co2=condition_on_co2,
            calendar=calendar,
            refiner_mlp_ratio=refiner_mlp_ratio,
            residual_film_only=residual_film_only,
            film=self.encoder.film,
            mask_policy=mask_policy,
        )
        self.mask_aware = bool(mask_aware)

    def encode(self, state, month, co2, *, node_mask=None):
        if self.mask_aware and node_mask is None:
            node_mask = torch.isfinite(state[:, 0])
        return self.encoder(state, month, co2, node_mask=node_mask)

    def decode(self, latent, month, co2, *, node_mask=None):
        return self.decoder(latent, month, co2, node_mask=node_mask)

    def forward(self, state, month, co2, *, node_mask=None):
        if self.mask_aware and node_mask is None:
            node_mask = torch.isfinite(state[:, 0])
        latent = self.encode(state, month, co2, node_mask=node_mask)
        fields = self.decode(latent, month, co2, node_mask=node_mask)
        return fields, latent

    def to_physical(self, fields, *, time=None):
        """Restore a normalized batch to the selected variables' physical units."""
        if self.preprocessor is None:
            raise RuntimeError(
                "Attach fitted preprocessing when constructing or loading the AE"
            )
        component = self.preprocessor.variables[0].component
        return self.preprocessor.decode_tensor(fields, component, time=time)

    @torch.no_grad()
    def reconstruct(self, fields, *, co2=None, batch_size=4):
        """Reconstruct an xarray Dataset in physical units, using this AE's statistics."""
        import numpy as np
        import xarray as xr
        from .rollout import co2_values, calendar_features
        from .preprocessing import validate_fields, validate_units

        if self.preprocessor is None:
            raise RuntimeError(
                "Attach fitted preprocessing before reconstructing physical fields"
            )
        variables = self.preprocessor.variables
        component = variables[0].component
        names = [variable.name for variable in variables]
        validate_fields(
            fields,
            names,
            cadence="6h" if self.encoder.calendar == "day_hour" else "monthly",
        )
        validate_units(fields, variables)
        grid_lat = (
            self.encoder.geometry.grid.lat.reshape(self.decoder.grid_shape)[:, 0]
            .cpu()
            .numpy()
        )
        grid_lon = (
            self.encoder.geometry.grid.lon.reshape(self.decoder.grid_shape)[0]
            .cpu()
            .numpy()
        )
        if not np.allclose(fields.lat, grid_lat, atol=1e-5) or not np.allclose(
            fields.lon, grid_lon, atol=1e-5
        ):
            raise ValueError("Dataset coordinates differ from the AE grid")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        device = next(self.parameters()).device
        months = calendar_features(fields.time, self.encoder.calendar).to(device)
        if self.encoder.condition_on_co2:
            forcing = torch.as_tensor(
                (
                    (co2_values(co2, fields.time) - self.preprocessor.co2_mean)
                    / self.preprocessor.co2_std
                ).values,
                dtype=torch.float32,
                device=device,
            )[:, None]
        else:
            forcing = torch.zeros(len(fields.time), 1, device=device)
        batches = []
        for start in range(0, len(fields.time), batch_size):
            normalized = self.preprocessor.encode(
                fields.isel(time=slice(start, start + batch_size)), component
            ).compute()
            state = torch.as_tensor(
                np.stack(
                    [
                        normalized[name].transpose("time", "lat", "lon").values
                        for name in names
                    ],
                    axis=1,
                ),
                dtype=torch.float32,
                device=device,
            )
            prediction, _ = self(
                state,
                months[start : start + batch_size],
                forcing[start : start + batch_size],
            )
            batches.append(
                self.to_physical(
                    prediction,
                    time=fields.time.isel(time=slice(start, start + batch_size)),
                )
                .cpu()
                .numpy()
            )
        values = np.concatenate(batches)
        result = xr.Dataset(
            {
                name: (("time", "lat", "lon"), values[:, index])
                for index, name in enumerate(names)
            },
            coords={name: fields[name] for name in ("time", "lat", "lon")},
        )
        for variable in variables:
            result[variable.name].attrs = dict(
                self.preprocessor.stats[variable.name]["attrs"]
            )
            result[variable.name].attrs.setdefault("units", variable.units)
        return result

    def save(self, path):
        """Save weights, geometry, preprocessing and latent normalization together."""
        from .loading import save_model

        save_model(self, path)

    @classmethod
    def load(cls, path, *, device=None):
        from .loading import load_autoencoder

        return load_autoencoder(path, device=device)


class CoupledAutoencoder(nn.Module):
    """Two independent AEs for the historical paired-training workflow."""

    def __init__(
        self,
        atmosphere_channels,
        ocean_channels,
        geometry,
        grid_shape,
        *,
        statics,
        width=102,
        latent_channels=4,
        refiners=(6, 6, 4),
        co2_film_levels=(),
        condition_on_co2=True,
        calendar="monthly",
        ocean_mask_aware=False,
        refiner_mlp_ratio=2,
        residual_film_only=True,
        mask_policy="support",
    ):
        super().__init__()
        self.atmosphere = Autoencoder(
            atmosphere_channels,
            geometry,
            grid_shape,
            statics=statics,
            width=width,
            latent_channels=latent_channels,
            refiners=refiners,
            co2_film_levels=co2_film_levels,
            condition_on_co2=condition_on_co2,
            calendar=calendar,
            refiner_mlp_ratio=refiner_mlp_ratio,
            residual_film_only=residual_film_only,
            mask_policy=mask_policy,
        )
        self.ocean = Autoencoder(
            ocean_channels,
            geometry,
            grid_shape,
            statics=statics,
            width=width,
            latent_channels=latent_channels,
            refiners=refiners,
            co2_film_levels=co2_film_levels,
            condition_on_co2=condition_on_co2,
            calendar=calendar,
            refiner_mlp_ratio=refiner_mlp_ratio,
            residual_film_only=residual_film_only,
            mask_aware=ocean_mask_aware,
            mask_policy=mask_policy,
        )
