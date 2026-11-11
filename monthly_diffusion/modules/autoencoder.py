"""Mesh-hierarchy autoencoders, assembled from ECCs and shared conditioners.

Tensors are [batch, channel, latitude, longitude] at the physical boundary and
[batch, node, channel] on meshes. Encoder and decoder share context modules, not
ECC weights. Missing physical values are zero-filled only at the encoder input.
"""

from dataclasses import dataclass

import torch
from torch import nn

from .ecc import EdgeConditionedConv, PaddedECC
from .film import ConditionalPreActivation
from .embeddings import PositionEmbedding


def _propagate_node_mask(graph_map, source_mask):
    """Area-free support propagation for the optional masked ocean branch."""
    # Geometry is cached on CPU; training masks may already be on MPS/CUDA.
    # Move only the small integer index vectors, not the graph itself.
    sender = graph_map.edge_set.sender.to(source_mask.device)
    receiver = graph_map.edge_set.receiver.to(source_mask.device)
    values = source_mask.new_zeros(source_mask.shape[0], graph_map.n_target)
    counts = source_mask.new_zeros(source_mask.shape[0], graph_map.n_target)
    values.scatter_add_(
        1, receiver[None].expand(source_mask.shape[0], -1), source_mask[:, sender]
    )
    counts.scatter_add_(
        1,
        receiver[None].expand(source_mask.shape[0], -1),
        source_mask.new_ones((source_mask.shape[0], sender.numel())),
    )
    return values / counts.clamp_min(1.0)


def _hierarchy_masks(geometry, grid_mask):
    masks = [grid_mask]
    levels = ["grid", *geometry.metadata["mesh_level_names"]]
    for source, target in zip(levels[:-1], levels[1:]):
        masks.append(
            _propagate_node_mask(geometry.maps[f"{source}_to_{target}"], masks[-1])
        )
    return masks


@dataclass(frozen=True)
class AutoencoderConfig:
    width: int = 102
    latent_channels: int = 4
    mesh_levels: tuple[int, ...] = (4, 3, 2)
    refiners: tuple[int, ...] = (6, 6, 4)
    rotation: tuple[float, float, float] = (17.0, 23.0, 11.0)
    # Level zero is the physical grid; higher indices are progressively
    # coarser meshes, ending at the latent mesh.  The default preserves MDv1.
    # Empty means every level present in the supplied geometry (the MDv1
    # default), which also keeps small test geometries valid.
    co2_film_levels: tuple[int, ...] = ()


class StaticPool(nn.Module):
    """Fixed map means; keep indices/counts on-device, not copied each forward."""

    def __init__(self, graph_map):
        super().__init__()
        self.n_target = graph_map.n_target
        self.register_buffer("sender", graph_map.edge_set.sender.clone())
        self.register_buffer("receiver", graph_map.edge_set.receiver.clone())
        self.register_buffer(
            "count",
            torch.bincount(self.receiver, minlength=self.n_target).float().clamp_min(1),
        )

    def forward(self, values):
        pooled = values.new_zeros(values.shape[0], self.n_target, values.shape[-1])
        pooled.scatter_add_(
            1,
            self.receiver[None, :, None].expand(values.shape[0], -1, values.shape[-1]),
            values[:, self.sender],
        )
        return pooled / self.count.to(values)[None, :, None]


class HierarchyContext(nn.Module):
    """Geometry and shared level-wise FiLM; one instance per component."""

    def __init__(self, geometry, width, co2_film_levels=()):
        super().__init__()
        self.geometry = geometry
        self.levels = ["grid", *geometry.metadata["mesh_level_names"]]
        selected = (
            tuple(range(len(self.levels))) if not co2_film_levels else co2_film_levels
        )
        self.co2_film_levels = frozenset(int(i) for i in selected)
        if not self.co2_film_levels.issubset(range(len(self.levels))):
            raise ValueError("co2_film_levels contains an unknown mesh level")
        self.conditions = nn.ModuleList(
            ConditionalPreActivation(width) for _ in self.levels
        )
        # Keep the inherited position heads; only the grid embedding contributes
        # to the active ECC encoder. Cross-map target seeds are NOT ECC inputs.
        self.positions = nn.ModuleList(PositionEmbedding(width) for _ in self.levels)
        self.pool_maps = [
            geometry.maps[f"{a}_to_{b}"]
            for a, b in zip(self.levels[:-1], self.levels[1:])
        ]
        self.pools = nn.ModuleList(
            StaticPool(graph_map) for graph_map in self.pool_maps
        )
        self.register_buffer("grid_xyz", geometry.graphs["grid"].xyz)

    def terms(self, statics, month, co2):
        values = statics.permute(0, 2, 3, 1).flatten(1, 2)
        fields = [values]
        for pool in self.pools:
            values = pool(values)
            fields.append(values)
        return [
            c.modulation_terms(s, month, co2, include_co2=i in self.co2_film_levels)
            for i, (c, s) in enumerate(zip(self.conditions, fields))
        ]


class SameMeshRefiners(nn.Module):
    def __init__(self, width, graph_map, depth):
        super().__init__()
        self.graph_map = graph_map
        self.layers = nn.ModuleList(
            PaddedECC(
                width,
                width,
                graph_map.edge_set.edge_attr_dim,
                width,
                residual=True,
                aggregation="mean",
            )
            for _ in range(depth)
        )

    def forward(self, h, condition, terms, *, node_mask=None):
        for layer in self.layers:
            h = layer(
                condition.apply_terms(h, terms),
                h,
                self.graph_map,
                source_mask=node_mask,
                target_mask=node_mask,
            )
        return h


class MeshHierarchyEncoder(nn.Module):
    """Physical grid -> finest mesh -> coarse latent; reusable on its own."""

    def __init__(self, channels, geometry, config=AutoencoderConfig(), *, context=None):
        super().__init__()
        self.context = (
            context
            if context is not None
            else HierarchyContext(geometry, config.width, config.co2_film_levels)
        )
        self.input_projection = nn.Linear(channels, config.width)
        self.latent_projection = nn.Linear(config.width, config.latent_channels)
        self.maps = self.context.pool_maps
        if len(config.refiners) != len(self.maps):
            raise ValueError("One refiner count is required per mesh level")
        self.down = nn.ModuleList()
        self.refiners = nn.ModuleList()
        for i, graph_map in enumerate(self.maps):
            operator = EdgeConditionedConv if i == 0 else PaddedECC
            self.down.append(
                operator(
                    config.width,
                    config.width,
                    graph_map.edge_set.edge_attr_dim,
                    config.width,
                    residual=False,
                    aggregation="mean",
                )
            )
            name = self.context.levels[i + 1]
            self.refiners.append(
                SameMeshRefiners(
                    config.width, geometry.maps[f"{name}_to_{name}"], config.refiners[i]
                )
            )

    def forward(self, state, statics, month, co2, *, terms=None, node_mask=None):
        terms = self.context.terms(statics, month, co2) if terms is None else terms
        if node_mask is not None:
            if node_mask.ndim == 4 and node_mask.shape[1] == 1:
                node_mask = node_mask[:, 0]
            if node_mask.ndim != 3:
                raise ValueError(
                    "node_mask must have shape [batch, latitude, longitude]"
                )
            masks = _hierarchy_masks(self.context.geometry, node_mask.flatten(1))
        else:
            masks = None
        nodes = (
            torch.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
            .permute(0, 2, 3, 1)
            .flatten(1, 2)
        )
        h = self.input_projection(nodes)
        h = h + self.context.positions[0](
            self.context.grid_xyz, h.shape[0], device=h.device, dtype=h.dtype
        )
        for i, (layer, graph_map, refine) in enumerate(
            zip(self.down, self.maps, self.refiners)
        ):
            h = layer(
                self.context.conditions[i].apply_terms(h, terms[i]),
                None,
                graph_map,
                source_mask=None if masks is None else masks[i],
                target_mask=None if masks is None else masks[i + 1],
            )
            h = refine(
                h,
                self.context.conditions[i + 1],
                terms[i + 1],
                node_mask=None if masks is None else masks[i + 1],
            )
        return self.latent_projection(h)


class MeshHierarchyDecoder(nn.Module):
    """Latent -> coarse refiners -> finer meshes -> physical grid."""

    def __init__(
        self,
        channels,
        geometry,
        grid_shape,
        config=AutoencoderConfig(),
        *,
        context=None,
    ):
        super().__init__()
        self.context = (
            context
            if context is not None
            else HierarchyContext(geometry, config.width, config.co2_film_levels)
        )
        self.grid_shape, self.channels = tuple(grid_shape), channels
        self.latent_expand = nn.Linear(config.latent_channels, config.width)
        self.output_projection = nn.Linear(config.width, channels)
        self.maps, self.up = [], nn.ModuleList()
        self.refiners = nn.ModuleList()
        for i in range(len(config.mesh_levels), 0, -1):
            source, target = self.context.levels[i], self.context.levels[i - 1]
            graph_map = geometry.maps[f"{source}_to_{target}"]
            self.maps.append(graph_map)
            self.up.append(
                PaddedECC(
                    config.width,
                    config.width,
                    graph_map.edge_set.edge_attr_dim,
                    config.width,
                    residual=False,
                    aggregation="mean",
                )
            )
            self.refiners.append(
                SameMeshRefiners(
                    config.width,
                    geometry.maps[f"{source}_to_{source}"],
                    config.refiners[i - 1],
                )
            )

    def forward(self, latent, statics, month, co2, *, terms=None, node_mask=None):
        terms = self.context.terms(statics, month, co2) if terms is None else terms
        if node_mask is not None:
            if node_mask.ndim == 4 and node_mask.shape[1] == 1:
                node_mask = node_mask[:, 0]
            masks = _hierarchy_masks(self.context.geometry, node_mask.flatten(1))
        else:
            masks = None
        h = self.latent_expand(latent)
        for j, (layer, graph_map, refine) in enumerate(
            zip(self.up, self.maps, self.refiners)
        ):
            i = len(self.maps) - j
            h = refine(
                h,
                self.context.conditions[i],
                terms[i],
                node_mask=None if masks is None else masks[i],
            )
            h = layer(
                self.context.conditions[i].apply_terms(h, terms[i]),
                None,
                graph_map,
                source_mask=None if masks is None else masks[i],
                target_mask=None if masks is None else masks[i - 1],
            )
        h = self.context.conditions[0].apply_terms(h, terms[0])
        return (
            self.output_projection(h)
            .reshape(latent.shape[0], *self.grid_shape, self.channels)
            .permute(0, 3, 1, 2)
        )


class MeshHierarchyAutoencoder(nn.Module):
    def __init__(
        self,
        channels,
        geometry,
        grid_shape,
        config=None,
        *,
        mask_aware=False,
        **parameters,
    ):
        super().__init__()
        if config is not None and parameters:
            raise ValueError("Supply individual AE parameters or config, not both")
        config = config or AutoencoderConfig(**parameters)
        context = HierarchyContext(geometry, config.width, config.co2_film_levels)
        self.encoder = MeshHierarchyEncoder(channels, geometry, config, context=context)
        self.decoder = MeshHierarchyDecoder(
            channels, geometry, grid_shape, config, context=context
        )
        self.mask_aware = bool(mask_aware)

    def encode(self, state, statics, month, co2, *, node_mask=None):
        if self.mask_aware and node_mask is None:
            node_mask = torch.isfinite(state[:, 0])
        return self.encoder(state, statics, month, co2, node_mask=node_mask)

    def decode(self, latent, statics, month, co2, *, node_mask=None):
        return self.decoder(latent, statics, month, co2, node_mask=node_mask)

    def forward(self, state, statics, month, co2, *, node_mask=None):
        if self.mask_aware and node_mask is None:
            node_mask = torch.isfinite(state[:, 0])
        terms = self.encoder.context.terms(statics, month, co2)
        z = self.encoder(state, statics, month, co2, terms=terms, node_mask=node_mask)
        return self.decoder(z, statics, month, co2, terms=terms, node_mask=node_mask), z


class CoupledAutoencoder(nn.Module):
    """Independent component autoencoders; coupled dynamics live elsewhere."""

    def __init__(
        self,
        atmosphere_channels,
        ocean_channels,
        geometry,
        grid_shape,
        config=AutoencoderConfig(),
        *,
        ocean_mask_aware=False,
    ):
        super().__init__()
        self.atmosphere = MeshHierarchyAutoencoder(
            atmosphere_channels, geometry, grid_shape, config
        )
        self.ocean = MeshHierarchyAutoencoder(
            ocean_channels, geometry, grid_shape, config, mask_aware=ocean_mask_aware
        )


def reconstruction_mse(prediction, target, latitudes):
    """Per-variable cosine-area MSE; normalize per sample's valid support."""
    valid = torch.isfinite(target)
    residual = torch.where(valid, prediction - torch.nan_to_num(target), 0.0)
    w = torch.cos(torch.deg2rad(latitudes.to(prediction))).clamp_min(0)[
        None, None, :, None
    ]
    denominator = (w * valid).sum((-2, -1))
    if (denominator == 0).any():
        raise ValueError("A reconstruction target has no valid cells")
    return ((w * residual.square()).sum((-2, -1)) / denominator).mean(0)
