"""Mesh-hierarchy autoencoders, assembled from ECCs and shared conditioners.

Tensors are [batch, channel, latitude, longitude] at the physical boundary and
[batch, node, channel] on meshes. Encoder and decoder share context modules, not
ECC weights. Missing physical values are zero-filled only at the encoder input.
"""

from dataclasses import dataclass

import torch
from torch import nn

from .layers.ecc import EdgeConditionedConv, PaddedECC
from .layers.conditioning import ConditionalPreActivation
from .layers.joint_conditioning import JointConditionalPreActivation
from .layers.contextual_conditioning import ContextualCO2PreActivation, XYZContextualCO2PreActivation
from .layers.embeddings import PositionEmbedding


def _propagate_node_mask(graph_map, source_mask):
    """Floating-point mean of source support over incoming geometric edges."""
    source_mask = source_mask.to(dtype=torch.float32)
    # Geometry is cached on CPU; training masks may already be on MPS/CUDA.
    # Move only the small integer index vectors, not the graph itself.
    sender = graph_map.edge_set.sender.to(source_mask.device)
    receiver = graph_map.edge_set.receiver.to(source_mask.device)
    values = source_mask.new_zeros(source_mask.shape[0], graph_map.n_target)
    counts = source_mask.new_zeros(source_mask.shape[0], graph_map.n_target)
    values.scatter_add_(1, receiver[None].expand(source_mask.shape[0], -1), source_mask[:, sender])
    counts.scatter_add_(
        1,
        receiver[None].expand(source_mask.shape[0], -1),
        source_mask.new_ones((source_mask.shape[0], sender.numel())),
    )
    return values / counts.clamp_min(1.0)


def _hierarchy_masks(geometry, grid_mask):
    """Binary grid -> any-wet finest mesh -> mean support on coarser meshes.

    This explicitly reproduces the published models' training-time operator.
    Boolean scatter accumulation previously made the first step an any-wet
    operation, whereas float input accidentally selected a different operator.
    The rule here is independent of the input mask's dtype.
    """
    if not torch.all((grid_mask == 0) | (grid_mask == 1)):
        raise ValueError("Grid support must be binary (Boolean or zero/one)")
    masks = [grid_mask.to(dtype=torch.float32)]
    levels = ["grid", *geometry.metadata["mesh_level_names"]]
    for i, (source, target) in enumerate(zip(levels[:-1], levels[1:])):
        support = _propagate_node_mask(geometry.maps[f"{source}_to_{target}"], masks[-1])
        masks.append((support > 0).float() if i == 0 else support)
    return masks


@dataclass(frozen=True)
class AutoencoderConfig:
    width: int = 102
    latent_channels: int = 4
    mesh_levels: tuple[int, ...] = (4, 3, 2)
    refiners: tuple[int, ...] = (6, 6, 4)
    rotation: tuple[float, float, float] = (17.0, 23.0, 11.0)
    # Level zero is the physical grid; higher indices are progressively
    # coarser meshes, ending at the latent mesh. Empty selects every level
    # present in the supplied geometry.
    co2_film_levels: tuple[int, ...] = ()
    conditioner: str = "factorized"
    position_embedding: bool = True
    static_channels: int = 2


class StaticPool(nn.Module):
    """Fixed map means; keep indices/counts on-device, not copied each forward."""

    def __init__(self, graph_map):
        super().__init__()
        self.n_target = graph_map.n_target
        self.register_buffer("sender", graph_map.edge_set.sender.clone())
        self.register_buffer("receiver", graph_map.edge_set.receiver.clone())
        self.register_buffer(
            "count", torch.bincount(self.receiver, minlength=self.n_target).float().clamp_min(1)
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

    ocean_mask_policy = "wet_any_then_mean_v1"

    def __init__(
        self,
        geometry,
        width,
        co2_film_levels=(),
        *,
        conditioner="factorized",
        position_embedding=True,
        static_channels=2,
    ):
        super().__init__()
        self.geometry = geometry
        self.levels = ["grid", *geometry.metadata["mesh_level_names"]]
        selected = tuple(range(len(self.levels))) if not co2_film_levels else co2_film_levels
        self.co2_film_levels = frozenset(int(i) for i in selected)
        if not self.co2_film_levels.issubset(range(len(self.levels))):
            raise ValueError("co2_film_levels contains an unknown mesh level")
        if conditioner not in ("factorized", "joint", "contextual_co2", "contextual_xyz_co2"):
            raise ValueError(
                "conditioner must be factorized, joint, contextual_co2, or contextual_xyz_co2"
            )
        if conditioner in ("factorized", "joint") and static_channels != 2:
            raise ValueError("The factorized and joint conditioners require two static channels")
        self.conditioner = conditioner
        operator = {
            "factorized": ConditionalPreActivation,
            "joint": JointConditionalPreActivation,
            "contextual_co2": ContextualCO2PreActivation,
            "contextual_xyz_co2": XYZContextualCO2PreActivation,
        }[conditioner]
        self.conditions = nn.ModuleList(
            operator(width, static_channels)
            if conditioner in ("contextual_co2", "contextual_xyz_co2")
            else operator(width)
            for _ in self.levels
        )
        # Only the grid position head enters the ECC encoder's forward path.
        # Cross-map ECCs consume source features, not target position seeds.
        self.positions = nn.ModuleList(
            PositionEmbedding(width) for _ in self.levels if position_embedding
        )
        self.pool_maps = [
            geometry.maps[f"{a}_to_{b}"] for a, b in zip(self.levels[:-1], self.levels[1:])
        ]
        self.pools = nn.ModuleList(StaticPool(graph_map) for graph_map in self.pool_maps)
        self.register_buffer("grid_xyz", geometry.graphs["grid"].xyz)
        for i in range(len(self.levels)):
            # None for unmasked branches. Fixed ocean buffers are shared by
            # encoder/decoder, move with the module, and enter state_dict.
            self.register_buffer(f"ocean_support_{i}", None)
        if conditioner in ("joint", "contextual_xyz_co2"):
            for i, name in enumerate(self.levels):
                # Fixed geometry moves with the model, but is rebuilt on load.
                self.register_buffer(f"xyz_{i}", geometry.graphs[name].xyz, persistent=False)

    def set_ocean_support(self, grid_mask):
        """Bind train-fitted support once; never infer it from latent values."""
        mask = torch.as_tensor(grid_mask, device=self.grid_xyz.device).reshape(1, -1)
        if mask.shape[1] != self.geometry.graphs["grid"].n_nodes:
            raise ValueError("Ocean support does not match the physical grid")
        if self.ocean_support_0 is not None:
            if not torch.equal(mask.float(), self.ocean_support_0):
                raise ValueError("Saved ocean support differs from fitted preprocessing")
            return
        for i, support in enumerate(_hierarchy_masks(self.geometry, mask)):
            setattr(self, f"ocean_support_{i}", support)

    def masks(self, batch, node_mask=None):
        """Use saved buffers, or explicit per-sample masks for an atomic module."""
        if self.ocean_support_0 is not None:
            if node_mask is not None:
                mask = node_mask.reshape(batch, -1).to(self.ocean_support_0)
                if not torch.equal(mask, self.ocean_support_0.expand(batch, -1)):
                    raise ValueError("Input ocean support differs from the fixed trained mask")
            return [
                getattr(self, f"ocean_support_{i}").expand(batch, -1)
                for i in range(len(self.levels))
            ]
        if node_mask is None:
            return None
        return _hierarchy_masks(self.geometry, node_mask.reshape(batch, -1))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # Atomic modules can restore persistent masks without pre-fitting them.
        # Old training-stage checkpoints have none; keep already-bound masks.
        keys = [prefix + f"ocean_support_{i}" for i in range(len(self.levels))]
        has_saved_support = any(key in state_dict for key in keys)
        for i, key in enumerate(keys):
            if key in state_dict:
                value = state_dict[key]
                expected = (1, self.geometry.graphs[self.levels[i]].n_nodes)
                if (
                    value.shape != expected
                    or not torch.isfinite(value).all()
                    or not ((value >= 0) & (value <= 1)).all()
                ):
                    error_msgs.append(f"Invalid ocean support buffer: {key}")
                setattr(
                    self, f"ocean_support_{i}", torch.empty_like(value, device=self.grid_xyz.device)
                )
            elif has_saved_support:
                error_msgs.append(f"Incomplete ocean support hierarchy: {key}")
            elif self.ocean_support_0 is not None:
                state_dict[key] = getattr(self, f"ocean_support_{i}")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def terms(self, statics, month, co2):
        values = statics.permute(0, 2, 3, 1).flatten(1, 2)
        fields = [values]
        for pool in self.pools:
            values = pool(values)
            fields.append(values)
        if self.conditioner in ("joint", "contextual_xyz_co2"):
            return [
                c.modulation_terms(
                    s,
                    month,
                    co2,
                    xyz=getattr(self, f"xyz_{i}"),
                    include_co2=i in self.co2_film_levels,
                )
                for i, (c, s) in enumerate(zip(self.conditions, fields))
            ]
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
            else HierarchyContext(
                geometry,
                config.width,
                config.co2_film_levels,
                conditioner=config.conditioner,
                position_embedding=config.position_embedding,
                static_channels=config.static_channels,
            )
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
        masks = self.context.masks(state.shape[0], node_mask)
        nodes = (
            torch.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
            .permute(0, 2, 3, 1)
            .flatten(1, 2)
        )
        h = self.input_projection(nodes)
        if self.context.positions:
            h = h + self.context.positions[0](
                self.context.grid_xyz, h.shape[0], device=h.device, dtype=h.dtype
            )
        for i, (layer, graph_map, refine) in enumerate(zip(self.down, self.maps, self.refiners)):
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

    def __init__(self, channels, geometry, grid_shape, config=AutoencoderConfig(), *, context=None):
        super().__init__()
        self.context = (
            context
            if context is not None
            else HierarchyContext(
                geometry,
                config.width,
                config.co2_film_levels,
                conditioner=config.conditioner,
                position_embedding=config.position_embedding,
                static_channels=config.static_channels,
            )
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
                    config.width, geometry.maps[f"{source}_to_{source}"], config.refiners[i - 1]
                )
            )

    def forward(self, latent, statics, month, co2, *, terms=None, node_mask=None):
        terms = self.context.terms(statics, month, co2) if terms is None else terms
        masks = self.context.masks(latent.shape[0], node_mask)
        h = self.latent_expand(latent)
        for j, (layer, graph_map, refine) in enumerate(zip(self.up, self.maps, self.refiners)):
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
        self, channels, geometry, grid_shape, config=AutoencoderConfig(), *, mask_aware=False, share_conditioning=False
    ):
        super().__init__()
        context = HierarchyContext(
            geometry,
            config.width,
            config.co2_film_levels,
            conditioner=config.conditioner,
            position_embedding=config.position_embedding,
            static_channels=config.static_channels,
        )
        self.encoder = MeshHierarchyEncoder(channels, geometry, config, context=context)
        # Separate learned conditioning by default; geometry itself remains shared.
        decoder_context = context
        if not share_conditioning:
            decoder_context = HierarchyContext(
                geometry, config.width, config.co2_film_levels,
                conditioner=config.conditioner,
                position_embedding=config.position_embedding,
                static_channels=config.static_channels,
            )
            decoder_context.load_state_dict(context.state_dict())
        self.share_conditioning = bool(share_conditioning)
        self.decoder = MeshHierarchyDecoder(channels, geometry, grid_shape, config, context=decoder_context)
        self.mask_aware = bool(mask_aware)

    def set_ocean_support(self, grid_mask):
        """Bind training-fitted fixed support to encoder and decoder."""
        self.encoder.context.set_ocean_support(grid_mask)
        if not self.share_conditioning:
            self.decoder.context.set_ocean_support(grid_mask)

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
        decoder_terms = terms if self.share_conditioning else None
        return self.decoder(z, statics, month, co2, terms=decoder_terms, node_mask=node_mask), z
