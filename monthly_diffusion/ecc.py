"""Edge-conditioned convolution (ECC), with two production execution layouts.

BucketedECC groups destination nodes by neighbor count before padding.
PaddedECC pads all destinations to the largest neighbor count in the map.
ScatterECC is the simple edge-list reference used by numerical tests.

All three use the same learned projections, geometric kernel and aggregation:
    out_proj(mean_i(kernel(edge_ij) * source_proj(x_i)))
with an optional target residual. Only the execution layout differs.
Floating-point reduction order can produce small numerical differences.

Fixed map layouts are cached by device and dtype. Do not mutate maps in place;
use clear_cache() when explicitly replacing cached geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from monthly_diffusion._geometry.graph import SphericalMap
from monthly_diffusion._ecc_reference import ScatterECC
from monthly_diffusion._ecc_padded import PaddedECC

__all__ = ["BucketedECC", "PaddedECC", "ScatterECC"]


@dataclass(frozen=True)
class _Bucket:
    sender: torch.Tensor
    edge_attr: torch.Tensor
    valid: torch.Tensor
    inverse_degree: torch.Tensor


@dataclass(frozen=True)
class _Layout:
    # Retain the object so a recycled Python id cannot hit a stale cache entry.
    graph_map: SphericalMap
    buckets: tuple[_Bucket, ...]
    restore_order: torch.Tensor
    real_edges: int
    global_padded_slots: int
    bucketed_slots: int
    bucket_shapes: tuple[tuple[int, int], ...]


def _build_layout(
    graph_map: SphericalMap, device: torch.device, dtype: torch.dtype
) -> _Layout:
    edges = graph_map.edge_set
    sender, receiver = edges.sender.cpu(), edges.receiver.cpu()
    attributes = edges.edge_attr.cpu()
    n_target = graph_map.n_target
    counts = torch.bincount(receiver, minlength=n_target)
    if n_target == 0:
        raise ValueError("Bucketed ECC requires at least one target node.")

    # Stable ordering preserves the original COO edge order within each target.
    order = torch.argsort(receiver, stable=True)
    sorted_receiver = receiver[order]
    starts = counts.cumsum(0) - counts
    slots = torch.arange(sender.numel()) - starts[sorted_receiver]
    capacities = torch.tensor(
        [
            0 if degree == 0 else 1 << (degree - 1).bit_length()
            for degree in counts.tolist()
        ],
        dtype=torch.long,
    )
    buckets, target_order, shapes = [], [], []
    for capacity in capacities.unique(sorted=True).tolist():
        targets = torch.nonzero(capacities == capacity).flatten()
        width = int(counts[targets].max())
        local_index = torch.full((n_target,), -1, dtype=torch.long)
        local_index[targets] = torch.arange(targets.numel())
        in_bucket = local_index[sorted_receiver] >= 0
        selected_edges = order[in_bucket]
        rows = local_index[receiver[selected_edges]]
        columns = slots[in_bucket]
        shape = (targets.numel(), width)
        padded_sender = torch.zeros(shape, dtype=torch.long)
        padded_attributes = torch.zeros(
            (*shape, attributes.shape[-1]), dtype=attributes.dtype
        )
        valid = torch.zeros((*shape, 1), dtype=attributes.dtype)
        padded_sender[rows, columns] = sender[selected_edges]
        padded_attributes[rows, columns] = attributes[selected_edges]
        valid[rows, columns] = 1
        buckets.append(
            _Bucket(
                sender=padded_sender.to(device),
                edge_attr=padded_attributes.to(device=device, dtype=dtype),
                valid=valid.to(device=device, dtype=dtype),
                inverse_degree=counts[targets]
                .clamp_min(1)
                .to(device=device, dtype=dtype)
                .reciprocal()
                .view(1, -1, 1),
            )
        )
        target_order.append(targets)
        shapes.append(shape)
    restore = torch.argsort(torch.cat(target_order)).to(device)
    return _Layout(
        graph_map=graph_map,
        buckets=tuple(buckets),
        restore_order=restore,
        real_edges=edges.n_edges,
        global_padded_slots=n_target * int(counts.max()),
        bucketed_slots=sum(n * k for n, k in shapes),
        bucket_shapes=tuple(shapes),
    )


class BucketedECC(ScatterECC):
    """ECC padded separately for groups with similar incoming neighbor counts.

    Learned layers are inherited from ScatterECC; only aggregation is replaced.
    Each group needs less than twice as many slots as its real positive-degree
    edges. Maps with similar degrees use one group and need no reordering.
    """

    def __init__(self, *args, mask_policy="support", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if mask_policy not in {"support", "legacy"}:
            raise ValueError("mask_policy must be 'support' or 'legacy'")
        self.mask_policy = mask_policy
        if self.aggregation not in {"sum", "mean"}:
            raise ValueError("aggregation must be 'sum' or 'mean'.")
        if self.kernel_channels not in {"scalar", "per_channel"}:
            raise ValueError("kernel_channels must be 'scalar' or 'per_channel'.")
        self._bucket_cache: dict[tuple[int, torch.device, torch.dtype], _Layout] = {}

    def _layout(
        self, graph_map: SphericalMap, device: torch.device, dtype: torch.dtype
    ) -> _Layout:
        key = (id(graph_map), torch.device(device), dtype)
        if key not in self._bucket_cache:
            self._bucket_cache[key] = _build_layout(graph_map, device, dtype)
        return self._bucket_cache[key]

    def layout_stats(self, graph_map: SphericalMap) -> dict:
        """Return CPU layout counts; exclude learned activations and autograd."""
        layout = self._layout(graph_map, torch.device("cpu"), torch.float32)
        return {
            "real_edges": layout.real_edges,
            "global_padded_slots": layout.global_padded_slots,
            "bucketed_slots": layout.bucketed_slots,
            "bucket_shapes": [list(shape) for shape in layout.bucket_shapes],
        }

    def forward(
        self,
        source_nodes,
        target_nodes,
        graph_map: SphericalMap,
        *,
        source_mask=None,
        target_mask=None,
    ):
        if source_nodes.ndim != 3:
            raise ValueError("source_nodes must have shape [batch, source, channel].")
        batch, n_source, channels = source_nodes.shape
        if channels != self.in_dim or n_source != graph_map.n_source:
            raise ValueError("Source shape does not match ECC or graph map.")
        if graph_map.edge_set.edge_attr_dim != self.edge_attr_dim:
            raise ValueError("Edge feature dimension does not match ECC.")
        if target_nodes is not None and target_nodes.shape != (
            batch,
            graph_map.n_target,
            self.out_dim,
        ):
            raise ValueError("Target shape does not match ECC or graph map.")
        if self.residual and target_nodes is None:
            raise ValueError("target_nodes is required when residual=True.")

        def node_mask(mask, expected):
            if mask is None:
                return None
            if mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask[..., 0]
            if mask.shape != (batch, expected):
                raise ValueError(
                    f"ECC masks must have shape {(batch, expected)}, got {tuple(mask.shape)}"
                )
            return mask.to(device=source_nodes.device, dtype=source_nodes.dtype).clamp(
                0.0, 1.0
            )

        source_mask = node_mask(source_mask, n_source)
        target_mask = node_mask(target_mask, graph_map.n_target)
        if target_mask is not None and self.mask_policy == "support":
            # Match PaddedECC: fractions weight sources, not target amplitudes.
            target_mask = (target_mask > 0).to(source_nodes.dtype)

        layout = self._layout(graph_map, source_nodes.device, source_nodes.dtype)
        projected = self.source_proj(source_nodes)
        parts = []
        for bucket in layout.buckets:
            if bucket.sender.shape[1] == 0:
                # Keep the zero-gradient connection to source/kernel parameters,
                # even for an entirely empty graph, like the COO implementation.
                empty_kernels = self.kernel_mlp(bucket.edge_attr)
                part = projected.new_zeros(
                    (batch, bucket.sender.shape[0], self.out_dim)
                )
                part = part + projected.sum() * 0 + empty_kernels.sum() * 0
            else:
                kernels = self.kernel_mlp(bucket.edge_attr)
                gathered = projected[:, bucket.sender, :]
                edge_weight = bucket.valid.unsqueeze(0)
                if source_mask is not None:
                    edge_weight = edge_weight * source_mask[:, bucket.sender, None]
                part = (gathered * kernels.unsqueeze(0) * edge_weight).sum(2)
                if self.aggregation == "mean":
                    if source_mask is None:
                        part = part * bucket.inverse_degree
                    else:
                        total_weight = edge_weight.sum(2)
                        if self.mask_policy == "legacy":
                            total_weight = total_weight.clamp_min(1.0)
                        else:
                            total_weight = total_weight.masked_fill(
                                total_weight == 0, 1.0
                            )
                        part = part / total_weight
            parts.append(part)
        # One band contains all targets in their original order; regular maps
        # need neither concatenation nor an extra permutation gather.
        aggregated = (
            parts[0]
            if len(parts) == 1
            else torch.cat(parts, dim=1).index_select(1, layout.restore_order)
        )
        update = self.out_proj(aggregated)
        if target_mask is not None:
            update = update * target_mask[..., None]
        if self.residual:
            if target_mask is None:
                return target_nodes + update
            return target_nodes * target_mask[..., None] + update
        return update

    def clear_cache(self) -> None:
        super().clear_cache()
        self._bucket_cache.clear()

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self.clear_cache()
        return result
