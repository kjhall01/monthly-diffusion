"""Ocean support weights must not attenuate features or their skip gradients."""

import pytest
import torch

from monthly_diffusion.autoencoder import Autoencoder, RefinementBlock
from monthly_diffusion.ecc import BucketedECC, PaddedECC
from monthly_diffusion.mesh import build_mesh_hierarchy


@pytest.fixture(scope="module")
def geometry():
    return build_mesh_hierarchy(nlat=18, nlon=36, mesh_levels=(2, 1, 0))


@pytest.mark.parametrize("policy", ["support", "legacy"])
def test_six_refiners_preserve_wet_skip_values_and_gradients(geometry, policy):
    mapping = geometry.neighbor_maps[-1]
    block = RefinementBlock(4, mapping.edge_set.edge_attr_dim, mask_policy=policy)
    # Remove learned updates so the only operation left is the skip connection.
    with torch.no_grad():
        block.spatial.out_proj.weight.zero_()
        block.spatial.out_proj.bias.zero_()
        block.mlp[-1].weight.zero_()
        block.mlp[-1].bias.zero_()
    mask = torch.ones(1, mapping.n_target)
    mask[:, 0], mask[:, 1], mask[:, 2] = 0.0, 0.5, 0.01
    state = torch.randn(1, mapping.n_target, 4, requires_grad=True)
    actual = state
    for _ in range(6):
        actual = block(actual, actual, mapping, mask)
    gain = (mask > 0).float() if policy == "support" else mask**6
    torch.testing.assert_close(actual, state * gain[..., None], rtol=1e-6, atol=1e-7)
    actual.sum().backward()
    torch.testing.assert_close(state.grad, gain[..., None].expand_as(state))


@pytest.mark.parametrize("operator", [BucketedECC, PaddedECC])
def test_small_wet_support_preserves_a_constant_field(geometry, operator):
    mapping = geometry.neighbor_maps[-1]
    layer = operator(2, 2, mapping.edge_set.edge_attr_dim, 4, residual=False)
    with torch.no_grad():
        for projection in (layer.source_proj, layer.out_proj):
            projection.weight.copy_(torch.eye(2))
            projection.bias.zero_()
        layer.kernel_mlp.net[-1].weight.zero_()
        layer.kernel_mlp.net[-1].bias.fill_(1)
    state = (
        torch.tensor([3.0, -2.0])
        .expand(2, mapping.n_source, -1)
        .clone()
        .requires_grad_()
    )
    # Total incoming weight is less than one, but this is still a mean of 3/-2.
    source_mask = torch.full((2, mapping.n_source), 0.01)
    source_mask[1] = 0  # An all-dry sample must remain finite and zero.
    target_mask = torch.full((2, mapping.n_target), 0.25)
    target_mask[:, 0] = 0
    actual = layer(
        state, None, mapping, source_mask=source_mask, target_mask=target_mask
    )
    expected = state.clone()
    expected[:, 0], expected[1] = 0, 0
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(state.grad).all() and torch.count_nonzero(state.grad[1]) == 0
    assert all(torch.isfinite(parameter.grad).all() for parameter in layer.parameters())


def test_encoder_decoder_share_any_wet_finest_mesh_support(geometry):
    model = Autoencoder(
        2,
        geometry,
        (18, 36),
        statics=torch.zeros(2, 18, 36),
        width=4,
        refiners=(1, 1, 1),
        mask_aware=True,
    )
    state = torch.randn(2, 2, 18, 36)
    mask = torch.zeros(2, 18, 36, dtype=torch.bool)
    mask[0, 5:12, 5:17] = True
    mask[1, 2:9, 18:30] = True
    recorded = {}

    def capture(name):
        def hook(module, args, kwargs):
            recorded[name] = (
                kwargs["source_mask"].clone(),
                kwargs["target_mask"].clone(),
            )

        return hook

    hooks = []
    for side, layers in (("down", model.encoder.down), ("up", model.decoder.up)):
        for i, layer in enumerate(layers):
            hooks.append(
                layer.register_forward_pre_hook(capture((side, i)), with_kwargs=True)
            )
    month, co2 = torch.randn(2, 2), torch.randn(2, 1)
    boolean = model(state, month, co2, node_mask=mask)
    floating = model(state, month, co2, node_mask=mask.float())
    for hook in hooks:
        hook.remove()
    for a, b in zip(boolean, floating):
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    # Independently accumulate whether ANY incoming grid point is wet.
    mapping = geometry.down_maps[0]
    wet_count = torch.zeros(2, mapping.n_target)
    wet_count.scatter_add_(
        1,
        mapping.edge_set.receiver[None].expand(2, -1),
        mask.flatten(1)[:, mapping.edge_set.sender].float(),
    )
    torch.testing.assert_close(recorded["down", 0][1], (wet_count > 0).float())
    assert ((recorded["down", 1][1] > 0) & (recorded["down", 1][1] < 1)).any()
    for i in range(3):
        down_source, down_target = recorded["down", i]
        up_source, up_target = recorded["up", 2 - i]
        torch.testing.assert_close(down_source, up_target, rtol=0, atol=0)
        torch.testing.assert_close(down_target, up_source, rtol=0, atol=0)


@pytest.mark.parametrize("policy", ["support", "legacy"])
def test_mask_policy_survives_save_load_and_missing_means_legacy(
    geometry, tmp_path, policy
):
    model = Autoencoder(
        2,
        geometry,
        (18, 36),
        statics=torch.randn(2, 18, 36),
        width=4,
        refiners=(1, 1, 1),
        mask_aware=True,
        mask_policy=policy,
    )
    path = tmp_path / "ocean.pt"
    model.save(path)
    saved = torch.load(path, weights_only=False)
    assert saved["parameters"]["mask_policy"] == policy
    if policy == "legacy":
        del saved["parameters"]["mask_policy"]
        torch.save(saved, path)
    restored = Autoencoder.load(path, device="cpu")
    assert restored.encoder.mask_policy == restored.decoder.mask_policy == policy
    state = torch.randn(2, 2, 18, 36)
    state[:, :, :, :15] = float("nan")
    month, co2 = torch.randn(2, 2), torch.randn(2, 1)
    for actual, expected in zip(restored(state, month, co2), model(state, month, co2)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_all_wet_and_unmasked_outputs_are_unchanged(geometry):
    old = Autoencoder(
        2,
        geometry,
        (18, 36),
        statics=torch.randn(2, 18, 36),
        width=4,
        refiners=(1, 1, 1),
        mask_policy="legacy",
    )
    new = Autoencoder(
        2,
        geometry,
        (18, 36),
        statics=torch.zeros(2, 18, 36),
        width=4,
        refiners=(1, 1, 1),
    )
    new.load_state_dict(old.state_dict())
    # Static FiLM buffers are deliberately non-persistent; match them explicitly.
    for source, target in zip(old.encoder.film, new.encoder.film):
        target.static_fields.copy_(source.static_fields)
    state, month, co2 = torch.randn(2, 2, 18, 36), torch.randn(2, 2), torch.randn(2, 1)
    for mask in (None, torch.ones(2, 18, 36)):
        for actual, expected in zip(
            new(state, month, co2, node_mask=mask),
            old(state, month, co2, node_mask=mask),
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
