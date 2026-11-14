"""Protect the raw feature path and the per-forward conditioning reuse."""

from copy import deepcopy

import pytest
import torch

from monthly_diffusion.autoencoder import Autoencoder, FiLM, RefinementBlock
from monthly_diffusion.mesh import build_mesh_hierarchy


@pytest.fixture
def geometry():
    return build_mesh_hierarchy(nlat=18, nlon=36, mesh_levels=(1, 0))


def test_refinement_preserves_identity_and_gradient(geometry):
    mapping = geometry.neighbor_maps[0]
    block = RefinementBlock(8, mapping.edge_set.edge_attr_dim).double()
    film = FiLM(8, torch.randn(1, mapping.n_source, 2)).double()
    with torch.no_grad():
        for parameter in film.parameters():
            parameter.normal_()
        block.spatial.out_proj.weight.zero_()
        block.spatial.out_proj.bias.zero_()
        block.mlp[-1].weight.zero_()
        block.mlp[-1].bias.zero_()
    features = torch.randn(
        2, mapping.n_source, 8, dtype=torch.float64, requires_grad=True
    )
    month, co2 = torch.randn(2, 2).double(), torch.randn(2, 1).double()
    actual = block(features, film(features, month, co2), mapping)
    torch.testing.assert_close(actual, features, atol=0, rtol=0)
    actual.sum().backward()
    torch.testing.assert_close(features.grad, torch.ones_like(features), atol=0, rtol=0)


@pytest.mark.parametrize("masked", [False, True])
def test_full_network_is_affine_when_residual_updates_are_zero(geometry, masked):
    model = Autoencoder(
        2,
        geometry,
        (18, 36),
        statics=torch.randn(2, 18, 36),
        width=2,
        latent_channels=2,
        refiners=(1, 1),
    ).double()
    # Make transfers simple neighbor averages. This avoids a tiny random gain
    # hiding an accidental normalization in an approximate equality check.
    with torch.no_grad():
        for parameter in model.encoder.position.parameters():
            parameter.zero_()
        for side in (model.encoder, model.decoder):
            for level in side.refiners:
                for block in level:
                    block.spatial.out_proj.weight.zero_()
                    block.spatial.out_proj.bias.zero_()
                    block.mlp[-1].weight.zero_()
                    block.mlp[-1].bias.zero_()
        projections = [
            model.encoder.input_projection,
            model.encoder.latent_projection,
            model.decoder.latent_expand,
            model.decoder.output_projection,
        ]
        for transfer in list(model.encoder.down) + list(model.decoder.up):
            projections.extend([transfer.source_proj, transfer.out_proj])
            transfer.kernel_mlp.net[-1].weight.zero_()
            transfer.kernel_mlp.net[-1].bias.fill_(1)
        for projection in projections:
            projection.weight.copy_(torch.eye(2))
            projection.bias.fill_(0.2)
        for film in model.encoder.film:
            for parameter in film.parameters():
                parameter.normal_()
    state = torch.full((2, 2, 18, 36), 3.0, dtype=torch.float64, requires_grad=True)
    month, co2 = torch.randn(2, 2).double(), torch.randn(2, 1).double()
    mask = None
    if masked:
        mask = torch.ones(2, 18, 36, dtype=torch.float64)
        mask[:, :, :10] = 0
    baseline, _ = model(torch.zeros_like(state), month, co2, node_mask=mask)
    output, _ = model(state, month, co2, node_mask=mask)
    doubled, _ = model(2 * state, month + 4, co2 - 3, node_mask=mask)
    torch.testing.assert_close(
        doubled - baseline, 2 * (output - baseline), rtol=1e-12, atol=1e-12
    )
    assert (output - baseline).abs().max() > 1
    output.sum().backward()
    assert torch.isfinite(state.grad).all() and state.grad.abs().max() > 0.1


@pytest.mark.parametrize("include_co2", [False, True])
def test_reused_modulation_preserves_values_and_gradients(include_co2):
    torch.manual_seed(19)
    repeated = FiLM(8, torch.randn(1, 11, 2), include_co2=include_co2).double()
    with torch.no_grad():
        for parameter in repeated.parameters():
            parameter.normal_(std=0.2)
    reused = deepcopy(repeated)
    original = torch.randn(3, 11, 8, dtype=torch.float64, requires_grad=True)
    cached = original.detach().clone().requires_grad_()
    month, co2 = torch.randn(3, 2).double(), torch.randn(3, 1).double()
    terms = reused.modulation(month, co2)
    expected = original
    actual = cached
    for _ in range(3):
        expected = expected + repeated(expected, month, co2)
        actual = actual + reused(actual, month, co2, modulation=terms)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(cached.grad, original.grad, rtol=1e-11, atol=1e-12)
    for a, b in zip(repeated.parameters(), reused.parameters()):
        if a.grad is None:
            assert b.grad is None
        else:
            torch.testing.assert_close(b.grad, a.grad, rtol=1e-11, atol=1e-12)


def test_conditioning_projects_once_per_level_per_pass(geometry):
    model = Autoencoder(
        2,
        geometry,
        (18, 36),
        statics=torch.randn(2, 18, 36),
        width=8,
        refiners=(2, 3),
    )
    calls = []
    hooks = []
    for level, film in enumerate(model.encoder.film):
        for name in ("static", "month", "co2_shift"):
            hooks.append(
                getattr(film, name).register_forward_pre_hook(
                    lambda module, args, level=level, name=name: calls.append(
                        (level, name, args[0].shape[0])
                    )
                )
            )
    model(torch.randn(3, 2, 18, 36), torch.randn(3, 2), torch.randn(3, 1))
    assert not any(level == 0 for level, _, _ in calls)
    for level in (1, 2):
        assert calls.count((level, "static", 1)) == 2  # encoder and decoder
        assert calls.count((level, "month", 3)) == 2
        assert calls.count((level, "co2_shift", 3)) == 2
    for hook in hooks:
        hook.remove()


@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_remembers_path_and_missing_flag_means_legacy(
    geometry, tmp_path, legacy
):
    model = Autoencoder(
        2,
        geometry,
        (18, 36),
        statics=torch.randn(2, 18, 36),
        width=8,
        refiners=(1, 1),
        residual_film_only=not legacy,
    )
    path = tmp_path / "autoencoder.pt"
    model.save(path)
    if legacy:
        saved = torch.load(path, weights_only=False)
        saved["parameters"].pop("residual_film_only")
        torch.save(saved, path)
    restored = Autoencoder.load(path, device="cpu")
    assert restored.encoder.residual_film_only is (not legacy)
    state, month, co2 = torch.randn(2, 2, 18, 36), torch.randn(2, 2), torch.randn(2, 1)
    expected, latent = model(state, month, co2)
    actual, restored_latent = restored(state, month, co2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(restored_latent, latent, rtol=0, atol=0)


@pytest.mark.parametrize("condition_on_co2", [False, True])
def test_co2_switch_controls_encoder_decoder_and_survives_loading(
    geometry, tmp_path, condition_on_co2
):
    torch.manual_seed(57)
    model = Autoencoder(
        2, geometry, (18, 36), statics=torch.randn(2, 18, 36),
        width=8, latent_channels=4, refiners=(1, 1), mask_aware=True,
        condition_on_co2=condition_on_co2,
    ).eval()
    # Learned nonzero conditioning: zero-initialized CO2 heads would hide bugs.
    with torch.no_grad():
        for film in model.encoder.film:
            for parameter in film.parameters():
                parameter.normal_(std=0.2)
    state = torch.randn(2, 2, 18, 36)
    state[:, :, :, :5] = float("nan")
    mask = torch.isfinite(state[:, 0])
    month = torch.randn(2, 2)
    low, high = torch.full((2, 1), -4.0), torch.full((2, 1), 30.0)
    latent = model.encode(state, month, low)
    changed_latent = model.encode(state, month, high)
    decoded = model.decode(latent, month, low, node_mask=mask)
    changed_decoded = model.decode(latent, month, high, node_mask=mask)
    if condition_on_co2:
        assert not torch.equal(latent, changed_latent)
        assert not torch.equal(decoded, changed_decoded)
    else:
        torch.testing.assert_close(latent, changed_latent, rtol=0, atol=0)
        torch.testing.assert_close(decoded, changed_decoded, rtol=0, atol=0)
        # CO2 is not even evaluated: invalid values cannot leak into predictions.
        output, _ = model(state, month, torch.full_like(low, float("nan")))
        torch.testing.assert_close(output, decoded, rtol=0, atol=0)
        assert not torch.equal(latent, model.encode(state, month + 1, low))

    path = tmp_path / "ae.pt"
    model.save(path)
    if condition_on_co2:
        # Old checkpoints have no switch and must retain their CO2 response.
        saved = torch.load(path, weights_only=False)
        saved["parameters"].pop("condition_on_co2")
        torch.save(saved, path)
    restored = Autoencoder.load(path, device="cpu")
    assert restored.encoder.condition_on_co2 is condition_on_co2
    assert restored.decoder.condition_on_co2 is condition_on_co2
    torch.testing.assert_close(restored.encode(state, month, high), changed_latent, rtol=0, atol=0)
    torch.testing.assert_close(
        restored.decode(latent, month, high, node_mask=mask), changed_decoded, rtol=0, atol=0
    )
