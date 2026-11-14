"""The channel MLP must learn on both sides without reviving masked nodes."""

import pytest
import torch
from monthly_diffusion.autoencoder import (
    Autoencoder,
    Encoder,
    Decoder,
    RefinementBlock,
    MeanPool,
)
from monthly_diffusion.mesh import build_mesh_hierarchy


@pytest.fixture
def geometry():
    return build_mesh_hierarchy(nlat=18, nlon=36, mesh_levels=(1, 0))


def test_every_encoder_and_decoder_mlp_receives_gradients(geometry):
    torch.manual_seed(42)
    model = Autoencoder(
        2, geometry, (18, 36), statics=torch.randn(2, 18, 36), width=8, refiners=(1, 1)
    )
    state = torch.randn(2, 2, 18, 36)
    prediction, _ = model(state, torch.randn(2, 2), torch.randn(2, 1))
    (prediction - state).square().mean().backward()
    for side in (model.encoder, model.decoder):
        for level in side.refiners:
            for block in level:
                for parameter in block.mlp.parameters():
                    assert parameter.grad is not None
                    assert torch.isfinite(parameter.grad).all()
                    assert torch.count_nonzero(parameter.grad) > 0


def test_channel_mlp_keeps_inactive_nodes_zero(geometry):
    mapping = geometry.neighbor_maps[0]
    block = RefinementBlock(8, mapping.edge_set.edge_attr_dim)
    # A nonzero MLP bias would create features on land without the final mask.
    with torch.no_grad():
        block.mlp[-1].bias.fill_(2)
    features = torch.randn(2, mapping.n_target, 8)
    mask = torch.ones(2, mapping.n_target)
    mask[:, ::3] = 0
    output = block(features, features, mapping, mask)
    assert torch.count_nonzero(output[mask == 0]) == 0
    assert torch.count_nonzero(output[mask != 0]) > 0


def test_standalone_encoder_and_decoder_share_only_film(geometry):
    encoder = Encoder(
        2, geometry, statics=torch.randn(2, 18, 36), width=8, refiners=(1, 1)
    )
    decoder = Decoder(
        2, geometry, (18, 36), width=8, refiners=(1, 1), film=encoder.film
    )
    state = torch.randn(1, 2, 18, 36)
    month, co2 = torch.randn(1, 2), torch.randn(1, 1)
    latent = encoder(state, month, co2)
    output = decoder(latent, month, co2)
    assert latent.shape == (1, geometry.meshes[-1].n_nodes, 4)
    assert output.shape == state.shape
    assert torch.isfinite(output).all()
    assert encoder.film is decoder.film
    assert encoder.static_pools is not decoder.static_pools


def test_statics_pool_once_but_film_and_sample_masks_still_train(geometry, monkeypatch):
    calls = []
    original_pool = MeanPool.forward

    def counted_pool(pool, values):
        calls.append(values.shape)
        return original_pool(pool, values)

    monkeypatch.setattr(MeanPool, "forward", counted_pool)
    statics = torch.randn(2, 18, 36)
    model = Autoencoder(
        2, geometry, (18, 36), statics=statics, width=8, refiners=(1, 1)
    )
    assert len(calls) == len(geometry.meshes)
    assert model.encoder.film is model.decoder.film
    assert not any("static_fields" in name for name in model.state_dict())
    cached = [film.static_fields.clone() for film in model.encoder.film]
    expected = statics.permute(1, 2, 0).flatten(0, 1)
    torch.testing.assert_close(cached[0][0], expected)
    for mapping, actual in zip(geometry.down_maps, cached[1:]):
        senders, receivers = mapping.edge_set.sender, mapping.edge_set.receiver
        expected = torch.stack(
            [
                expected[senders[receivers == node]].mean(0)
                for node in range(mapping.n_target)
            ]
        )
        torch.testing.assert_close(actual[0], expected)
    statics.add_(100)  # the model owns its fixed fields, not a view into caller data
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    state, month, co2 = torch.randn(2, 2, 18, 36), torch.randn(2, 2), torch.randn(2, 1)
    initial_weight = model.encoder.film[1].static[-1].weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad()
        prediction, _ = model(state, month, co2)
        (prediction - state).square().mean().backward()
        optimizer.step()
    assert not torch.equal(initial_weight, model.encoder.film[1].static[-1].weight)
    for film, expected in zip(model.encoder.film, cached):
        torch.testing.assert_close(film.static_fields, expected, rtol=0, atol=0)
        assert not film.static_fields.requires_grad
    assert len(calls) == len(geometry.meshes)  # no geographic pooling in either pass

    mask = torch.ones(2, 18, 36)
    mask[0, :, :18] = 0
    model(state, month, co2, node_mask=mask)
    assert len(calls) == 3 * len(geometry.meshes)  # masks pooled in encoder AND decoder
    assert all(shape[-1] == 1 for shape in calls[len(geometry.meshes) :])


def test_static_buffers_follow_dtype_and_standalone_decoder_matches(geometry):
    statics = torch.randn(2, 18, 36)
    model = Autoencoder(
        2, geometry, (18, 36), statics=statics, width=8, refiners=(1, 1)
    ).double()
    decoder = Decoder(
        2, geometry, (18, 36), statics=statics, width=8, refiners=(1, 1)
    ).double()
    decoder.load_state_dict(model.decoder.state_dict())
    latent = torch.randn(2, geometry.meshes[-1].n_nodes, 4, dtype=torch.float64)
    month, co2 = (
        torch.randn(2, 2, dtype=torch.float64),
        torch.randn(2, 1, dtype=torch.float64),
    )
    torch.testing.assert_close(
        decoder(latent, month, co2), model.decode(latent, month, co2), rtol=0, atol=0
    )
    assert all(film.static_fields.dtype == torch.float64 for film in model.encoder.film)
