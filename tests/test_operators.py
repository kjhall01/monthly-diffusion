import torch
import pytest
from monthly_diffusion.mesh import (
    make_icosphere,
    build_mesh_adjacency_map,
    build_mesh_hierarchy,
)
from monthly_diffusion.ecc import BucketedECC, PaddedECC, ScatterECC
from monthly_diffusion.autoencoder import CoupledAutoencoder
from monthly_diffusion.diffusion import CoupledLatentDiffusion


@pytest.mark.parametrize("optimized", [BucketedECC, PaddedECC])
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_ecc_values_and_gradients(optimized, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    graph, faces = make_icosphere(1)
    mapping = build_mesh_adjacency_map(graph, faces, bidirectional=True)
    torch.manual_seed(20)
    ref = ScatterECC(4, 4, mapping.edge_set.edge_attr_dim, 8).to(device)
    fast = optimized(4, 4, mapping.edge_set.edge_attr_dim, 8).to(device)
    fast.load_state_dict(ref.state_dict())
    x = torch.randn(2, graph.n_nodes, 4, device=device, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    a, b = ref(x, x, mapping), fast(y, y, mapping)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    a.square().sum().backward()
    b.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=2e-5, rtol=2e-5)
    for p, q in zip(ref.parameters(), fast.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-5)


def test_canonical_parameter_counts_and_shared_film():
    # Grid size does not alter learned parameter counts; use cheap geometry.
    geometry = build_mesh_hierarchy(nlat=18, nlon=36, mesh_levels=(2, 1, 0))
    ae = CoupledAutoencoder(16, 9, geometry, (18, 36), statics=torch.zeros(2, 18, 36))
    transition = CoupledLatentDiffusion()
    assert sum(p.numel() for p in ae.parameters()) == 5647773
    assert sum(p.numel() for p in transition.atmosphere.parameters()) == 994520
    assert sum(p.numel() for p in transition.ocean.parameters()) == 994520
    assert ae.atmosphere.encoder.film is ae.atmosphere.decoder.film
    assert ae.atmosphere.encoder.film is not ae.ocean.encoder.film


@pytest.mark.parametrize("operator", [BucketedECC, PaddedECC])
def test_ecc_mask_renormalizes_supported_sources(operator):
    graph, faces = make_icosphere(1)
    mapping = build_mesh_adjacency_map(graph, faces, bidirectional=True)
    torch.manual_seed(8)
    layer = operator(2, 2, mapping.edge_set.edge_attr_dim, 4)
    source = torch.randn(1, graph.n_nodes, 2)
    target = torch.zeros(1, graph.n_nodes, 2)
    mask = torch.ones(1, graph.n_nodes)
    mask[:, ::3] = 0.0
    unmasked = layer(source, target, mapping)
    masked = layer(source, target, mapping, source_mask=mask, target_mask=mask)
    assert masked.shape == unmasked.shape
    assert torch.isfinite(masked).all()
    assert not torch.allclose(masked, unmasked)
