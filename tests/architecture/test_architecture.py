"""Small CPU tests; no climate data, weights, downloads or training required."""
import io
import pytest
import torch
from monthly_diffusion.architecture import (
    AutoencoderConfig, MeshHierarchyAutoencoder, DenoiserConfig, SphericalDiT,
    CoupledDenoiser, build_mesh_hierarchy_context,
)
from monthly_diffusion.architecture.geometry import (
    make_icosphere, build_mesh_adjacency_map, build_radius_map,
)
from monthly_diffusion.architecture.layers import EdgeConditionedConv, ReferenceECC

torch.set_num_threads(1)

@pytest.fixture(scope='module')
def geometry():
    return build_mesh_hierarchy_context(nlat=8, nlon=16, mesh_levels=(1,0),
                                       radius_query_fraction_edge_length=1.0)

@pytest.mark.parametrize('aggregation', ['mean', 'sum'])
def test_ecc_matches_reference_outputs_and_gradients(aggregation):
    mesh, faces = make_icosphere(0)
    edges = build_mesh_adjacency_map(mesh, faces)
    kwargs = dict(in_dim=3, out_dim=3, edge_attr_dim=edges.edge_set.edge_attr_dim,
                  hidden_dim=5, aggregation=aggregation, residual=False)
    ref, fast = ReferenceECC(**kwargs), EdgeConditionedConv(**kwargs)
    fast.load_state_dict(ref.state_dict())
    x = torch.randn(2, mesh.n_nodes, 3, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    a, b = ref(x,None,edges), fast(y,None,edges)
    torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
    a.square().sum().backward(); b.square().sum().backward()
    torch.testing.assert_close(x.grad,y.grad,rtol=1e-4,atol=1e-6)
    for p,q in zip(ref.parameters(),fast.parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=1e-5)


def test_hierarchy_direction_and_coverage(geometry):
    for edge_map in geometry.maps.values():
        e = edge_map.edge_set
        assert e.sender.min() >= 0 and e.sender.max() < edge_map.n_source
        assert e.receiver.min() >= 0 and e.receiver.max() < edge_map.n_target
        assert torch.bincount(e.receiver,minlength=edge_map.n_target).min() > 0


def inputs():
    return (torch.randn(2,2,8,16),torch.randn(2,2,8,16),
            torch.tensor([[0.,1.],[1.,0.]]),torch.tensor([[0.],[1.]]))

@pytest.mark.parametrize('shared',[False,True])
def test_autoencoder_forward_backward_and_roundtrip(geometry,shared):
    cfg=AutoencoderConfig(width=8,latent_channels=2,mesh_levels=(1,0),refiners=(1,1))
    model=MeshHierarchyAutoencoder(2,geometry,(8,16),cfg,share_conditioning=shared)
    x,statics,month,co2=inputs()
    pred,z=model(x,statics,month,co2)
    assert pred.shape==x.shape and z.shape==(2,12,2)
    torch.testing.assert_close(pred,model.decode(model.encode(x,statics,month,co2),statics,month,co2))
    assert (model.encoder.context is model.decoder.context)==shared
    if not shared:
        for p,q in zip(model.encoder.context.parameters(),model.decoder.context.parameters()):
            assert p.data_ptr()!=q.data_ptr()
            torch.testing.assert_close(p,q)
    pred.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
    copy=MeshHierarchyAutoencoder(2,geometry,(8,16),cfg,share_conditioning=shared)
    copy.load_state_dict(torch.load(buffer,weights_only=True))
    torch.testing.assert_close(copy(x,statics,month,co2)[0],pred)


def test_frozen_encoder_and_masked_ocean(geometry):
    cfg=AutoencoderConfig(width=8,latent_channels=2,mesh_levels=(1,0),refiners=(1,1))
    model=MeshHierarchyAutoencoder(2,geometry,(8,16),cfg,mask_aware=True)
    mask=torch.ones(8,16);mask[:2]=0
    model.set_ocean_support(mask)
    model.encoder.requires_grad_(False)
    before={k:v.clone() for k,v in model.encoder.state_dict().items()}
    x,s,m,c=inputs();x[:,:,:2]=float('nan')
    pred,z=model(x,s,m,c)
    assert torch.isfinite(pred).all() and torch.isfinite(z).all()
    opt=torch.optim.Adam(model.decoder.parameters(),lr=.001)
    pred.square().mean().backward();opt.step()
    for k,v in model.encoder.state_dict().items(): torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    assert any(p.grad is not None for p in model.decoder.parameters())


def test_denoiser_and_explicit_coupling():
    cfg=DenoiserConfig(latent_channels=2,n_latent_tokens=12,condition_channels=4,
                       width=8,depth=1,n_heads=2,n_forcings=2,time_embedding_channels=8)
    pair=CoupledDenoiser(SphericalDiT(cfg),SphericalDiT(cfg))
    x=torch.randn(2,12,2);context=torch.randn(2,12,4);condition=torch.randn(2,2)
    a,o=pair(x,x,atmosphere_context=context,ocean_context=context,
             condition=condition,time=torch.tensor([.2,.8]))
    assert a.shape==x.shape and o.shape==x.shape
    (a.square().mean()+o.square().mean()).backward()
    assert not hasattr(pair.atmosphere,'schedule')
    with pytest.raises(ValueError): pair.atmosphere(x,condition,context,None)
