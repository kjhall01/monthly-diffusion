"""Development-only parity against the original implementation.

Set MONTHLY_DIFFUSION_SOURCE_ROOT to the parent umd checkout. This test is
optional; neither the installed package nor its regular tests depend on umd.
"""
import importlib
import os
import sys
import pytest
import torch

ROOT = os.environ.get("MONTHLY_DIFFUSION_SOURCE_ROOT")
pytestmark = pytest.mark.skipif(not ROOT, reason="Optional source-checkout parity")
SOURCE = "umd.mdv1_e3smv2_r1i1p1f1_frozenae150_dit4_1m_ordered_ocean6"


def test_original_autoencoder_values_and_gradients():
    sys.path.insert(0, ROOT)
    old = importlib.import_module(SOURCE+".autoencoder")
    from monthly_diffusion.modules.autoencoder import MeshHierarchyAutoencoder, AutoencoderConfig
    from monthly_diffusion.modules.mesh import build_mesh_hierarchy_context
    cfg = old.CapsuleConfig(nlat=18, nlon=36, atmosphere_channels=2, ocean_channels=1,
        width=8, mesh_levels=(2, 1, 0), m0_local_refinement_layers=1,
        m1_local_refinement_layers=1, m2_local_refinement_layers=1)
    source = old.SpatialBranch(input_channels=2, config=cfg)
    geometry = build_mesh_hierarchy_context(nlat=18, nlon=36, mesh_levels=(2, 1, 0))
    new = MeshHierarchyAutoencoder(2, geometry, (18, 36), AutoencoderConfig(width=8, mesh_levels=(2,1,0), refiners=(1,1,1)))
    pairs = [(source.input_projection, new.encoder.input_projection),
             (source.output_projection, new.decoder.output_projection),
             (source.latent_projection, new.encoder.latent_projection),
             (source.latent_expand, new.decoder.latent_expand)]
    for i, name in enumerate(("grid", "m0", "m1", "m2")):
        pairs += [(source.condition[name], new.encoder.context.conditions[i]),
                  (source.position[name], new.encoder.context.positions[i])]
    for i, name in enumerate(("down_g_m0", "down_m0_m1", "down_m1_m2")):
        block = getattr(source, name)
        pairs += [(block.cross, new.encoder.down[i]), (block.same.local, new.encoder.refiners[i].layers)]
    for i, name in enumerate(("up_m2_m1", "up_m1_m0", "up_m0_g")):
        pairs.append((getattr(source, name).cross, new.decoder.up[i]))
    pairs += [(source.latent_block.local, new.decoder.refiners[0].layers),
              (source.up_m2_m1.same.local, new.decoder.refiners[1].layers),
              (source.up_m1_m0.same.local, new.decoder.refiners[2].layers)]
    for src, dst in pairs:
        dst.load_state_dict(src.state_dict())
    x = torch.randn(2, 2, 18, 36, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    static, months, co2 = torch.randn_like(x), torch.randn(2, 2), torch.randn(2, 1)
    a, za = source(x, static, months, co2)
    b, zb = new(y, static, months, co2)
    torch.testing.assert_close(za, zb, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    a.square().mean().backward(); b.square().mean().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=5e-4, atol=2e-7)
    for src, dst in pairs:
        for p, q in zip(src.parameters(), dst.parameters()):
            if p.grad is None:
                assert q.grad is None
            else:
                torch.testing.assert_close(p.grad, q.grad, rtol=5e-4, atol=2e-6)


def test_original_diffusion_loss_sampler_and_rollout_gradients():
    sys.path.insert(0, ROOT)
    old = importlib.import_module(SOURCE+".diffusion")
    old_sampler = importlib.import_module(SOURCE+".coupled_sampler")
    from monthly_diffusion.components.coupled_latent_diffusion import CoupledLatentDiffusion, CoupledDiffusionConfig
    src = old.PairedLatentDiffusion(old.PairedLatentDiffusionConfig(n_latent_tokens=4,
        atmosphere_width=8, ocean_width=8, n_forcings=6, depth=1, n_heads=2, time_embedding_channels=8))
    dst = CoupledLatentDiffusion(CoupledDiffusionConfig(n_latent_tokens=4, width=8, depth=1, n_heads=2, time_embedding_channels=8))
    dst.atmosphere.load_state_dict(src.atmosphere.state_dict())
    dst.ocean.load_state_dict(src.ocean.state_dict())
    ah, oh = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
    a, o, f = torch.randn(2, 4, 4), torch.randn(2, 4, 4), torch.randn(2, 6)
    torch.manual_seed(13)
    expected = src.loss_terms(a, o, f, ah, oh)["latent_diffusion"]
    torch.manual_seed(13)
    actual = dst.loss_terms(a, o, f, ah, oh)["transition"]
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    na, no = torch.randn_like(a), torch.randn_like(o)
    expected = old_sampler.denoise_paired_with_timesteps(src, f, ah, oh,
        timesteps=torch.tensor([999,674,373,120,29,6,0]), method="dpmpp2m_midpoint", atmosphere_noise=na, ocean_noise=no)
    actual = dst.sample_for_training(f, ah, oh, noise=(na, no))[:2]
    for p, q in zip(expected, actual):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    sum(v.square().mean() for v in expected).backward()
    sum(v.square().mean() for v in actual).backward()
    for component in ("atmosphere", "ocean"):
        for p, q in zip(getattr(src, component).parameters(), getattr(dst, component).parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


def test_three_horizon_objective_matches_corrected_source():
    from types import SimpleNamespace
    sys.path.insert(0, ROOT)
    original_model = importlib.import_module(SOURCE+".model").ComponentLatentDynamicsModel
    old = importlib.import_module(SOURCE+".diffusion")
    terms = importlib.import_module(SOURCE+".frozen_training").transition_terms
    from monthly_diffusion.components.coupled_latent_diffusion import CoupledLatentDiffusion, CoupledDiffusionConfig
    src = original_model.__new__(original_model)
    torch.nn.Module.__init__(src)
    src.config = SimpleNamespace(rollout_steps=3, rollout_training_sample_steps=7,
        rollout_training_sampler="dpmpp2m_midpoint", rollout_training_timestep_spacing="logsnr",
        rollout_training_timestep_rho=.7, transition_loss_weight=.01)
    src.history_steps, src.forcing_channels, src.calendar_channels = 2, 4, 2
    src.joint_latent_channels, src.atmosphere_latent_channels = 8, 4
    src.transition = old.PairedLatentDiffusion(old.PairedLatentDiffusionConfig(n_latent_tokens=4,
        atmosphere_width=8, ocean_width=8, n_forcings=6, depth=1, n_heads=2, time_embedding_channels=8))
    dst = CoupledLatentDiffusion(CoupledDiffusionConfig(n_latent_tokens=4, width=8, depth=1, n_heads=2, time_embedding_channels=8))
    dst.atmosphere.load_state_dict(src.transition.atmosphere.state_dict())
    dst.ocean.load_state_dict(src.transition.ocean.state_dict())
    a, o, month = torch.randn(2, 5, 4, 4), torch.randn(2, 5, 4, 4), torch.randn(2, 5, 2)
    z, forcing = torch.cat((a, o), -1), torch.cat((month, torch.zeros_like(month)), -1)
    batch = dict(history=z[:, [1, 0]], targets=z[:, 2:], history_forcing=forcing[:, [1, 0]], target_forcing=forcing[:, 2:])
    torch.manual_seed(35)
    expected = terms(src, batch)
    torch.manual_seed(35)
    actual = dst.rollout_loss(dict(atmosphere=a, ocean=o, month=month))
    torch.testing.assert_close(expected["loss"], actual["loss"], rtol=1e-6, atol=1e-7)
    for i in range(1, 4):
        torch.testing.assert_close(expected[f"atmosphere_rollout_{i}"], actual[f"atmosphere_transition_step{i}"])
        torch.testing.assert_close(expected[f"ocean_rollout_{i}"], actual[f"ocean_transition_step{i}"])
    expected["loss"].backward(); actual["loss"].backward()
    for component in ("atmosphere", "ocean"):
        for p, q in zip(getattr(src.transition, component).parameters(), getattr(dst, component).parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-7)
