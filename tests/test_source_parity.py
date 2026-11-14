"""Development-only parity against the original implementation.

Set MONTHLY_DIFFUSION_SOURCE_ROOT to the parent umd checkout. This test is
optional; neither the installed package nor its regular tests depend on umd.
"""

from workflows.training.losses import loss_terms
from workflows.training.losses import rollout_loss
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
    old = importlib.import_module(SOURCE + ".autoencoder")
    from monthly_diffusion.autoencoder import Autoencoder, FiLM
    from monthly_diffusion.mesh import build_mesh_hierarchy

    cfg = old.CapsuleConfig(
        nlat=18,
        nlon=36,
        atmosphere_channels=2,
        ocean_channels=1,
        width=8,
        mesh_levels=(2, 1, 0),
        m0_local_refinement_layers=1,
        m1_local_refinement_layers=1,
        m2_local_refinement_layers=1,
    )
    source = old.SpatialBranch(input_channels=2, config=cfg)
    geometry = build_mesh_hierarchy(nlat=18, nlon=36, mesh_levels=(2, 1, 0))
    statics = torch.randn(2, 18, 36)
    new = Autoencoder(
        2,
        geometry,
        (18, 36),
        statics=statics,
        width=8,
        refiners=(1, 1, 1),
        refiner_mlp_ratio=0,
        residual_film_only=False,
    )
    pairs = [
        (source.input_projection, new.encoder.input_projection),
        (source.output_projection, new.decoder.output_projection),
        (source.latent_projection, new.encoder.latent_projection),
        (source.latent_expand, new.decoder.latent_expand),
    ]
    for i, name in enumerate(("grid", "m0", "m1", "m2")):
        pairs.append((source.condition[name], new.encoder.film[i]))
        if i == 0:
            pairs.append((source.position[name].net, new.encoder.position))
    for i, name in enumerate(("down_g_m0", "down_m0_m1", "down_m1_m2")):
        block = getattr(source, name)
        pairs += [
            (block.cross, new.encoder.down[i]),
            (
                block.same.local,
                torch.nn.ModuleList([b.spatial for b in new.encoder.refiners[i]]),
            ),
        ]
    for i, name in enumerate(("up_m2_m1", "up_m1_m0", "up_m0_g")):
        pairs.append((getattr(source, name).cross, new.decoder.up[i]))
    pairs += [
        (
            source.latent_block.local,
            torch.nn.ModuleList([b.spatial for b in new.decoder.refiners[0]]),
        ),
        (
            source.up_m2_m1.same.local,
            torch.nn.ModuleList([b.spatial for b in new.decoder.refiners[1]]),
        ),
        (
            source.up_m1_m0.same.local,
            torch.nn.ModuleList([b.spatial for b in new.decoder.refiners[2]]),
        ),
    ]
    for src, dst in pairs:
        weights = src.state_dict()
        if isinstance(dst, FiLM):
            weights = {
                name.replace("static.net.", "static.")
                .replace("time.month.", "month.")
                .replace("time.net.", "month_affine.")
                .replace("time.co2_shift.", "co2_shift."): value
                for name, value in weights.items()
            }
        dst.load_state_dict(weights)
    x = torch.randn(2, 2, 18, 36, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    static = statics[None].expand(2, -1, -1, -1)
    months, co2 = torch.randn(2, 2), torch.randn(2, 1)
    a, za = source(x, static, months, co2)
    b, zb = new(y, months, co2)
    torch.testing.assert_close(za, zb, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    a.square().mean().backward()
    b.square().mean().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=5e-4, atol=2e-7)
    for src, dst in pairs:
        for p, q in zip(src.parameters(), dst.parameters()):
            if p.grad is None:
                assert q.grad is None
            else:
                torch.testing.assert_close(p.grad, q.grad, rtol=5e-4, atol=2e-6)


def test_original_diffusion_loss_sampler_and_rollout_gradients():
    sys.path.insert(0, ROOT)
    old = importlib.import_module(SOURCE + ".diffusion")
    old_sampler = importlib.import_module(SOURCE + ".coupled_sampler")
    from monthly_diffusion.diffusion import CoupledLatentDiffusion

    src = old.PairedLatentDiffusion(
        old.PairedLatentDiffusionConfig(
            n_latent_tokens=4,
            atmosphere_width=8,
            ocean_width=8,
            n_forcings=6,
            depth=1,
            n_heads=2,
            time_embedding_channels=8,
        )
    )
    dst = CoupledLatentDiffusion(
        n_latent_tokens=4, width=8, depth=1, n_heads=2, time_embedding_channels=8
    )
    dst.atmosphere.load_state_dict(src.atmosphere.state_dict())
    dst.ocean.load_state_dict(src.ocean.state_dict())
    ah, oh = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
    a, o, f = torch.randn(2, 4, 4), torch.randn(2, 4, 4), torch.randn(2, 6)
    torch.manual_seed(13)
    expected = src.loss_terms(a, o, f, ah, oh)["latent_diffusion"]
    torch.manual_seed(13)
    actual = loss_terms(dst, a, o, f, ah, oh)["transition"]
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    na, no = torch.randn_like(a), torch.randn_like(o)
    expected = old_sampler.denoise_paired_with_timesteps(
        src,
        f,
        ah,
        oh,
        timesteps=torch.tensor([999, 674, 373, 120, 29, 6, 0]),
        method="dpmpp2m_midpoint",
        atmosphere_noise=na,
        ocean_noise=no,
    )
    actual = dst.denoise(f, ah, oh, noise=(na, no))[:2]
    for p, q in zip(expected, actual):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    sum(v.square().mean() for v in expected).backward()
    sum(v.square().mean() for v in actual).backward()
    for component in ("atmosphere", "ocean"):
        for p, q in zip(
            getattr(src, component).parameters(), getattr(dst, component).parameters()
        ):
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


def test_three_horizon_objective_matches_corrected_source():
    from types import SimpleNamespace

    sys.path.insert(0, ROOT)
    original_model = importlib.import_module(
        SOURCE + ".model"
    ).ComponentLatentDynamicsModel
    old = importlib.import_module(SOURCE + ".diffusion")
    terms = importlib.import_module(SOURCE + ".frozen_training").transition_terms
    from monthly_diffusion.diffusion import CoupledLatentDiffusion

    src = original_model.__new__(original_model)
    torch.nn.Module.__init__(src)
    src.config = SimpleNamespace(
        rollout_steps=3,
        rollout_training_sample_steps=7,
        rollout_training_sampler="dpmpp2m_midpoint",
        rollout_training_timestep_spacing="logsnr",
        rollout_training_timestep_rho=0.7,
        transition_loss_weight=0.01,
    )
    src.history_steps, src.forcing_channels, src.calendar_channels = 2, 4, 2
    src.joint_latent_channels, src.atmosphere_latent_channels = 8, 4
    src.transition = old.PairedLatentDiffusion(
        old.PairedLatentDiffusionConfig(
            n_latent_tokens=4,
            atmosphere_width=8,
            ocean_width=8,
            n_forcings=6,
            depth=1,
            n_heads=2,
            time_embedding_channels=8,
        )
    )
    dst = CoupledLatentDiffusion(
        n_latent_tokens=4, width=8, depth=1, n_heads=2, time_embedding_channels=8
    )
    dst.atmosphere.load_state_dict(src.transition.atmosphere.state_dict())
    dst.ocean.load_state_dict(src.transition.ocean.state_dict())
    a, o, month = torch.randn(2, 5, 4, 4), torch.randn(2, 5, 4, 4), torch.randn(2, 5, 2)
    z, forcing = torch.cat((a, o), -1), torch.cat((month, torch.zeros_like(month)), -1)
    batch = dict(
        history=z[:, [1, 0]],
        targets=z[:, 2:],
        history_forcing=forcing[:, [1, 0]],
        target_forcing=forcing[:, 2:],
    )
    torch.manual_seed(35)
    expected = terms(src, batch)
    torch.manual_seed(35)
    actual = rollout_loss(dst, a, o, month).metrics()
    torch.testing.assert_close(expected["loss"], actual["loss"], rtol=1e-6, atol=1e-7)
    for i in range(1, 4):
        torch.testing.assert_close(
            expected[f"atmosphere_rollout_{i}"],
            actual[f"atmosphere_transition_step{i}"],
        )
        torch.testing.assert_close(
            expected[f"ocean_rollout_{i}"], actual[f"ocean_transition_step{i}"]
        )
    expected["loss"].backward()
    actual["loss"].backward()
    for component in ("atmosphere", "ocean"):
        for p, q in zip(
            getattr(src.transition, component).parameters(),
            getattr(dst, component).parameters(),
        ):
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-7)
