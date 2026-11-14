"""CO2 conditioning, coherent dropout, CFG, and forecast restart behavior."""

import numpy as np
import pytest
import torch
import xarray as xr

from monthly_diffusion.diffusion import CoupledLatentDiffusion
from monthly_diffusion.loading import load_diffusion, save_model
from workflows.training.losses import rollout_loss
from workflows.data.dataloader import LatentSequences


def small_model(coupling="jacobi", **options):
    return CoupledLatentDiffusion(
        atmosphere_latent_channels=2,
        ocean_latent_channels=2,
        n_latent_tokens=4,
        width=8,
        depth=1,
        n_heads=2,
        time_embedding_channels=8,
        coupling=coupling,
        condition_on_co2=True,
        co2_mean=350,
        co2_std=20,
        **options,
    )


def test_null_condition_is_not_mean_co2_and_cfg_blends_velocities():
    torch.manual_seed(4)
    model = small_model().eval()
    head = model.atmosphere
    noisy = torch.randn(2, 4, 2)
    calendar = torch.randn(2, 6)
    history = torch.randn(2, 4, head.condition_channels)
    t = torch.tensor([999, 373])
    args = noisy, calendar, history, t
    conditional = head(*args, co2=350)
    null = head(*args, co2=None)
    assert not torch.equal(conditional, null)
    assert not torch.equal(conditional, head(*args, co2=560))
    assert not torch.equal(null, head(noisy, calendar + 1, history, t, co2=None))
    keep = torch.zeros(2, dtype=torch.bool)
    torch.testing.assert_close(
        head(*args, co2=280, co2_keep=keep), null, rtol=0, atol=0
    )
    torch.testing.assert_close(
        head(*args, co2=800, co2_keep=keep), null, rtol=0, atol=0
    )
    torch.testing.assert_close(
        head.predict_v(*args, co2=350, guidance_scale=0), null, rtol=0, atol=0
    )
    torch.testing.assert_close(
        head.predict_v(*args, co2=350, guidance_scale=1), conditional, rtol=0, atol=0
    )
    torch.testing.assert_close(
        head.predict_v(*args, co2=350, guidance_scale=2.5),
        null + 2.5 * (conditional - null),
        rtol=0,
        atol=0,
    )
    with pytest.raises(ValueError, match="target-month CO2"):
        head.predict_v(*args)


@pytest.mark.parametrize("coupling", ["jacobi", "atmosphere_first"])
def test_dropout_mask_is_shared_across_heads_targets_and_solver_steps(
    monkeypatch, coupling
):
    torch.manual_seed(7)
    model = small_model(coupling, co2_dropout=0.1).train()
    keep = torch.tensor([False, True])
    masks = []

    def choose_mask(batch, device):
        masks.append(batch)
        return keep

    monkeypatch.setattr(model, "co2_keep_mask", choose_mask)
    seen = []
    hooks = []
    for head in (model.atmosphere, model.ocean):
        hooks.append(
            head.register_forward_pre_hook(
                lambda module, args, kwargs: seen.append(
                    (kwargs["co2"].clone(), kwargs["co2_keep"])
                ),
                with_kwargs=True,
            )
        )
    atmosphere = torch.randn(2, 5, 4, 2)
    ocean = torch.randn_like(atmosphere)
    months = torch.randn(2, 5, 2)
    co2 = torch.tensor(
        [[320.0, 321.0, 322.0, 323.0, 324.0], [420.0, 421.0, 422.0, 423.0, 424.0]]
    )
    losses = rollout_loss(model, atmosphere, ocean, months, co2=co2)
    losses.total.backward()
    assert masks == [2]  # exactly one draw for the entire rollout
    assert {tuple(ppm.tolist()) for ppm, _ in seen} == {
        (322.0, 422.0),
        (323.0, 423.0),
        (324.0, 424.0),
    }
    assert all(mask is keep for _, mask in seen)
    for head in (model.atmosphere, model.ocean):
        assert head.null_co2.grad.abs().sum() > 0
        assert head.co2_mlp[0].weight.grad.abs().sum() > 0
        assert torch.isfinite(head.null_co2.grad).all()
    for hook in hooks:
        hook.remove()


def test_dropout_frequency_and_validation_have_no_random_drop():
    model = small_model(co2_dropout=0.1).train()
    torch.manual_seed(19)
    keep = model.co2_keep_mask(10000, "cpu")
    assert 0.08 < float((~keep).float().mean()) < 0.12
    model.eval()
    before = torch.get_rng_state()
    assert model.co2_keep_mask(100, "cpu").all()
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("coupling", ["jacobi", "atmosphere_first"])
def test_guided_sampling_roundtrip_and_null_co2_invariance(tmp_path, coupling):
    torch.manual_seed(8)
    model = small_model(coupling, co2_dropout=0.1).eval()
    ah, oh, calendar = torch.randn(2, 4, 4), torch.randn(2, 4, 4), torch.randn(2, 6)
    noise = torch.randn(2, 4, 2), torch.randn(2, 4, 2)
    low = model.sample(calendar, ah, oh, noise=noise, co2=280, guidance_scale=0)
    high = model.sample(calendar, ah, oh, noise=noise, co2=800, guidance_scale=0)
    for a, b in zip(low, high):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    guided = model.sample(calendar, ah, oh, noise=noise, co2=560, guidance_scale=2)
    assert all(torch.isfinite(z).all() for z in guided)
    assert not torch.equal(guided[0], low[0])
    path = tmp_path / "diffusion.pt"
    save_model(model, path)
    restored = load_diffusion(path, device="cpu")
    assert restored.condition_on_co2 and restored.co2_dropout == 0.1
    assert restored.atmosphere.co2_mean == 350 and restored.atmosphere.co2_std == 20
    actual = restored.sample(calendar, ah, oh, noise=noise, co2=560, guidance_scale=2)
    for a, b in zip(actual, guided):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_latent_windows_align_co2_with_the_targets():
    time = xr.DataArray(xr.date_range("2000-01", periods=8, freq="MS"), dims="time")
    states = torch.arange(8.0).reshape(8, 1, 1)
    data = LatentSequences(states, states, time, steps=3, co2=np.arange(8.0) + 350)
    atmosphere, _, _, co2 = data[2]
    torch.testing.assert_close(co2, atmosphere[:, 0, 0] + 350)
    assert co2[2:].tolist() == [354, 355, 356]


def test_guidance_in_rollout_and_resume(tiny_data, tmp_path, train_small_run):
    model = train_small_run(tiny_data, tmp_path / "run", condition_on_co2=False)
    assert model.transition.condition_on_co2
    assert not model.autoencoder.atmosphere.encoder.condition_on_co2
    assert not model.autoencoder.ocean.encoder.condition_on_co2
    # AEs ignore CO2; at scale 0, the complete free forecast must do so too.
    low = model.rollout(
        co2=280, gaussian_start="1966-01", months=2, members=2, guidance_scale=0
    )
    high = model.rollout(
        co2=800, gaussian_start="1966-01", months=2, members=2, guidance_scale=0
    )
    xr.testing.assert_identical(low.drop_vars("co2"), high.drop_vars("co2"))
    guided = model.rollout(
        co2=560, gaussian_start="1966-01", months=2, members=2, guidance_scale=2
    )
    assert not np.array_equal(guided.t2m, low.t2m)
    options = dict(co2=560, gaussian_start="1966-01", members=2, guidance_scale=2)
    model.rollout(**options, months=1, path=tmp_path / "archive", open_result=False)
    with pytest.raises(ValueError, match="differs from archive"):
        model.rollout(
            **{**options, "guidance_scale": 1},
            months=2,
            path=tmp_path / "archive",
            resume=True,
            extend=True,
        )
    result = model.rollout(
        **options, months=2, path=tmp_path / "archive", resume=True, extend=True
    )
    try:
        xr.testing.assert_identical(result.compute(), guided.astype("float32"))
    finally:
        result.close()
