from workflows.training.losses import rollout_loss
import math
import types
import torch
from monthly_diffusion.diffusion import ar1_noise
from monthly_diffusion.diffusion import CoupledLatentDiffusion


def tiny(coupling="jacobi", cross=False):
    return CoupledLatentDiffusion(
        n_latent_tokens=4,
        width=8,
        depth=1,
        n_heads=2,
        time_embedding_channels=8,
        coupling=coupling,
        cross_gradients=cross,
    )


def test_explicit_newest_first_all_training_steps(monkeypatch):
    model = tiny()
    seen = []

    def loss(self, a, o, calendar, ah, oh, **kwargs):
        seen.append((ah.clone(), oh.clone(), calendar.clone()))
        return dict(
            transition=a.sum() * 0,
            atmosphere_transition=a.sum() * 0,
            ocean_transition=o.sum() * 0,
        )

    def sample(self, calendar, ah, oh, **kwargs):
        a, o = ah[..., :4] + 1, oh[..., :4] + 1
        return a, o, a * 0, o * 0

    monkeypatch.setattr("workflows.training.losses.loss_terms", loss)
    model.denoise = types.MethodType(sample, model)
    a = torch.arange(5.0).view(1, 5, 1, 1).expand(1, 5, 4, 4)
    months = torch.arange(10.0).view(1, 5, 2)
    rollout_loss(model, a, a + 10, months).metrics()
    assert len(seen) == 3
    for step, (ah, oh, dates) in enumerate(seen):
        assert torch.all(ah[..., :4] == step + 1) and torch.all(ah[..., 4:] == step)
        assert torch.all(oh[..., :4] == step + 11) and torch.all(
            oh[..., 4:] == step + 10
        )
        torch.testing.assert_close(
            dates,
            torch.cat((months[:, step + 1], months[:, step], months[:, step + 2]), -1),
        )


def test_no_cross_gradient_context_and_ar1():
    model = tiny()
    a = torch.randn(2, 4, 8, requires_grad=True)
    o = torch.randn(2, 4, 8, requires_grad=True)
    on = torch.randn(2, 4, 4, requires_grad=True)
    model._atmosphere_context(a, o, on).sum().backward()
    assert a.grad is not None and o.grad is None and on.grad is None
    previous = torch.ones(5)
    actual = ar1_noise(previous, previous, tau_months=6, innovation=torch.zeros(5))
    torch.testing.assert_close(actual, previous * math.exp(-1 / 6))


def test_gauss_seidel_generated_condition_is_reused():
    model = tiny("atmosphere_first", True)
    records = []
    original = model._ocean_context

    def record(self, oh, ah, generated):
        records.append(generated)
        return original(oh, ah, generated)

    model._ocean_context = types.MethodType(record, model)
    data = dict(
        atmosphere=torch.randn(2, 5, 4, 4),
        ocean=torch.randn(2, 5, 4, 4),
        month=torch.randn(2, 5, 2),
    )
    result = rollout_loss(model, data['atmosphere'], data['ocean'], data['month']).metrics()
    # Two ocean contexts per intermediate month (loss, sampling), one at final.
    assert len(records) == 5
    assert records[0] is records[1] and records[2] is records[3]
    assert records[0].requires_grad and records[-1].requires_grad
    result["ocean_transition"].backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.atmosphere.parameters()
    )
