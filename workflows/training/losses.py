"""Physical reconstruction and latent rollout objectives used by training scripts."""

from dataclasses import dataclass
import torch
from torch.nn import functional as F
from monthly_diffusion.diffusion import Coupling


def reconstruction_mse(prediction, target, latitudes):
    """MSE per variable, averaged over samples.

    For each sample/variable, cosine-latitude weights over finite target cells
    sum to one. Call .mean() on the result to give each variable equal weight.
    """
    valid = torch.isfinite(target)
    residual = torch.where(valid, prediction - torch.nan_to_num(target), 0.0)
    w = torch.cos(torch.deg2rad(latitudes.to(prediction))).clamp_min(0)[
        None, None, :, None
    ]
    denominator = (w * valid).sum((-2, -1))
    if (denominator == 0).any():
        raise ValueError("A reconstruction target has no valid cells")
    return ((w * residual.square()).sum((-2, -1)) / denominator).mean(0)


def diffusion_loss(model, z, calendar, history, *, t=None, noise=None):
    noisy, target, t = model.diffuse(z, t, noise)
    return F.mse_loss(model(noisy, calendar, history, t), target)


def loss_terms(
    model,
    a,
    o,
    calendar,
    ah,
    oh,
    *,
    t=None,
    noise_a=None,
    noise_o=None,
    atmosphere_condition=None,
    co2=None,
    co2_keep=None,
):
    t = model.atmosphere.prepare_timestep(t, a.shape[0], a.device)
    an, av, _ = model.atmosphere.diffuse(a, t, noise_a)
    on, ov, _ = model.ocean.diffuse(o, t, noise_o)
    if model.coupling == Coupling.ATMOSPHERE_FIRST and atmosphere_condition is None:
        raise ValueError(
            "Atmosphere-first loss requires a GENERATED atmosphere, not its target"
        )
    ao = an if model.coupling == Coupling.JACOBI else atmosphere_condition
    ap = model.atmosphere(
        an,
        calendar,
        model._atmosphere_context(ah, oh, on),
        t,
        co2=co2,
        co2_keep=co2_keep,
    )
    op = model.ocean(
        on, calendar, model._ocean_context(oh, ah, ao), t, co2=co2, co2_keep=co2_keep
    )
    al, ol = F.mse_loss(ap, av), F.mse_loss(op, ov)
    return {
        "transition": (al + ol) / 2,
        "atmosphere_transition": al,
        "ocean_transition": ol,
    }


@dataclass
class DiffusionLoss:
    """One loss per forecast month, plus the scalar used for backpropagation."""

    atmosphere: torch.Tensor
    ocean: torch.Tensor
    loss_weight: float = 0.01

    @property
    def transition(self):
        return ((self.atmosphere + self.ocean) / 2).mean()

    @property
    def total(self):
        return self.loss_weight * self.transition

    def metrics(self):
        # CSV names are kept stable for existing analysis scripts.
        values = {}
        for step in range(len(self.atmosphere)):
            values[f"atmosphere_transition_step{step + 1}"] = self.atmosphere[step]
            values[f"ocean_transition_step{step + 1}"] = self.ocean[step]
            values[f"transition_step{step + 1}"] = (
                self.atmosphere[step] + self.ocean[step]
            ) / 2
        values["atmosphere_transition"] = self.atmosphere.mean()
        values["ocean_transition"] = self.ocean.mean()
        values["transition"] = self.transition
        values["loss"] = self.total
        return values


def rollout_loss(model, atmosphere, ocean, months, *, co2=None, loss_weight=0.01):
    """Score observed targets while feeding generated forecasts into later histories.

    Inputs are chronological: previous month, current month, forecast targets.
    Each component and forecast month contributes equal weight to the loss.
    """
    atmos_previous, atmos_current = atmosphere[:, 0], atmosphere[:, 1]
    ocean_previous, ocean_current = ocean[:, 0], ocean[:, 1]
    previous_noise = (None, None)
    atmosphere_losses, ocean_losses = [], []
    steps = atmosphere.shape[1] - 2
    if model.condition_on_co2:
        if co2 is None or co2.shape != atmosphere.shape[:2]:
            raise ValueError("Supply CO2 ppm for every month of each latent sequence")
        if not torch.isfinite(co2).all() or (co2 <= 0).any():
            raise ValueError("CO2 must be positive and finite")
    # Reuse the same decision in losses AND generated histories. Redrawing it
    # inside the solver would mix conditional and null trajectories during fit.
    co2_keep = model.co2_keep_mask(atmosphere.shape[0], atmosphere.device)
    for step in range(steps):
        target_co2 = None if co2 is None else co2[:, step + 2]
        atmosphere_history = torch.cat((atmos_current, atmos_previous), dim=-1)
        ocean_history = torch.cat((ocean_current, ocean_previous), dim=-1)
        calendar = torch.cat(
            (months[:, step + 1], months[:, step], months[:, step + 2]), dim=-1
        )
        if model.coupling == Coupling.ATMOSPHERE_FIRST:
            noise_a, noise_o = model.terminal_noise(
                atmosphere_history, ocean_history, previous_noise
            )
            predicted_atmosphere = model.atmosphere.denoise(
                calendar,
                model._atmosphere_context(atmosphere_history, ocean_history),
                noise_a,
                co2=target_co2,
                co2_keep=co2_keep,
            )
            result = loss_terms(
                model,
                atmosphere[:, step + 2],
                ocean[:, step + 2],
                calendar,
                atmosphere_history,
                ocean_history,
                atmosphere_condition=predicted_atmosphere,
                co2=target_co2,
                co2_keep=co2_keep,
            )
            if step + 1 < steps:
                predicted_ocean = model.ocean.denoise(
                    calendar,
                    model._ocean_context(
                        ocean_history, atmosphere_history, predicted_atmosphere
                    ),
                    noise_o,
                    co2=target_co2,
                    co2_keep=co2_keep,
                )
                previous_noise = noise_a, noise_o
        else:
            result = loss_terms(
                model,
                atmosphere[:, step + 2],
                ocean[:, step + 2],
                calendar,
                atmosphere_history,
                ocean_history,
                co2=target_co2,
                co2_keep=co2_keep,
            )
            if step + 1 < steps:
                predicted_atmosphere, predicted_ocean, noise_a, noise_o = model.denoise(
                    calendar,
                    atmosphere_history,
                    ocean_history,
                    previous_noise=previous_noise,
                    co2=target_co2,
                    co2_keep=co2_keep,
                )
                previous_noise = noise_a, noise_o
        atmosphere_losses.append(result["atmosphere_transition"])
        ocean_losses.append(result["ocean_transition"])
        if step + 1 < steps:
            atmos_previous, atmos_current = atmos_current, predicted_atmosphere
            ocean_previous, ocean_current = ocean_current, predicted_ocean
    return DiffusionLoss(
        torch.stack(atmosphere_losses), torch.stack(ocean_losses), loss_weight
    )
