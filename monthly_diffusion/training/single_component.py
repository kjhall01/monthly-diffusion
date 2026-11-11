"""Single-component diffusion training using the same explicit history order."""

import torch
from .loop import fit
from ..components.coupling import ar1_noise


def train_diffusion(
    model, train, validation, *, config, directory, tau_months=0.0, resume=False
):
    """Datasets yield {'latent': [history+future,N,C], 'month': [history+future,2]}."""

    def objective(model, batch):
        z, months = batch["latent"], batch["month"]
        current, previous, noise = z[:, 1], z[:, 0], None
        losses = []
        for i in range(z.shape[1] - 2):
            history = torch.cat((current, previous), -1)
            calendar = torch.cat((months[:, i + 1], months[:, i], months[:, i + 2]), -1)
            losses.append(model.loss(z[:, i + 2], calendar, history))
            if i < z.shape[1] - 3:
                noise = ar1_noise(current, noise, tau_months=tau_months)
                predicted = model.sample_for_training(calendar, history, noise)
                current, previous = predicted, current
        terms = {f"transition_step{i + 1}": value for i, value in enumerate(losses)}
        terms["transition"] = torch.stack(losses).mean()
        terms["loss"] = config.loss_weight * terms["transition"]
        return terms

    return fit(model, train, validation, objective, config, directory, resume=resume)
