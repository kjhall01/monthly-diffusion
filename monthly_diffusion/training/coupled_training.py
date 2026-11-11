"""Paired diffusion-only training on fixed cached latent sequences."""

from .loop import fit


def train_coupled_diffusion(
    model, train, validation, *, config, directory, resume=False
):
    return fit(
        model,
        train,
        validation,
        lambda model, batch: model.rollout_loss(batch, loss_weight=config.loss_weight),
        config,
        directory,
        resume=resume,
    )
