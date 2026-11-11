"""Reconstruction-only first stage; no transition or EMA latent targets."""

import torch
from .loop import fit
from ..modules.autoencoder import reconstruction_mse


def train_autoencoder(
    model,
    train,
    validation,
    *,
    statics,
    latitudes,
    weights,
    config,
    directory,
    resume=False,
):
    """Fit paired AEs or a single MeshHierarchyAutoencoder.

    A single-component dataset yields state/month/co2; weights is a vector.
    A paired dataset yields atmosphere/ocean/month/co2; weights is a dict.
    Both use transformed physical targets, with NaNs retained for masking.
    """

    def objective(model, batch):
        terms, loss = {}, 0.0
        static_batch = statics.expand(batch["month"].shape[0], -1, -1, -1)
        if "state" in batch:
            prediction, _ = model(
                batch["state"], static_batch, batch["month"], batch["co2"]
            )
            per_field = reconstruction_mse(prediction, batch["state"], latitudes)
            return {
                "loss": (per_field * weights.to(per_field)).sum(),
                "reconstruction": per_field.mean(),
            }
        for component in ("atmosphere", "ocean"):
            component_model = getattr(model, component)
            node_mask = (
                torch.isfinite(batch[component][:, 0])
                if getattr(component_model, "mask_aware", False)
                else None
            )
            prediction, _ = component_model(
                batch[component],
                static_batch,
                batch["month"],
                batch["co2"],
                node_mask=node_mask,
            )
            per_field = reconstruction_mse(prediction, batch[component], latitudes)
            terms[component + "_reconstruction"] = per_field.mean()
            for i, value in enumerate(per_field):
                terms[f"{component}_field{i}"] = value
            loss = loss + (per_field * weights[component].to(per_field)).sum()
        terms["loss"] = loss
        return terms

    # Reconstruction is deterministic: only one validation pass is necessary.
    from dataclasses import replace

    return fit(
        model,
        train,
        validation,
        objective,
        replace(config, validation_seeds=(config.validation_seeds[0],)),
        directory,
        resume=resume,
    )
