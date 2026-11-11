"""Restore a saved run while retaining the requested training limits."""


def restore_training_state(model, checkpoint):
    """Keep the existing wrapper object, with weights and statistics restored."""
    requested = model.config
    restored = model.load(checkpoint, device=model.device)
    old, new = restored.config.to_dict(), requested.to_dict()
    for key in ("autoencoder_training", "diffusion_training"):
        old.pop(key)
        new.pop(key)
    if old != new:
        raise ValueError("Resume architecture/preprocessing differs from saved run")
    model.__dict__.update(restored.__dict__)
    model.config = requested
