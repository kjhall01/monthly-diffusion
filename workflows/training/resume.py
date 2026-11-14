"""Restore weights and statistics while retaining requested training limits."""


def restore_training_state(model, checkpoint):
    requested, training = model.config, model.training_settings
    restored = model.load(checkpoint, device=model.device)
    if restored.config.to_dict() != requested.to_dict():
        raise ValueError("Resume architecture/preprocessing differs from saved run")
    model.__dict__.update(restored.__dict__)
    model.config, model.training_settings = requested, training
