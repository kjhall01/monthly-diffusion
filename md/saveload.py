import torch
import os
import inspect
import pickle


def save_model(model, path):
    """
    Save model weights + constructor arguments using pickle only.
    This supports nested custom classes and arbitrary Python objects.
    """
    os.makedirs(path, exist_ok=True)

    # Save weights
    torch.save(model.state_dict(), os.path.join(path, "weights.pt"))

    # Get constructor args (excluding 'self')
    sig = inspect.signature(model.__class__.__init__)
    ctor_args = [p.name for p in sig.parameters.values()
                 if p.name != "self" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]

    # Collect args into config dict
    config = {}
    for arg in ctor_args:
        if hasattr(model, arg):
            config[arg] = getattr(model, arg)

    # Save config with pickle
    with open(os.path.join(path, "config.pkl"), "wb") as f:
        pickle.dump(config, f)


def load_model(model_class, path, map_location=None, override_kwargs={'loading': True}):
    """
    Load model with constructor arguments restored from pickle.
    """
    # Load config
    with open(os.path.join(path, "config.pkl"), "rb") as f:
        config = pickle.load(f)


    config.update(override_kwargs)
    print("Loading model with config:", config  )
    # Instantiate model
    model = model_class(**config)

    # Load weights
    state_dict = torch.load(os.path.join(path, "weights.pt"), map_location=map_location)
    model.load_state_dict(state_dict)
    return model
