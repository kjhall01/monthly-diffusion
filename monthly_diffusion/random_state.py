"""Save and restore random streams without consuming someone else's draws."""

from contextlib import contextmanager
import random
import numpy as np
import torch


def rng_state(device):
    state = dict(
        torch=torch.get_rng_state(),
        numpy=np.random.get_state(),
        python=random.getstate(),
    )
    if device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"].cpu())
    if "cuda" in state:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


@contextmanager
def fixed_rng(seed, device):
    state = rng_state(device)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        restore_rng(state)
