"""Exponential moving averages of model parameters."""

from contextlib import contextmanager
import torch


class EMA:
    """Parameter EMA only; buffers copied, effective decay warms up each step."""

    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model, step):
        decay = min(self.decay, (1 + step) / (10 + step))
        params = {n for n, _ in model.named_parameters(remove_duplicate=False)}
        for name, value in model.state_dict().items():
            if name in params:
                self.state[name].lerp_(value, 1 - decay)
            else:
                self.state[name].copy_(value)

    @contextmanager
    def applied(self, model):
        online = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.state)
        try:
            yield
        finally:
            model.load_state_dict(online)
