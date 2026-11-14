"""Wait for accelerator work before measuring elapsed time."""

import torch


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    if device.type == "cuda":
        torch.cuda.synchronize()
