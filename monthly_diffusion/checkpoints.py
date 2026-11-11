"""Write a complete checkpoint before replacing its previous version."""

from pathlib import Path
import torch


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)
