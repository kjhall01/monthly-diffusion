"""Encode each split once and normalize with training-only latent statistics."""

from pathlib import Path
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from ..checkpoints import atomic_save
from ..data.preprocessing import LatentNormalizer

log = logging.getLogger(__name__)


def cache_latents(model, training_data, validation_data, directory):
    """Return a validated cache, creating it with the selected frozen AE if needed."""
    directory = Path(directory)
    datasets = (training_data, validation_data)
    training_time = training_data.datasets["atmosphere"].time
    validation_time = validation_data.datasets["atmosphere"].time
    statics = (
        model.preprocessor.static_tensor(model.statics).to(model.device).unsqueeze(0)
    )
    cache_path = directory / "latents.pt"
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache["ae_digest"] != model.weight_digest(autoencoder_only=True):
            raise ValueError("Latent cache belongs to a different selected autoencoder")
        if not np.array_equal(
            cache["training_time"], training_time
        ) or not np.array_equal(cache["validation_time"], validation_time):
            raise ValueError(
                "Cached training/validation dates differ from supplied splits"
            )
        model.latent_normalizers = cache["normalizers"]
    else:
        log.info(
            "Stage 2: encoding training/validation states once; fitting train-only latent statistics"
        )
        arrays = []
        with torch.no_grad():
            for dataset in datasets:
                result = {component: [] for component in model.names}
                loader = DataLoader(
                    dataset, batch_size=model.config.autoencoder_training.batch_size
                )
                for batch in loader:
                    batch = {k: v.to(model.device) for k, v in batch.items()}
                    for component in model.names:
                        node_mask = (
                            torch.isfinite(batch[component][:, 0])
                            if getattr(model.autoencoder, component, None).mask_aware
                            else None
                        )
                        z = getattr(model.autoencoder, component).encode(
                            batch[component],
                            statics.expand(len(batch["month"]), -1, -1, -1),
                            batch["month"],
                            batch["co2"],
                            node_mask=node_mask,
                        )
                        result[component].append(z.cpu())
                arrays.append({k: torch.cat(v) for k, v in result.items()})
        model.latent_normalizers = {
            k: LatentNormalizer().fit(arrays[0][k]) for k in model.names
        }
        for item in arrays:
            for key in item:
                item[key] = model.latent_normalizers[key].normalize(item[key])
        cache = dict(
            training=arrays[0],
            validation=arrays[1],
            normalizers=model.latent_normalizers,
            ae_digest=model.weight_digest(autoencoder_only=True),
            training_time=training_time,
            validation_time=validation_time,
        )
        atomic_save(cache, cache_path)
    return cache
