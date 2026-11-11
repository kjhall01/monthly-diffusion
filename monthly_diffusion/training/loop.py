"""The training loop: optimize, validate the EMA, save, and check patience."""

from dataclasses import asdict
from pathlib import Path
import csv
import logging
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from ..checkpoints import atomic_save
from ..random_state import rng_state, restore_rng, fixed_rng
from .ema import EMA

log = logging.getLogger(__name__)


def fit(
    model, training_data, validation_data, objective, config, directory, *, resume=False
):
    """Select absolute best EMA validation; patience counts meaningful improvements.

    A training step is a complete batch (no gradient clipping/accumulation). Only
    EMA validation is run. Validation RNG is replayed without consuming training
    draws. Checkpoint selection and patience use separate best-so-far values.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=config.optimizer_eps,
    )
    ema = EMA(model, config.ema_decay)
    history, step, first_epoch, stale = [], 0, 1, 0
    best, patience_best = float("inf"), float("inf")
    checkpoint = directory / "latest.pt"
    if checkpoint.exists() and not resume:
        raise FileExistsError(f"Use resume=True or a new run directory: {checkpoint}")
    if resume and checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        prior = saved["training_config"]
        for name in (
            "batch_size",
            "learning_rate",
            "weight_decay",
            "optimizer_eps",
            "ema_decay",
            "seed",
            "rollout_steps",
            "loss_weight",
        ):
            if prior[name] != asdict(config)[name]:
                raise ValueError(
                    f"Resume changes training setting {name}; use a new run"
                )
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        ema.state = saved["ema"]
        history, step = saved["history"], saved["step"]
        # Older diffusion histories selected on the scaled ``loss`` (which is
        # loss_weight * transition).  Continue them using the raw transition
        # objective so min_delta has an interpretable physical scale.  Reset
        # patience once at the migration boundary; the historical rows remain
        # intact for provenance and plotting.
        if history and "val_transition" in history[-1]:
            best = min(float(row["val_transition"]) for row in history)
            patience_best = best
            stale = 0
        else:
            stale = saved["stale"]
            best, patience_best = saved["best"], saved["patience_best"]
        first_epoch = saved["epoch"] + 1
        restore_rng(saved["rng"])
    else:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
    validation_loader = DataLoader(
        validation_data, batch_size=config.batch_size, shuffle=False
    )
    for epoch in range(first_epoch, config.max_epochs + 1):
        started = time.monotonic()
        model.train()
        generator = torch.Generator().manual_seed(config.seed + epoch)
        loader = DataLoader(
            training_data,
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
        )
        totals, count = {}, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            terms = objective(model, batch)
            if not torch.isfinite(terms["loss"]):
                raise FloatingPointError("Nonfinite training loss")
            terms["loss"].backward()
            optimizer.step()
            step += 1
            ema.update(model, step)
            n = len(next(iter(batch.values())))
            count += n
            for name, value in terms.items():
                totals[name] = totals.get(name, 0.0) + n * float(value.detach())
        row = {
            "epoch": epoch,
            "optimizer_steps": step,
            **{f"train_{k}": v / count for k, v in totals.items()},
        }
        totals, count = {}, 0
        model.eval()
        with ema.applied(model), torch.no_grad():
            for seed in config.validation_seeds:
                with fixed_rng(seed, device):
                    for batch in validation_loader:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        terms = objective(model, batch)
                        n = len(next(iter(batch.values())))
                        count += n
                        for name, value in terms.items():
                            totals[name] = totals.get(name, 0.0) + n * float(value)
        row.update({f"val_{k}": v / count for k, v in totals.items()})
        row["seconds"] = time.monotonic() - started
        # For coupled diffusion, ``loss`` is deliberately multiplied by
        # loss_weight for gradient balancing.  Checkpoint selection and stale
        # patience must use the unscaled transition objective; otherwise a
        # min_delta of 1e-4 means a raw improvement must exceed 1e-2.
        score = row.get("val_transition", row["val_loss"])
        if not np.isfinite(score):
            raise FloatingPointError("Nonfinite validation loss")
        improved = score < best
        if improved:
            best = score
            atomic_save(
                dict(model=ema.state, epoch=epoch, score=score), directory / "best.pt"
            )
        if score < patience_best - config.min_delta:
            stale, patience_best = 0, score
        else:
            stale += 1
        row["stale_epochs"] = stale
        history.append(row)
        atomic_save(
            dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                ema=ema.state,
                history=history,
                epoch=epoch,
                step=step,
                best=best,
                patience_best=patience_best,
                stale=stale,
                rng=rng_state(device),
                training_config=asdict(config),
            ),
            checkpoint,
        )
        with (directory / "history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        log.info(
            "%s epoch %d | train %.6f val %.6f | %.1fs | stale %d/%d",
            directory.name,
            epoch,
            row["train_loss"],
            score,
            row["seconds"],
            stale,
            config.patience,
        )
        if stale >= config.patience:
            break
    chosen = torch.load(directory / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(chosen["model"])
    model.eval()
    return dict(epoch=chosen["epoch"], score=chosen["score"], history=history)
