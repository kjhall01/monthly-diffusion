"""Render lightweight diagnostics for a frozen-stage training history."""

from __future__ import annotations
# Make the repository workflow helpers available when running this file directly.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse, json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("history", type=Path)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
d = pd.read_csv(args.history)
args.output.mkdir(parents=True, exist_ok=True)
best = d.loc[d.val_loss.idxmin()]
report = {
    "epochs_completed": int(d.epoch.max()),
    "selected_best_epoch": int(best.epoch),
    "best_scaled_validation_loss": float(best.val_loss),
    "best_unscaled_validation_transition": float(best.val_transition),
    "best_validation_atmosphere_transition": float(best.val_atmosphere_transition),
    "best_validation_ocean_transition": float(best.val_ocean_transition),
    "final_stale_epochs": int(d.stale_epochs.iloc[-1]),
    "first_epoch_stale_threshold_reached": int(
        d.loc[d.stale_epochs >= 5, "epoch"].iloc[0]
    )
    if (d.stale_epochs >= 5).any()
    else None,
    "loss_multiplier_inferred": float(
        d.train_loss.iloc[0] / d.train_transition.iloc[0]
    ),
    "validation_transition_improvement_epoch1_to_last": float(
        d.val_transition.iloc[0] - d.val_transition.iloc[-1]
    ),
    "validation_loss_improvement_epoch1_to_last": float(
        d.val_loss.iloc[0] - d.val_loss.iloc[-1]
    ),
}
(args.output / "summary.json").write_text(json.dumps(report, indent=2))
fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
ax[0, 0].plot(d.epoch, d.train_transition, label="train raw transition")
ax[0, 0].plot(d.epoch, d.val_transition, label="validation raw transition")
ax[0, 0].set_title("Raw transition objective")
ax[0, 0].set_xlabel("epoch")
ax[0, 0].legend()
for label, prefix, axis in (
    ("atmosphere", "atmosphere_transition", ax[0, 1]),
    ("ocean", "ocean_transition", ax[1, 0]),
):
    axis.plot(d.epoch, d[f"train_{prefix}"], label="train")
    axis.plot(d.epoch, d[f"val_{prefix}"], label="validation")
    for step in (1, 2, 3):
        axis.plot(
            d.epoch,
            d[f"val_{prefix}_step{step}"],
            linestyle="--",
            alpha=0.65,
            label=f"val step {step}",
        )
    axis.set_title(label + " transition")
    axis.set_xlabel("epoch")
    axis.legend(fontsize=8)
ax[1, 1].plot(d.epoch, d.val_loss, label="scaled validation loss")
ax[1, 1].plot(
    d.epoch,
    d.stale_epochs / max(d.stale_epochs.max(), 1) * d.val_loss.max(),
    label="stale count (rescaled)",
)
ax[1, 1].set_title("Selection signal and patience")
ax[1, 1].set_xlabel("epoch")
ax[1, 1].legend()
fig.savefig(args.output / "training_diagnostics.png", dpi=220)
fig.savefig(args.output / "training_diagnostics.pdf")
print(json.dumps(report, indent=2))
