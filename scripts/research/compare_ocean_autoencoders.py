"""Refresh physical validation RMSE plots for the old and corrected ocean AEs.

Run with the monthly-diffusion environment. The production references are cached
beside the figures, with their checkpoint, validation period, and recovery method.
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-monthly-comparison")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

RUN_DIR = Path(__file__).resolve().parents[1] / "runs/test_gridpoint_no_co2"
OUTPUT = RUN_DIR / "analysis/ocean_training_comparison"
REFERENCE = OUTPUT / "frozen_reference_provenance.json"
RUNS = ("ocean_broken", "ocean")

# Convert RMSEs only for display; the histories retain their original SI units.
variables = [
    ("sst", "SST", "°C", 1.0),
    ("ssh", "Sea-surface height", "cm", 100.0),
    ("sss", "Sea-surface salinity", "psu", 1.0),
    ("mld", "Mixed-layer depth", "m", 1.0),
    ("sic", "Sea-ice concentration", "percentage points", 100.0),
    ("sivol", "Sea-ice equivalent thickness", "m", 1.0),
    ("d20", "20°C isotherm depth", "m", 1.0),
    ("ohc300", "OHC 0–300 m", "GJ/m²", 1.0e-9),
    ("ohc700_minus_ohc300", "OHC 300–700 m", "GJ/m²", 1.0e-9),
]

# Histories are replaced atomically by training. Read each file once per refresh.
histories = {}
for run in RUNS:
    with (RUN_DIR / run / "history.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    histories[run] = {int(row["epoch"]): row for row in rows}
    epochs = sorted(histories[run])
    if not epochs or epochs != list(range(1, epochs[-1] + 1)):
        raise ValueError(f"Incomplete epoch history for {run}")
latest = {run: max(history) for run, history in histories.items()}
shared_epoch = max(set(histories[RUNS[0]]) & set(histories[RUNS[1]]))
reference = json.loads(REFERENCE.read_text())
references = {row["variable"]: row for row in reference["metrics"]}
OUTPUT.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")

curves, comparisons, reference_rows = [], [], []
for name, title, unit, factor in variables:
    for run, history in histories.items():
        for epoch, row in sorted(history.items()):
            rmse = float(row[f"val_rmse_{name}"]) * factor
            if not np.isfinite(rmse) or rmse < 0:
                raise ValueError(f"Invalid physical RMSE: {run}, {name}, {epoch}")
            curves.append(
                dict(
                    run=run,
                    epoch=epoch,
                    variable=name,
                    units=unit,
                    validation_rmse=rmse,
                )
            )
    old = float(histories["ocean_broken"][shared_epoch][f"val_rmse_{name}"]) * factor
    new = float(histories["ocean"][shared_epoch][f"val_rmse_{name}"]) * factor
    ref = references[name]
    baseline = ref["reference_physical_rmse"] * factor
    if not np.isfinite(baseline) or baseline <= 0:
        raise ValueError(f"Invalid production reference for {name}")
    comparisons.append(
        dict(
            variable=name,
            units=unit,
            epoch=shared_epoch,
            ocean_broken_rmse=old,
            ocean_rmse=new,
            change_percent=100 * (new / old - 1),
            production_rmse=baseline,
            production_epoch=ref["reference_epoch"],
        )
    )
    reference_rows.append(
        dict(
            variable=name,
            units=unit,
            physical_rmse=baseline,
            epoch=ref["reference_epoch"],
            method=ref["reference_method"],
        )
    )

for filename, rows in (
    ("validation_curves.csv", curves),
    ("matched_epoch.csv", comparisons),
    ("frozen_reference.csv", reference_rows),
):
    with (OUTPUT / filename).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
(OUTPUT / "snapshot.json").write_text(
    json.dumps(
        dict(
            refreshed=timestamp,
            latest_epochs=latest,
            shared_epoch=shared_epoch,
            histories={run: str(RUN_DIR / run / "history.csv") for run in RUNS},
            metric=reference["metric"],
            reference=str(REFERENCE),
        ),
        indent=2,
    )
    + "\n"
)

plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)
fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
colors = {"ocean_broken": "#2066a8", "ocean": "#d45c24"}
for ax, (name, title, unit, factor), comparison in zip(
    axes.flat, variables, comparisons
):
    for run in RUNS:
        rows = [r for r in curves if r["variable"] == name and r["run"] == run]
        ax.plot(
            [r["epoch"] for r in rows],
            [r["validation_rmse"] for r in rows],
            color=colors[run],
            lw=1.9,
        )
    ref = references[name]
    terminal = ref["reference_epoch"] == ref["terminal_epoch"]
    color, style = ("#208042", "--") if terminal else ("#8451a1", "-.")
    ax.axhline(comparison["production_rmse"], color=color, linestyle=style, lw=1.6)
    ax.text(
        0.98,
        0.94,
        f"Frozen e{ref['reference_epoch']}: {comparison['production_rmse']:.4g}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=color,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2),
    )
    ax.set_title(
        f"{title}  |  epoch {shared_epoch}: {comparison['change_percent']:+.1f}%",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.set_ylabel(f"RMSE [{unit}]")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.17)
    ax.axvline(shared_epoch, color="#777777", linestyle=":", linewidth=0.9, alpha=0.7)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
for ax in axes[-1]:
    ax.set_xlabel("Epoch")
fig.suptitle(
    "Ocean autoencoders: physical-unit validation RMSE",
    fontsize=17,
    fontweight="bold",
    y=0.985,
)
subtitle = fig.text(0.5, 0.955, "", ha="center", fontsize=10.5)
fig.legend(
    handles=[
        Line2D(
            [0],
            [0],
            color=colors["ocean_broken"],
            lw=2,
            label="ocean_broken — previous masks",
        ),
        Line2D([0], [0], color=colors["ocean"], lw=2, label="ocean — corrected masks"),
        Line2D(
            [0],
            [0],
            color="#208042",
            linestyle="--",
            lw=1.6,
            label="Production — terminal epoch 180",
        ),
        Line2D(
            [0],
            [0],
            color="#8451a1",
            linestyle="-.",
            lw=1.6,
            label="Production — selected epoch 176*",
        ),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, 0.941),
    ncol=2,
    frameon=False,
    fontsize=10,
)
fig.text(
    0.5,
    0.017,
    "Validation: 1958–1965, EMA weights; cosine-latitude-weighted RMSE over valid cells. Lower is better.\n"
    "*MLD, ice concentration, ice thickness and D20 use the saved epoch-176 production AE: terminal physical errors cannot be recovered.\n"
    f"Panel percentages compare ocean against ocean_broken at the same epoch ({shared_epoch}). Refreshed {timestamp}.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.085, 1, 0.887), h_pad=1.9, w_pad=1.8)

for stem, xmax in (
    ("physical_validation_rmse", max(latest.values())),
    ("physical_validation_rmse_matched_epochs", shared_epoch),
):
    subtitle.set_text(
        f"ocean_broken: epochs 1–{latest['ocean_broken']}  •  ocean: epochs 1–{latest['ocean']}"
        f"  •  Showing epochs 1–{xmax}"
    )
    for ax in axes.flat:
        ax.set_xlim(1, max(2, xmax))
    for extension in ("png", "pdf"):
        path = OUTPUT / f"{stem}.{extension}"
        fig.savefig(path, dpi=180)
        print(path)
plt.close(fig)
print("Latest epochs:", latest)
