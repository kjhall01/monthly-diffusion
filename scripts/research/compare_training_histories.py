"""Read-only comparison of two staged MDv1 runs, including loss attribution.

Histories contain online training losses and EMA validation losses: their gap
is descriptive, not an exactly matched evaluation of one set of weights.
Each run also uses its own training-fitted physical and latent normalization.
Optional validation replay uses the selected AE's immutable latent cache and
the original per-month SST message mask. It decomposes, but never changes,
the reconstruction objective into cells inside/outside that support.
"""

from pathlib import Path
import argparse
import hashlib
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from monthly_diffusion.experiments.compare_training_histories import (
    RUNS,
    LABELS,
    COLORS,
    save,
    validation_support,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output", type=Path, default=ROOT / "analysis/era5_e3sm_training_comparison"
)
parser.add_argument("--replay-validation", action="store_true")
parser.add_argument("--e3sm-run", type=Path, default=RUNS["e3sm"])
parser.add_argument("--era-run", type=Path, default=RUNS["era"])
args = parser.parse_args()
runs = dict(e3sm=args.e3sm_run.resolve(), era=args.era_run.resolve())
args.output.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
histories, selected, configs, summaries, minima, identity = {}, {}, {}, [], [], {}
for key, run in runs.items():
    configs[key] = yaml.safe_load((run / "config.yaml").read_text())
    identity[key] = dict(
        run=str(run),
        sources=yaml.safe_load((run / "data_sources.yaml").read_text()),
        history_sha256={},
    )
    for stage in ("autoencoder", "diffusion"):
        path = run / stage / "history.csv"
        identity[key]["history_sha256"][stage] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        d = pd.read_csv(path)
        histories[key, stage] = d
        score = "val_loss" if stage == "autoencoder" else "val_transition"
        row = d.loc[d[score].idxmin()]
        chosen = torch.load(
            run / stage / "best.pt", map_location="cpu", weights_only=False
        )
        assert chosen["epoch"] == int(row.epoch)
        np.testing.assert_allclose(chosen["score"], row[score])
        selected[key, stage] = row
        summaries.append(
            dict(
                dataset=key,
                stage=stage,
                epochs_completed=int(d.epoch.max()),
                **{k: float(v) for k, v in row.items()},
                last_val_score=float(d[score].iloc[-1]),
                last20_improvement=float(d[score].iloc[-21] - d[score].iloc[-1]),
                total_hours=float(d.seconds.sum() / 3600),
            )
        )
        for column in d.columns:
            if column.startswith("val_"):
                r = d.loc[d[column].idxmin()]
                minima.append(
                    dict(
                        dataset=key,
                        stage=stage,
                        term=column,
                        minimum=float(r[column]),
                        epoch=int(r.epoch),
                    )
                )
identity["config_differences"] = {
    k: [configs["e3sm"].get(k), configs["era"].get(k)]
    for k in configs["e3sm"].keys() | configs["era"].keys()
    if configs["e3sm"].get(k) != configs["era"].get(k)
}
marker = args.output / "configuration.json"
if marker.exists():
    prior = json.loads(marker.read_text())
    if any(prior.get(k) != identity[k] for k in runs):
        raise ValueError("Comparison inputs changed; choose a new output directory")
marker.write_text(json.dumps(identity, indent=2))
pd.DataFrame(summaries).to_csv(args.output / "selected_checkpoints.csv", index=False)
pd.DataFrame(minima).to_csv(args.output / "per_term_best_epochs.csv", index=False)
rows = []
for component in ("atmosphere", "ocean"):
    specs = [v for v in configs["e3sm"]["variables"] if v["component"] == component]
    for i, spec in enumerate(specs):
        result = dict(component=component, variable=spec["name"], weight=spec["weight"])
        for key in runs:
            for split in ("train", "val"):
                result[key + "_" + split] = selected[key, "autoencoder"][
                    f"{split}_{component}_field{i}"
                ]
        result["weighted_gap"] = result["weight"] * (
            result["era_val"] - result["e3sm_val"]
        )
        rows.append(result)
fields = pd.DataFrame(rows)
gap = selected["era", "autoencoder"].val_loss - selected["e3sm", "autoencoder"].val_loss
np.testing.assert_allclose(fields.weighted_gap.sum(), gap, rtol=2e-6)
fields["fraction_of_total_gap"] = fields.weighted_gap / gap
fields["relative_mse_percent"] = 100 * (fields.era_val / fields.e3sm_val - 1)
fields.to_csv(args.output / "autoencoder_field_attribution.csv", index=False)
fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
for row, stage in enumerate(("autoencoder", "diffusion")):
    terms = (
        ["loss", "atmosphere_reconstruction", "ocean_reconstruction"]
        if stage == "autoencoder"
        else ["transition", "atmosphere_transition", "ocean_transition"]
    )
    for ax, term, title in zip(axes[row], terms, ("Total", "Atmosphere", "Ocean")):
        for key in runs:
            d = histories[key, stage]
            for split, linestyle in (("train", "--"), ("val", "-")):
                ax.plot(
                    d.epoch,
                    d[f"{split}_{term}"],
                    color=COLORS[key],
                    ls=linestyle,
                    lw=1.5,
                )
            chosen = selected[key, stage]
            ax.scatter(
                chosen.epoch, chosen[f"val_{term}"], color=COLORS[key], s=40, zorder=4
            )
        ax.set(
            title=f"{stage.title()} · {title}",
            xlabel="Epoch",
            ylabel="Weighted loss" if term == "loss" else "Mean MSE",
        )
        ax.set_yscale("log")
        ax.grid(alpha=0.18)
handles = [Line2D([], [], color=COLORS[k], label=LABELS[k]) for k in runs]
handles += [
    Line2D([], [], color="gray", ls=ls, label=label)
    for ls, label in (("--", "Online training"), ("-", "EMA validation"))
]
fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
fig.suptitle(
    "Matched architectures · separate data-fitted normalizations\nAE: physical transformed-coordinate error; diffusion: normalized-latent velocity loss",
    fontsize=12,
)
save(fig, args.output / "training_histories")
fig, axes = plt.subplots(1, 2, figsize=(14, 8), constrained_layout=True)
y = np.arange(len(fields))
width = 0.35
for key, offset in (("e3sm", -width / 2), ("era", width / 2)):
    axes[0].barh(
        y + offset, fields[key + "_val"], width, color=COLORS[key], label=LABELS[key]
    )
axes[0].set(
    yticks=y,
    yticklabels=fields.variable,
    xlabel="Validation MSE · own training coordinates",
    title="Selected autoencoder: per-variable errors",
)
axes[0].invert_yaxis()
axes[0].legend(frameon=False)
axes[1].barh(
    y,
    fields.weighted_gap,
    color=np.where(fields.weighted_gap >= 0, "#D55E00", "#009E73"),
)
axes[1].set(
    yticks=y,
    yticklabels=fields.variable,
    xlabel="Contribution to ERA − E3SM total validation loss",
    title=f"Exact loss-gap attribution · total {gap:.5f}",
)
axes[1].invert_yaxis()
axes[1].axvline(0, color="black", lw=0.7)
for ax in axes:
    ax.grid(axis="x", alpha=0.18)
axes[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
save(fig, args.output / "autoencoder_loss_attribution")
fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 7.5),
    constrained_layout=True,
    gridspec_kw=dict(width_ratios=(1, 1)),
)
all_relative = fields.relative_mse_percent.to_numpy()
relative_span = max(np.ptp(all_relative), 20.0)
relative_limits = (
    min(all_relative.min(), 0) - 0.18 * relative_span,
    max(all_relative.max(), 0) + 0.18 * relative_span,
)
for ax, component in zip(axes, ("atmosphere", "ocean")):
    part = fields[fields.component == component]
    y = np.arange(len(part))
    values = part.relative_mse_percent.to_numpy()
    ax.barh(y, values, color=np.where(values > 0, "#D55E00", "#0072B2"), height=0.65)
    ax.axvline(0, color="black", lw=0.9)
    ax.set(
        yticks=y,
        yticklabels=part.variable,
        title=component.title(),
        xlabel="ERA5/ORAS5 MSE relative to E3SMv2 (%)",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.15)
    span = relative_span
    ax.set_xlim(*relative_limits)
    for j, v in enumerate(values):
        ax.text(
            v + (0.02 * span if v >= 0 else -0.02 * span),
            j,
            f"{v:+.1f}%",
            ha="left" if v >= 0 else "right",
            va="center",
            fontsize=9,
        )
ae_epochs = {k: int(selected[k, "autoencoder"].epoch) for k in runs}
support = configs["era"].get("ocean_support", "per_variable")
fig.suptitle(
    f"Autoencoder validation · ERA5/ORAS5 epoch {ae_epochs['era']} vs E3SMv2 epoch {ae_epochs['e3sm']}\n"
    f"1958–1965 · ERA ocean support: {support}",
    fontsize=13,
)
fig.supxlabel(
    "Blue: lower MSE · Orange: higher MSE · Each model uses its own fitted normalization and valid domain",
    fontsize=10,
)
save(fig, args.output / "autoencoder_relative_errors")
fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
for i, component in enumerate(("atmosphere", "ocean")):
    for j in range(3):
        ax = axes[i, j]
        for key in runs:
            d = histories[key, "diffusion"]
            for split, ls in (("train", "--"), ("val", "-")):
                ax.plot(
                    d.epoch,
                    d[f"{split}_{component}_transition_step{j + 1}"],
                    color=COLORS[key],
                    ls=ls,
                )
            ax.axvline(
                selected[key, "diffusion"].epoch, color=COLORS[key], lw=0.7, alpha=0.6
            )
        ax.set(
            title=f"{component.title()} · step {j + 1}",
            xlabel="Epoch",
            ylabel="Raw diffusion MSE",
            ylim=(0.07, 0.65) if i else (0.3, 1.05),
        )
        ax.grid(alpha=0.18)
fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
save(fig, args.output / "diffusion_horizons")
print(
    fields.sort_values("weighted_gap", ascending=False).to_string(index=False),
    flush=True,
)
if args.replay_validation:
    validation_support(args.output, histories, runs)
support_path = args.output / "validation_support.csv"
if support_path.exists():
    support = pd.read_csv(support_path)
    totals = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    ocean = support[support.component == "ocean"]
    e3sm = ocean[ocean.dataset == "e3sm"].set_index("variable")
    era = ocean[ocean.dataset == "era"].set_index("variable").loc[e3sm.index]
    y = np.arange(len(e3sm))
    axes[0].barh(
        y - 0.18,
        e3sm.total,
        0.36,
        color=COLORS["e3sm"],
        label="E3SMv2 · all valid cells",
    )
    axes[0].barh(
        y + 0.18,
        era.inside,
        0.36,
        color=COLORS["era"],
        label="ERA · inside SST support",
    )
    axes[0].barh(
        y + 0.18,
        era.outside,
        0.36,
        left=era.inside,
        color="#E69F00",
        label="ERA · outside SST support",
    )
    axes[0].set(
        yticks=y,
        yticklabels=e3sm.index,
        xlabel="Validation MSE · original full-support denominator",
        title="Ocean reconstruction loss attribution",
    )
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, fontsize=9)
    for i, key in enumerate(runs):
        s = support[support.dataset == key]
        inside = float((s.inside * s.weight).sum())
        outside = float((s.outside * s.weight).sum())
        totals[key] = dict(
            full_loss=inside + outside,
            outside_sst_contribution=outside,
            inside_support_renormalized_loss=float((s.inside_mse * s.weight).sum()),
        )
        axes[1].bar(i, inside, color=COLORS[key], width=0.55)
        axes[1].bar(i, outside, bottom=inside, color="#E69F00", width=0.55)
        axes[1].text(
            i, inside + outside + 0.001, f"{inside + outside:.4f}", ha="center"
        )
    fraction = (
        totals["era"]["outside_sst_contribution"]
        - totals["e3sm"]["outside_sst_contribution"]
    ) / gap
    totals["outside_support_fraction_of_logged_loss_gap"] = fraction
    totals["caveat"] = (
        "Diagnostic restriction changes valid support, not model weights or fitted statistics; not a repaired-model evaluation."
    )
    (args.output / "support_attribution_summary.json").write_text(
        json.dumps(totals, indent=2)
    )
    axes[1].set(
        xticks=[0, 1],
        xticklabels=[LABELS[k] for k in runs],
        ylabel="Weighted total validation loss",
        ylim=(0, 0.075),
        title=f"Outside-support terms account for {fraction:.0%} of loss gap",
    )
    for ax in axes:
        ax.grid(axis="y" if ax == axes[1] else "x", alpha=0.18)
    fig.suptitle(
        "Saved-checkpoint replay · all 96 validation months · no retraining or mask changes",
        fontsize=12,
    )
    save(fig, args.output / "support_loss_attribution")
print("Saved", args.output, flush=True)
