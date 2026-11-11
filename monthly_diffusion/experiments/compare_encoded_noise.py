"""Compare white-noise inputs encoded with matched month/CO2 against real latents.

Noise is IID N(0,1) in the model's PREPROCESSED field coordinates, not physical
units. Preserve each sampled source month's exact finite-data mask. Encode with
the selected AE, normal statics/month/CO2 and immutable transition normalization.
No model fitting, decoding, or rollout. Cached real latents must match this AE.

Monthly comparisons use identical sampled training years. Mean-pattern errors
and temporal spread are calculated per node before pooling: spatial climatology
must not masquerade as variability. These are unweighted latent-coordinate
diagnostics (not physical area-weighted climate metrics). Also retain all 492
real training latents to permit alternate comparisons. IID latent initialization
and encoding zero-valued standardized fields are explanatory controls.
"""

from pathlib import Path
import argparse
import json
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import torch
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[2]
from monthly_diffusion import MonthlyDiffusionV1
from monthly_diffusion.data.sources import open_canonical_state, open_co2
from monthly_diffusion.data.dataloader import month_features, co2_values

COMPONENTS = ("atmosphere", "ocean")
COLORS = dict(
    real="#222222",
    encoded_noise="#D55E00",
    direct_latent="#0072B2",
    encoded_zero="#777777",
)
LABELS = dict(
    real="Encoded real fields",
    encoded_noise="Encoded field noise",
    direct_latent="Direct latent noise",
    encoded_zero="Encoded zero input",
)


def monthly_metrics(real, candidate):
    """[sample,node,channel]; ddof=1 temporal variance, no spatial centering.

    Mean-pattern RMSE is divided by real within-month RMS temporal SD. Spread ratio
    uses EACH distribution's own per-node mean; it does not count a mean bias as
    spread. Real interannual variance includes any within-training-period trend.
    """
    real, candidate = np.asarray(real, float), np.asarray(candidate, float)
    reference_variance = real.var(axis=0, ddof=1).mean()
    bias = candidate.mean(0) - real.mean(0)
    return dict(
        reference_temporal_rms_std=float(np.sqrt(reference_variance)),
        temporal_rms_std=float(np.sqrt(candidate.var(axis=0, ddof=1).mean())),
        std_ratio=float(
            np.sqrt(candidate.var(axis=0, ddof=1).mean() / reference_variance)
        ),
        mean_pattern_rmse=float(np.sqrt(np.mean(bias**2))),
        mean_pattern_error_in_reference_sd=float(
            np.sqrt(np.mean(bias**2) / reference_variance)
        ),
        pooled_mean=float(candidate.mean()),
        pooled_std=float(candidate.std()),
    )


def plot_results(ds, table, output):
    for component in COMPONENTS:
        fig, axes = plt.subplots(4, 3, figsize=(11, 9.8))
        fig.subplots_adjust(
            left=0.09, right=0.98, bottom=0.12, top=0.92, hspace=0.40, wspace=0.23
        )
        for month, ax in enumerate(axes.flat, 1):
            chosen = ds.time.dt.month.values == month
            arrays = {
                "real": ds[f"real_{component}"].values[chosen],
                "encoded_noise": ds[f"encoded_noise_{component}"].values[chosen],
                "direct_latent": ds[f"direct_latent_{component}"].values[chosen],
            }
            joined = np.concatenate([x.ravel() for x in arrays.values()])
            lower, upper = float(joined.min()), float(joined.max())
            bins = np.linspace(
                lower - 0.02 * (upper - lower), upper + 0.02 * (upper - lower), 151
            )
            for kind, values in arrays.items():
                density, _ = np.histogram(values, bins=bins, density=True)
                ax.plot(
                    (bins[1:] + bins[:-1]) / 2,
                    gaussian_filter1d(density, 1),
                    color=COLORS[kind],
                    label=LABELS[kind],
                    lw=1.6,
                    ls="--" if kind == "direct_latent" else "-",
                )
            ax.set_title(pd.Timestamp(2000, month, 1).strftime("%B"), fontsize=10)
            ax.grid(alpha=0.15)
        fig.supxlabel(
            "Transition-normalized latent value · nodes and channels pooled", y=0.066
        )
        fig.supylabel("Density", x=0.02)
        fig.suptitle(
            f"{component.title()} · calendar-month latent distributions", fontsize=13
        )
        fig.legend(
            *axes.flat[0].get_legend_handles_labels(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012),
            ncol=3,
            frameon=False,
        )
        for suffix in ("png", "pdf"):
            fig.savefig(
                output / f"monthly_{component}_distributions.{suffix}",
                dpi=190,
                bbox_inches="tight",
            )
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for i, component in enumerate(COMPONENTS):
        for kind in ("encoded_noise", "direct_latent", "encoded_zero"):
            rows = table[
                (table.component == component)
                & (table.kind == kind)
                & (table.channel == "all")
            ]
            for j, metric in enumerate(
                ("mean_pattern_error_in_reference_sd", "std_ratio")
            ):
                axes[i, j].plot(
                    rows.month,
                    rows[metric],
                    "o-",
                    color=COLORS[kind],
                    label=LABELS[kind],
                    ms=3,
                )
        axes[i, 0].axhline(0, color="black", lw=0.7)
        axes[i, 1].axhline(1, color="black", ls="--", lw=0.8)
        for ax in axes[i]:
            ax.set_xticks(range(1, 13), list("JFMAMJJASOND"))
            ax.grid(alpha=0.15)
        axes[i, 0].set_ylabel(
            f"{component.title()}\nMean-pattern error / real temporal SD"
        )
        axes[i, 1].set_ylabel("Within-month temporal SD / real SD")
    axes[0, 0].set_title("Monthly mean pattern · ideal 0")
    axes[0, 1].set_title("Spread around own monthly mean · ideal 1")
    fig.suptitle("Noise initialization versus encoded training fields", fontsize=13)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            output / f"monthly_latent_comparison.{suffix}", dpi=190, bbox_inches="tight"
        )
    plt.close(fig)
