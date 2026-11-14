"""Inspect retained spin-up latents without decoding, refitting or inference.

Primary anomalies subtract the FIXED encoded-training calendar-month mean,
per node/channel. No spin-up monthly climatology, linear trend, ensemble mean
or extrapolated training trend is removed. ENSO probes are the already fitted
training-only ridge readouts, not physical SST. PSD/ACF center their own windows
as required for these statistics, but never linearly detrend them. Window means
are retained separately so a persistent displacement cannot hide in the ACF.
"""

# Make the repository workflow helpers available when running this file directly.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from monthly_diffusion.checkpoints import load_checkpoint

import argparse
import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from workflows.analysis.long_rollout_plots import finish


from workflows.experiments.diagnose_spinup_latents import COMPONENTS, COLORS, fixed_monthly_anomalies, window_statistics, load_latent_trajectory

p = argparse.ArgumentParser(description=__doc__)
p.add_argument(
    "--forecast",
    type=Path,
    default=ROOT
    / "runs/mdv1-era5-oras5-sst-support/long_rollout/ssp370_1850_2100_5member_encoded_noise_spinup1800",
)
p.add_argument(
    "--readouts",
    type=Path,
    default=ROOT / "analysis/encoded_vs_direct_noise_enso_rollouts",
)
p.add_argument("--output", type=Path)
p.add_argument(
    "--through-year",
    type=int,
    help="Append saved monthly restart latents through December of this year; no physical data are read",
)
args = p.parse_args()
torch.set_num_threads(4)
output = args.output or args.forecast / (
    "spinup_diagnostics"
    if args.through_year is None
    else f"latent_diagnostics_through{args.through_year}"
)
output.mkdir(parents=True, exist_ok=True)
metadata = json.loads((args.forecast / "forecast.json").read_text())
probes_metadata = json.loads((args.readouts / "configuration.json").read_text())
if metadata["checkpoint_sha256"] != probes_metadata["checkpoint_sha256"]:
    raise ValueError("Readouts were fitted on a different checkpoint")
checkpoint = Path(probes_metadata["checkpoint"])
payload = load_checkpoint(checkpoint, map_location="cpu", weights_only=False)
cache = load_checkpoint(
    checkpoint.parent / "latents.pt", map_location="cpu", weights_only=False
)
digest = hashlib.sha256()
for name, tensor in payload["autoencoder"].items():
    digest.update(name.encode())
    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
if digest.hexdigest() != cache["ae_digest"]:
    raise ValueError("Training latent cache and checkpoint autoencoder differ")
for name, tensor in payload["transition"].items():
    digest.update(name.encode())
    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
if digest.hexdigest() != metadata["checkpoint_sha256"]:
    raise ValueError("Selected checkpoint changed after this rollout")
del payload

raw, times = load_latent_trajectory(args.forecast, metadata, args.through_year)
initial = load_checkpoint(
    args.forecast / "initial_state.pt", map_location="cpu", weights_only=False
)
train_dates = np.asarray(cache["training_time"])
train_months = pd.DatetimeIndex(train_dates).month.to_numpy()
months = pd.DatetimeIndex(times).month.to_numpy()
ntime, members, nodes, channels = raw["ocean"].shape
years = ntime // 12
year0 = pd.Timestamp(times[0]).year
phase = "Spin-up" if args.through_year is None else "Simulation"
period_label = f"{pd.Timestamp(times[0]).year}–{pd.Timestamp(times[-1]).year}"
# Decade summaries use only complete ten-year windows; an extra final year
# still appears in trajectories and the last-20-year dynamics comparison.
decade_starts = range(0, ntime - 119, 120)
comparison_windows = [
    ("Encoded training reference", None, "#222222"),
    (
        f"{phase} years 1–20"
        if args.through_year is None
        else f"{year0}–{year0 + 19} (early spin-up)",
        slice(0, 240),
        "#D55E00",
    ),
]
if args.through_year is not None:
    lo, hi = metadata["spinup_months"] - 240, metadata["spinup_months"]
    label = f"{pd.Timestamp(times[lo]).year}–{pd.Timestamp(times[hi - 1]).year} (late spin-up)"
    comparison_windows.append((label, slice(lo, hi), "#009E73"))
comparison_windows.append(
    (
        f"{phase} years {years - 19}–{years}"
        if args.through_year is None
        else f"{pd.Timestamp(times[-240]).year}–{pd.Timestamp(times[-1]).year}",
        slice(-240, None),
        "#0072B2",
    )
)
probes, readouts, references, residuals, ref_residuals, initial_readouts = (
    {},
    {},
    {},
    {},
    {},
    {},
)
rows, channel_rows, coupled_rows, products = [], [], [], {}
for ci, c in enumerate(COMPONENTS):
    if not np.isfinite(raw[c]).all():
        raise FloatingPointError(f"Nonfinite {c} spin-up latents")
    training = cache["training"][c].numpy().astype(float)
    coef = np.load(args.readouts / f"{c}_readout.npz")["coef"]
    probes[c] = coef
    residuals[c], ref_residuals[c] = fixed_monthly_anomalies(
        raw[c], months, training, train_months
    )
    readouts[c] = residuals[c].reshape(ntime, members, -1) @ coef
    references[c] = ref_residuals[c].reshape(len(training), -1) @ coef
    initial_month = pd.Timestamp(initial["times"][0]).month
    initial_readouts[c] = (
        initial["history"][ci][:, 0].numpy()
        - training[train_months == initial_month].mean(0)
    ).reshape(members, -1) @ coef
    ref_sd = float(references[c].std())
    ref_temporal_rms = float(np.sqrt(ref_residuals[c].var(axis=0).mean()))
    products[f"{c}_enso"] = (("time", "member"), readouts[c])
    products[f"{c}_training_enso"] = (("training_time",), references[c])
    products[f"{c}_initial_enso"] = (("member",), initial_readouts[c])
    products[f"{c}_anomaly_rms"] = (
        ("time", "member"),
        np.sqrt(np.mean(residuals[c] ** 2, axis=(2, 3))) / ref_temporal_rms,
    )
    for start in decade_starts:
        end = min(start + 120, ntime)
        field, x = residuals[c][start:end], readouts[c][start:end]
        _, acf = window_statistics(x)
        # Temporal variance FIRST at each node/channel/member, then average
        # features. Pooling the fixed spatial climatology would inflate it.
        spread = np.sqrt(field.var(axis=0).mean(axis=(1, 2))) / ref_temporal_rms
        shift = (
            np.sqrt(np.mean(field.mean(axis=0) ** 2, axis=(1, 2))) / ref_temporal_rms
        )
        for j in range(members):
            rows.append(
                dict(
                    component=c,
                    member=j + 1,
                    first_year=start // 12 + 1,
                    last_year=end // 12,
                    first_calendar_year=year0 + start // 12,
                    last_calendar_year=year0 + end // 12 - 1,
                    enso_mean=float(x[:, j].mean()),
                    enso_std=float(x[:, j].std()),
                    reference_std=ref_sd,
                    enso_std_ratio=float(x[:, j].std() / ref_sd),
                    acf24=float(acf.isel(member=j).sel(lag_months=24)),
                    latent_temporal_sd_ratio=float(spread[j]),
                    latent_mean_pattern_rms_ratio=float(shift[j]),
                )
            )
        for ch in range(channels):
            channel_rows.append(
                dict(
                    component=c,
                    channel=ch + 1,
                    first_year=start // 12 + 1,
                    last_year=end // 12,
                    raw_mean=float(raw[c][start:end, ..., ch].mean()),
                    raw_std=float(raw[c][start:end, ..., ch].std()),
                    fixed_month_anomaly_mean=float(field[..., ch].mean()),
                    temporal_sd_ratio=float(
                        np.sqrt(
                            field[..., ch].var(axis=0).mean()
                            / ref_residuals[c][..., ch].var(axis=0).mean()
                        )
                    ),
                )
            )
for start in decade_starts:
    for j in range(members):
        coupled_rows.append(
            dict(
                first_year=start // 12 + 1,
                last_year=min(start + 120, ntime) // 12,
                member=j + 1,
                enso_atmos_ocean_correlation=float(
                    np.corrcoef(
                        readouts["atmosphere"][start : start + 120, j],
                        readouts["ocean"][start : start + 120, j],
                    )[0, 1]
                ),
            )
        )
table = pd.DataFrame(rows)
table.to_csv(output / "window_member_metrics.csv", index=False)
summary = (
    table.groupby(["component", "first_year", "last_year"])
    .mean(numeric_only=True)
    .drop(columns="member")
    .reset_index()
)
summary.to_csv(output / "window_summary.csv", index=False)
pd.DataFrame(channel_rows).to_csv(output / "channel_metrics.csv", index=False)
pd.DataFrame(coupled_rows).to_csv(output / "cross_component_enso.csv", index=False)
ds = xr.Dataset(
    products,
    coords=dict(time=times, member=np.arange(members), training_time=train_dates),
)
ds.attrs.update(
    checkpoint_sha256=metadata["checkpoint_sha256"],
    ae_digest=cache["ae_digest"],
    readout="fixed training-only ridge; Celsius equivalent, NOT decoded SST",
    anomalies="fixed 1966-2006 encoded-training calendar-month mean; no trend or ensemble mean removed",
    spinup_co2_ppm=float(metadata["spinup_co2_ppm"][0]),
    window_statistics="std and ACF center each member/window; no linear detrending",
)
ppm = (
    metadata["spinup_co2_ppm"]
    + metadata["co2_ppm"][: max(0, ntime - metadata["spinup_months"])]
)
ds["co2_ppm"] = ("time", np.asarray(ppm[:ntime]))
ds["is_spinup"] = (
    "time",
    (np.arange(ntime) < metadata["spinup_months"]).astype("int8"),
)
ds.to_netcdf(
    output / "readouts_and_adjustment.nc",
    encoding={n: dict(zlib=True, complevel=2) for n in ds.data_vars},
)
record = dict(
    forecast=str(args.forecast.resolve()),
    readouts=str(args.readouts.resolve()),
    checkpoint_sha256=metadata["checkpoint_sha256"],
    completed_spinup_months=min(ntime, metadata["spinup_months"]),
    included_months=ntime,
    through_year=args.through_year,
    dates=[str(times[0]), str(times[-1])],
    years=years,
    members=members,
    training=[1966, 2006],
    anomalies=ds.attrs["anomalies"],
    inference="none; read saved spin-up and optional monthly restart latents only",
    spectra="Welch 120-month Hann segments, constant detrend; ACF constant detrend, max lag 60 months",
    comparison=[name for name, _, _ in comparison_windows],
    decade_summaries="complete 10-year windows only; final partial decade appears in trajectory and last-20-year statistics",
    caveat="training-reference offsets are not themselves evidence of continuing drift; no attractor or stationarity proof",
)
(output / "configuration.json").write_text(json.dumps(record, indent=2))

# Show EVERY member and the initial state, with no smoothing or re-centering.
fig, axes = plt.subplots(
    members,
    2,
    figsize=(15, 2 * members + 1.3),
    sharex=True,
    sharey=True,
    constrained_layout=True,
    squeeze=False,
)
elapsed = np.arange(ntime + 1) / 12
xcoord = elapsed if args.through_year is None else year0 + elapsed
for j in range(members):
    for ci, c in enumerate(COMPONENTS):
        ax = axes[j, ci]
        sd = references[c].std()
        ax.axhspan(-sd, sd, color="0.9", label="Encoded reference ±1 SD")
        ax.plot(
            xcoord,
            np.r_[initial_readouts[c][j], readouts[c][:, j]],
            color=COLORS[j % len(COLORS)],
            lw=0.85,
        )
        ax.axhline(0, color="0.35", lw=0.6)
        if args.through_year is None:
            ax.axvline(10, color="0.6", ls=":", lw=0.7)
        else:
            ax.axvline(1850, color="0.3", ls="--", lw=0.9)
        ax.set(xlim=(xcoord[0], xcoord[-1]))
        ax.grid(alpha=0.12)
        if ci == 0:
            ax.set_ylabel(f"Member {j + 1}\nENSO (°C-equivalent)")
        if j == 0:
            ax.set_title(f"{c.title()} latent ENSO readout")
        if j == members - 1:
            ax.set_xlabel(
                "Years after initialization"
                if args.through_year is None
                else "Simulation calendar year"
            )
fig.suptitle(
    f"Encoded-noise {phase.lower()} · {period_label} · all members · fixed training-month anomalies\n"
    "No detrending or ensemble-mean removal; gray: reference ±1 SD"
    + ("; dashed: end of spin-up (1850)" if args.through_year else ""),
    fontsize=12,
)
finish(fig, output / "enso_all_member_trajectories")

fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
for ci, c in enumerate(COMPONENTS):
    for j in range(members):
        window = table[(table.component == c) & (table.member == j + 1)]
        mid = (window.first_year + window.last_year - 1) / 2
        if args.through_year is not None:
            mid = mid + year0
        axes[ci, 0].plot(
            mid,
            window.enso_mean,
            "o-",
            color=COLORS[j % len(COLORS)],
            lw=1,
            label=f"Member {j + 1}",
        )
        axes[ci, 1].plot(
            mid, window.enso_std_ratio, "o-", color=COLORS[j % len(COLORS)], lw=1
        )
        axes[ci, 2].plot(
            mid,
            window.latent_temporal_sd_ratio,
            "o-",
            color=COLORS[j % len(COLORS)],
            lw=1,
        )
    axes[ci, 0].axhline(0, color="0.3", ls="--", lw=0.8)
    for ax in axes[ci, 1:]:
        ax.axhline(1, color="0.3", ls="--", lw=0.8)
        ax.set_ylim(bottom=0)
    for ax in axes[ci]:
        ax.set(
            xlabel=("Spin-up year" if args.through_year is None else "Calendar year")
            + " · complete 10-year windows",
            xlim=(xcoord[0], xcoord[-1]),
        )
        if args.through_year is not None:
            ax.axvline(1850, color="0.3", ls=":", lw=0.9)
        ax.grid(alpha=0.15)
    axes[ci, 0].set_ylabel(f"{c.title()}\nENSO mean (°C-equivalent)")
    axes[ci, 1].set_ylabel("ENSO SD / encoded-reference SD")
    axes[ci, 2].set_ylabel("Latent temporal RMS SD / reference")
for ax, title in zip(
    axes[0],
    [
        "Persistent ENSO displacement",
        "ENSO fluctuation amplitude",
        "All latent nodes/channels",
    ],
):
    ax.set_title(title)
fig.legend(
    *axes[0, 0].get_legend_handles_labels(),
    loc="outside lower center",
    ncol=members,
    frameon=False,
)
fig.suptitle(
    f"{phase} adjustment · {period_label} · fixed training climatology · no linear detrending",
    fontsize=13,
)
finish(fig, output / "window_adjustment")

# Fixed-reference offsets remain in the PDFs. Only PSD/ACF center windows.
fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
curves = []
metric_rows = []
for ci, c in enumerate(COMPONENTS):
    windows = [
        (label, references[c][:, None] if sl is None else readouts[c][sl], color)
        for label, sl, color in comparison_windows
    ]
    all_values = np.concatenate([x.ravel() for _, x, _ in windows])
    density_x = np.linspace(all_values.min() - 0.2, all_values.max() + 0.2, 500)
    for label, x, color in windows:
        axes[ci, 0].plot(
            density_x, gaussian_kde(x.ravel())(density_x), color=color, label=label
        )
        psd, acf = window_statistics(x)
        period = 1 / (12 * psd.frequency.values[1:])
        axes[ci, 1].plot(period, psd.mean("member").values[1:], color=color)
        axes[ci, 2].plot(acf.lag_months / 12, acf.mean("member"), color=color)
        if x.shape[1] > 1:
            axes[ci, 2].fill_between(
                acf.lag_months / 12,
                acf.min("member"),
                acf.max("member"),
                color=color,
                alpha=0.10,
            )
        curves.append(
            xr.Dataset(dict(psd=psd, acf=acf)).expand_dims(
                component=[c], window=[label]
            )
        )
        for j in range(x.shape[1]):
            metric_rows.append(
                dict(
                    component=c,
                    window=label,
                    member=j + 1,
                    mean=float(x[:, j].mean()),
                    std=float(x[:, j].std()),
                    std_ratio=float(x[:, j].std() / references[c].std()),
                    acf24=float(acf.isel(member=j).sel(lag_months=24)),
                )
            )
    axes[ci, 0].set(
        xlabel="ENSO readout (°C-equivalent)", ylabel=f"{c.title()}\nDensity"
    )
    axes[ci, 1].set(
        xscale="log",
        yscale="log",
        xlabel="Period (years)",
        ylabel="PSD (°C-equivalent² month)",
    )
    axes[ci, 1].set_xticks(
        [0.25, 0.5, 1, 2, 5, 10], ["0.25", "0.5", "1", "2", "5", "10"]
    )
    axes[ci, 2].axhline(0, color=".4", lw=0.7)
    axes[ci, 2].set(xlabel="Lag (years)", ylabel="Autocorrelation", ylim=(-0.8, 1.05))
    for ax in axes[ci]:
        ax.grid(alpha=0.15)
for ax, title in zip(
    axes[0], ["Distribution (offsets retained)", "Spectrum", "Lag ACF · member range"]
):
    ax.set_title(title)
fig.legend(
    *axes[0, 0].get_legend_handles_labels(),
    loc="outside lower center",
    ncol=2 if args.through_year else 3,
    frameon=False,
)
fig.suptitle(
    f"Latent ENSO · {period_label} · early / late comparison\nFixed training-month anomalies; PSD/ACF remove window means, not linear trends",
    fontsize=12,
)
finish(fig, output / "enso_early_late_dynamics")
pd.DataFrame(metric_rows).to_csv(output / "early_late_member_metrics.csv", index=False)
# Reference has one trajectory, spin-up has five; preserve missing members,
# don't invent extra reference trajectories when combining coordinates.
for d in curves:
    d.coords["member"] = np.arange(d.sizes["member"])
xr.combine_by_coords(curves, join="outer").to_netcdf(output / "enso_spectra_acf.nc")

# Numerical sensitivity ONLY. Do not silently hide these slow shifts in
# the main plots: record how much own-window climatology/trend removal
# changes the ACF, using the same treatment for the reference.
sensitivity = []
for c in COMPONENTS:
    for label, sl, _ in comparison_windows:
        x = references[c][:, None] if sl is None else readouts[c][sl]
        dates = train_dates if sl is None else times[sl]
        calendar = pd.DatetimeIndex(dates).month.to_numpy() - 1
        matrix = np.column_stack((np.eye(12)[calendar], np.linspace(-1, 1, len(dates))))
        residual = x - matrix @ np.linalg.lstsq(matrix, x, rcond=None)[0]
        _, acf = window_statistics(residual)
        for j in range(x.shape[1]):
            sensitivity.append(
                dict(
                    component=c,
                    window=label,
                    member=j + 1,
                    method="own-window monthly climatology + linear trend removed",
                    std=float(residual[:, j].std()),
                    retained_variance_fraction=float(
                        residual[:, j].var() / x[:, j].var()
                    ),
                    acf24=float(acf.isel(member=j).sel(lag_months=24)),
                )
            )
pd.DataFrame(sensitivity).to_csv(output / "detrending_sensitivity.csv", index=False)

fig, axes = plt.subplots(
    2, channels, figsize=(3.7 * channels, 6.4), constrained_layout=True, squeeze=False
)
for ci, c in enumerate(COMPONENTS):
    for ch in range(channels):
        ax = axes[ci, ch]
        arrays = [
            (
                "Encoded training reference",
                cache["training"][c][..., ch].numpy().ravel(),
                "#222222",
            ),
            ("Spin-up years 1–10", raw[c][:120, ..., ch].ravel(), "#D55E00"),
            (
                f"{phase} years {years - 9}–{years}"
                if args.through_year is None
                else f"{pd.Timestamp(times[-120]).year}–{pd.Timestamp(times[-1]).year}",
                raw[c][-120:, ..., ch].ravel(),
                "#0072B2",
            ),
        ]
        limits = np.concatenate([x for _, x, _ in arrays])
        bins = np.linspace(limits.min(), limits.max(), 151)
        for label, values, color in arrays:
            density, edges = np.histogram(values, bins=bins, density=True)
            ax.plot(
                (edges[1:] + edges[:-1]) / 2, density, color=color, label=label, lw=1.1
            )
        ax.set(title=f"Channel {ch + 1}", xlabel="Fixed-normalized latent value")
        ax.grid(alpha=0.12)
        if ch == 0:
            ax.set_ylabel(f"{c.title()}\nDensity")
fig.legend(
    *axes[0, 0].get_legend_handles_labels(),
    loc="outside lower center",
    ncol=3,
    frameon=False,
)
fig.suptitle(
    "Raw latent distributions · pooled over nodes, members and time\nNo anomaly removal; matching these marginals does not establish matching temporal dynamics",
    fontsize=12,
)
finish(fig, output / "latent_channel_distributions")
print(summary.to_string(index=False))
print(f"Figures and retained numeric products: {output}")
