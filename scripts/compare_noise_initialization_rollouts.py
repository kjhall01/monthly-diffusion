"""Small paired latent-only ENSO test: direct Gaussian versus encoded field noise.

Both histories are [current, previous], with independent draws for the two
months. Encoded noise is N(0,1) in preprocessed field coordinates, masked and
conditioned at actual November/December 1966 CO2. The transition receives no
CO2; later calendar dates simply measure elapsed simulation time. Matched
innovations and identical AR(1) initialization isolate the initial histories.

Train-only, monthly/trend-controlled ridge ENSO probes reuse the established
latent-readout implementation. They are independently checked on validation.
Do not extrapolate the fitted trend into synthetic rollouts. Primary metrics
remove each member's own window monthly climatology and linear trend; they do
not remove the ensemble mean. Save a finite-N-corrected ensemble-mean-removal
sensitivity as well. These are latent readouts, not decoded SST forecasts.
"""
from pathlib import Path
import argparse
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import torch
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monthly_diffusion import MonthlyDiffusionV1
from monthly_diffusion.forecast import Forecast
from monthly_diffusion.data.sources import open_canonical_state, open_co2
from monthly_diffusion.data.dataloader import month_features, co2_values
from monthly_diffusion.training.common import atomic_save
from monthly_diffusion.analysis.indices import enso_index
from monthly_diffusion.analysis.spectra import lag_autocorrelation
from diagnose_enso_initialization import fit_readout, design, metrics
from diagnose_latent_attractor import anomalies

COMPONENTS = ("atmosphere", "ocean")
KINDS = ("direct_latent", "encoded_noise")
COLORS = dict(direct_latent="#0072B2", encoded_noise="#D55E00", reference="#222222")
LABELS = dict(direct_latent="Direct latent noise", encoded_noise="Encoded field noise",
              reference="Encoded real training data")


@torch.inference_mode()
def initial_histories(model, source, co2, *, first, members, seed):
    """Noise in preprocessed field units; source is used ONLY for masks."""
    first = np.datetime64(first, "M")
    dates = np.array([first-np.timedelta64(k, "M") for k in (1, 2)]).astype("datetime64[ns]")
    times = xr.DataArray(dates, dims="time", coords=dict(time=dates))
    months = month_features(times).to(model.device)
    forcing = torch.tensor(((co2_values(co2, times)-model.preprocessor.co2_mean)/model.preprocessor.co2_std).values,
                           dtype=torch.float32, device=model.device)[:, None]
    statics = model.preprocessor.static_tensor(model.statics).to(model.device)[None].expand(2, -1, -1, -1)
    histories = []
    for i, component in enumerate(COMPONENTS):
        normalized = model.preprocessor.encode(source[i].sel(time=dates), component).compute()
        valid = np.isfinite(np.stack([normalized[v].transpose("time", "lat", "lon").values
                                     for v in model.names[component]], axis=1))
        net = getattr(model.autoencoder, component)
        mask = torch.tensor(valid[:, 0], device=model.device) if net.mask_aware else None
        rng = np.random.default_rng(seed+1000+i)
        draws = []
        for _ in range(members):
            x = rng.standard_normal(valid.shape).astype("float32")
            x[~valid] = np.nan
            latent = net.encode(torch.tensor(x, device=model.device), statics, months, forcing, node_mask=mask)
            draws.append(model.latent_normalizers[component].normalize(latent).cpu())
        histories.append(torch.stack(draws))
    return histories


@torch.inference_mode()
def paired_rollouts(model, initial, output, *, first, years, members, seed, co2):
    """Use the package iterator unchanged; checkpoint both groups every year."""
    forecasts = {kind: Forecast(model, None, None, co2=co2, members=members,
                               seed=seed, gaussian_start=first) for kind in KINDS}
    forecasts["encoded_noise"].history = [h.clone().to(model.device) for h in initial]
    initial_path = output/"initial.pt"
    if not initial_path.exists():
        atomic_save({k: f.state_dict() for k, f in forecasts.items()}, initial_path)
    shards = sorted(output.glob("year-*.pt"))
    assert [p.name for p in shards] == [f"year-{i:03d}.pt" for i in range(1, len(shards)+1)]
    if shards:
        previous = torch.load(shards[-1], map_location="cpu", weights_only=False)
        for kind, forecast in forecasts.items():
            forecast.load_state_dict(previous["states"][kind])
            assert forecast.step == len(shards)*12
    tick = time.monotonic()
    start = len(shards)
    for year in range(start+1, years+1):
        blocks = {k: {c: [] for c in COMPONENTS} for k in KINDS}
        dates = []
        for month in range(12):
            for kind, forecast in forecasts.items():
                result = next(forecast)
                expected = np.datetime64(first, "M")+np.timedelta64((year-1)*12+month, "M")
                assert result[0].time.values[0].astype("datetime64[M]") == expected
                for component, z in zip(COMPONENTS, result):
                    if not np.isfinite(z).all():
                        raise FloatingPointError("Nonfinite latent rollout")
                    blocks[kind][component].append(z.values[0])
            # The actual terminal AR(1) noises must stay paired, not merely seeds.
            for n1, n2 in zip(forecasts[KINDS[0]].noise, forecasts[KINDS[1]].noise):
                torch.testing.assert_close(n1, n2, rtol=0, atol=0)
            dates.append(result[0].time.values[0])
        atomic_save(dict(time=np.array(dates), latents={k: {c: np.stack(v) for c, v in d.items()}
                         for k, d in blocks.items()}, states={k: f.state_dict() for k, f in forecasts.items()}),
                    output/f"year-{year:03d}.pt")
        print(f"Paired latent rollouts: {year}/{years} years; "
              f"{(time.monotonic()-tick)/((year-start)*12):.3f}s/paired month", flush=True)


def member_metrics(values):
    """values[time,member]; never concatenate members before ACF or spectrum."""
    x = xr.DataArray(values.T, dims=("member", "time"))
    acf = lag_autocorrelation(x, max_lag=min(60, len(values)-1), detrend="constant").values
    frequency, psd = signal.welch(values, fs=1., nperseg=min(120, len(values)), axis=0, detrend="constant")
    return dict(std=values.std(axis=0), acf=acf, frequency=frequency, psd=psd.T)


def analyze(ds, cache, probes, output):
    references = {}
    rows, curves, memory = [], [], []
    dates = ds.time.values
    n_years = len(dates)//12
    windows = [(1, min(5, n_years)), (6, n_years)]
    windows += [(i, min(i+9, n_years)) for i in range(1, n_years+1, 10)]
    for component in COMPONENTS:
        reference_raw = cache["training"][component].numpy().reshape(len(cache["training_time"]), -1)@probes[component]["coef"]
        ref = anomalies(reference_raw[:, None], cache["training_time"], "window_monthly_detrend")
        references[component] = ref
        ref_stats = member_metrics(ref)
        ref_band = (ref_stats["frequency"] >= 1/84) & (ref_stats["frequency"] <= 1/24)
        ref_band_power = (ref_stats["psd"][:, ref_band].sum(1)
                          *np.diff(ref_stats["frequency"])[0]).mean()
        delta = (ds[f"enso_{component}"].sel(initialization=KINDS[1]).values
                 -ds[f"enso_{component}"].sel(initialization=KINDS[0]).values)
        for step, differences in enumerate(delta, 1):
            memory.append(dict(component=component, lead_month=step,
                               paired_rms_difference=float(np.sqrt(np.mean(differences**2))),
                               relative_to_reference_sd=float(np.sqrt(np.mean(differences**2))/ref.std())))
        for kind in KINDS:
            raw = ds[f"enso_{component}"].sel(initialization=kind).values
            for first_year, last_year in windows:
                sl = slice((first_year-1)*12, last_year*12)
                for method in ("memberwise", "ensemble_mean_removed"):
                    values = raw[sl]
                    if method == "ensemble_mean_removed":
                        n = values.shape[1]
                        values = np.sqrt(n/(n-1))*(values-values.mean(axis=1, keepdims=True))
                    cleaned = anomalies(values, dates[sl], "window_monthly_detrend")
                    stats = member_metrics(cleaned)
                    for member in range(raw.shape[1]):
                        negative_lobe = stats["acf"][member, 12:61]
                        band = (stats["frequency"] >= 1/84) & (stats["frequency"] <= 1/18)
                        peak = stats["frequency"][band][stats["psd"][member, band].argmax()]
                        enso_band = (stats["frequency"] >= 1/84) & (stats["frequency"] <= 1/24)
                        enso_power = stats["psd"][member, enso_band].sum()*np.diff(stats["frequency"])[0]
                        rows.append(dict(component=component, initialization=kind, method=method,
                            first_year=first_year, last_year=last_year, member=member,
                            std=float(stats["std"][member]), reference_std=float(ref.std()),
                            std_ratio=float(stats["std"][member]/ref.std()),
                            band_2_7yr_std_ratio=float(np.sqrt(enso_power/ref_band_power)),
                            acf_12=float(stats["acf"][member, 12]),
                            acf_24=float(stats["acf"][member, 24]),
                            min_acf_12_60=float(negative_lobe.min()),
                            min_acf_lag=int(12+negative_lobe.argmin()),
                            peak_period_18_84_months=float(1/peak)))
                    if first_year == 6 and method == "memberwise":
                        curves.append(xr.Dataset(dict(
                            acf=(("member", "lag_months"), stats["acf"]),
                            psd=(("member", "frequency"), stats["psd"])),
                            coords=dict(member=ds.member, lag_months=np.arange(stats["acf"].shape[1]),
                                        frequency=stats["frequency"])).expand_dims(component=[component], initialization=[kind]))
    table = pd.DataFrame(rows)
    table.to_csv(output/"window_member_metrics.csv", index=False)
    pd.DataFrame(memory).to_csv(output/"paired_initialization_memory.csv", index=False)
    summary = table.groupby(["component", "initialization", "method", "first_year", "last_year"])[
        ["std", "std_ratio", "band_2_7yr_std_ratio", "acf_12", "acf_24", "min_acf_12_60"]].mean().reset_index()
    summary.to_csv(output/"window_summary.csv", index=False)
    xr.combine_by_coords(curves).to_netcdf(output/"late_spectra_acf.nc")
    xr.Dataset({c: (("training_time",), x[:, 0]) for c, x in references.items()},
               coords=dict(training_time=cache["training_time"])).to_netcdf(output/"reference_enso_readouts.nc")

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for i, component in enumerate(COMPONENTS):
        ref = references[component]
        ref_stats = member_metrics(ref)
        for kind in KINDS:
            selected = table[(table.component == component) & (table.initialization == kind) &
                             (table.method == "memberwise") & table.first_year.isin([1, 11, 21]) &
                             (table.last_year > 5)]
            for _, group in selected.groupby("first_year"):
                mid = (group.first_year.iloc[0]+group.last_year.iloc[0])/2
                mean = float(group.std_ratio.mean())
                axes[i, 0].errorbar(mid, mean, yerr=[[mean-group.std_ratio.min()], [group.std_ratio.max()-mean]],
                                   color=COLORS[kind], marker="o", capsize=3)
            late = anomalies(ds[f"enso_{component}"].sel(initialization=kind).values[60:], dates[60:], "window_monthly_detrend")
            stats = member_metrics(late)
            axes[i, 2].plot(stats["frequency"][1:]*12, stats["psd"].mean(0)[1:]/12,
                            color=COLORS[kind], label=LABELS[kind])
            axes[i, 3].plot(np.arange(stats["acf"].shape[1]), stats["acf"].mean(0), color=COLORS[kind])
            axes[i, 3].fill_between(np.arange(stats["acf"].shape[1]), stats["acf"].min(0), stats["acf"].max(0),
                                    color=COLORS[kind], alpha=.10)
            bins = np.linspace(-4*ref.std(), 4*ref.std(), 101)
            density, _ = np.histogram(late, bins=bins, density=True)
            axes[i, 1].plot((bins[1:]+bins[:-1])/2, gaussian_filter1d(density, 1.2), color=COLORS[kind])
        density, _ = np.histogram(ref, bins=bins, density=True)
        axes[i, 1].plot((bins[1:]+bins[:-1])/2, gaussian_filter1d(density, 1.2), color=COLORS["reference"])
        axes[i, 2].plot(ref_stats["frequency"][1:]*12, ref_stats["psd"].mean(0)[1:]/12,
                        color=COLORS["reference"], label=LABELS["reference"])
        axes[i, 3].plot(np.arange(ref_stats["acf"].shape[1]), ref_stats["acf"].mean(0), color=COLORS["reference"])
        axes[i, 0].axhline(1, color="black", ls="--", lw=.8)
        axes[i, 3].axhline(0, color="black", lw=.7)
        axes[i, 0].set(xlabel="Simulation year · 10-year windows", ylabel=f"{component.title()} ENSO\nSD / encoded reference SD", ylim=(0, None))
        axes[i, 1].set(xlabel="ENSO readout (°C)", ylabel="Density")
        axes[i, 2].set(xscale="log", yscale="log", xlabel="Period (years)", ylabel="PSD (°C² / cycles yr⁻¹)")
        axes[i, 2].set_xticks([.1, .2, .5, 1, 2, 6], ["10", "5", "2", "1", "0.5", "0.17"])
        axes[i, 2].xaxis.set_minor_formatter(NullFormatter())
        axes[i, 2].set_xlim(.1, 6)
        axes[i, 3].set(xlabel="Lag (months)", ylabel="Autocorrelation", ylim=(-.6, 1.05))
        for ax in axes[i]:
            ax.grid(alpha=.15)
    for ax, title in zip(axes[0], ("Amplitude · member range", f"Years 6–{n_years}", f"Years 6–{n_years}", f"Years 6–{n_years} · member range")):
        ax.set_title(title, fontsize=11)
    fig.suptitle("ERA5/ORAS5 · paired noise-initialization ENSO readouts\n"
                 "Own monthly climatology + linear trend removed per member/window; no ensemble-mean subtraction", fontsize=12)
    fig.legend(*axes[0, 2].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False)
    for suffix in ("png", "pdf"):
        fig.savefig(output/f"enso_comparison.{suffix}", dpi=190, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 6), constrained_layout=True)
    for ax, component in zip(axes, COMPONENTS):
        reference_raw = cache["training"][component].numpy().reshape(len(cache["training_time"]), -1)@probes[component]["coef"]
        training_months = pd.DatetimeIndex(cache["training_time"]).month
        climate = np.array([reference_raw[training_months == m].mean() for m in range(1, 13)])
        for kind in KINDS:
            values = ds[f"enso_{component}"].sel(initialization=kind).values
            values = values-climate[pd.DatetimeIndex(dates).month-1, None]
            ax.plot(np.arange(len(dates))/12, values[:, 0], color=COLORS[kind], label=LABELS[kind], lw=1)
        ax.axvspan(0, 5, color="gray", alpha=.10)
        ax.axhline(0, color="black", lw=.6)
        ax.set(ylabel=f"{component.title()} ENSO (°C)", xlabel="Simulation year")
        ax.grid(alpha=.15)
    fig.suptitle("Paired member 1 · fixed training-month means removed\nOffsets and initial adjustment retained; gray = first five years", fontsize=12)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False)
    for suffix in ("png", "pdf"):
        fig.savefig(output/f"enso_sample_trajectories.{suffix}", dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(summary[(summary.first_year == 6) & (summary.method == "memberwise")].to_string(index=False), flush=True)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT/"runs/mdv1-era5-oras5-sst-support")
    p.add_argument("--output", type=Path, default=ROOT/"analysis/encoded_vs_direct_noise_enso_rollouts")
    p.add_argument("--years", type=int, default=30)
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.years < 10 or args.members < 2:
        raise ValueError("Need >=10 years and >=2 members")
    torch.set_num_threads(4)
    model = MonthlyDiffusionV1.load(args.run/"model.pt", device=args.device)
    model.autoencoder.eval(); model.transition.eval()
    cache = torch.load(args.run/"latents.pt", map_location="cpu", weights_only=False)
    assert cache["ae_digest"] == model.weight_digest(autoencoder_only=True)
    sources = yaml.safe_load((args.run/"data_sources.yaml").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(checkpoint=str(args.run.resolve()/"model.pt"), checkpoint_sha256=model.weight_digest(),
                    stages=model.stages, config=model.config.to_dict(), sources=sources,
                    first="1967-01", history=["1966-12", "1966-11"], years=args.years,
                    members=args.members, seed=args.seed, device=args.device,
                    noise="Paired diffusion innovations, AR histories initially None; package seven-step sampler unchanged",
                    initialization="Direct N(0,1) transition coordinates vs encoded IID N(0,1) preprocessed fields",
                    readout="Train-only ridge, blocked-year CV selects alpha; monthly intercepts and trend controlled",
                    statistics="Primary: own monthly climatology and linear trend removed per member/window; no ensemble mean removal",
                    limitations="Short five-member latent-only test; no decoded SST and no proof of attractor convergence")
    record = args.output/"configuration.json"
    if record.exists() and json.loads(record.read_text()) != metadata:
        raise ValueError("Diagnostic inputs changed; use a new output directory")
    record.write_text(json.dumps(metadata, indent=2))
    a, o, _ = open_canonical_state(sources["state_zarr"], expected_attrs=sources.get("expected_attrs"))
    co2 = open_co2(sources["co2_netcdf"], expected_sha256=sources.get("co2_sha256"))
    target_path = args.output/"physical_nino34.nc"
    if not target_path.exists():
        enso_index(o[["sst"]].sel(time=slice("1958", "2006"), lat=slice(-5, 5), lon=slice(190, 240)),
                   remove_monthly=False).compute().to_netcdf(target_path)
    target = xr.load_dataarray(target_path)
    probes, skill = {}, []
    for component in COMPONENTS:
        path = args.output/f"{component}_readout.npz"
        if path.exists():
            probe = dict(np.load(path))
        else:
            dates = cache["training_time"]
            x = cache["training"][component].numpy().reshape(len(dates), -1)
            probe = fit_readout(x, target.sel(time=dates).values, dates)
            np.savez(path, **probe)
        probes[component] = probe
        for split in ("training", "validation"):
            dates = cache[split+"_time"]
            x = cache[split][component].numpy().reshape(len(dates), -1)
            prediction = (x-design(dates)@probe["latent_baseline"])@probe["coef"]
            truth = target.sel(time=dates).values-design(dates)@probe["physical_baseline"]
            skill.append(dict(component=component, split=split, alpha=float(probe["alpha"]), **metrics(prediction, truth)))
    pd.DataFrame(skill).to_csv(args.output/"readout_skill.csv", index=False)
    print(pd.DataFrame(skill).to_string(index=False), flush=True)
    initial_path = args.output/"encoded_history.pt"
    if initial_path.exists():
        initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    else:
        initial = initial_histories(model, (a, o), co2, first="1967-01", members=args.members, seed=args.seed)
        atomic_save(initial, initial_path)
    a.close(); o.close()
    paired_rollouts(model, initial, args.output, first="1967-01", years=args.years,
                    members=args.members, seed=args.seed, co2=model.preprocessor.co2_mean)
    shards = [torch.load(p, map_location="cpu", weights_only=False) for p in sorted(args.output.glob("year-*.pt"))]
    fields = {}
    for component in COMPONENTS:
        values = np.stack([np.concatenate([s["latents"][k][component] for s in shards]) for k in KINDS])
        fields[f"latent_{component}"] = (("initialization", "time", "member", "node", "channel"), values)
        projections = values.reshape(*values.shape[:3], -1)@probes[component]["coef"]
        fields[f"enso_{component}"] = (("initialization", "time", "member"), projections)
    ds = xr.Dataset(fields, coords=dict(initialization=list(KINDS), time=np.concatenate([s["time"] for s in shards]),
                    member=np.arange(args.members)), attrs=dict(checkpoint_sha256=model.weight_digest(),
                    readout="Raw projection, no extrapolated trend; see metrics for anomaly removal"))
    ds.to_netcdf(args.output/"rollout_latents_readouts.nc", encoding={v:dict(zlib=True, complevel=2) for v in ds.data_vars})
    analyze(ds, cache, probes, args.output)
    print("Saved", args.output, flush=True)


if __name__ == "__main__":
    main()
