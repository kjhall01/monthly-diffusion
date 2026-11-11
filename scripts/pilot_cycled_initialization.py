"""Small encoded-noise -> coupled prediction -> decode -> encode spin-up pilot.

One trajectory per branch. Both branches start with identical independent
N(0,1) preprocessed-field histories and use identical stochastic innovations.
The control evolves only in latent space. The cycled branch decodes ALL fields
in physical units and re-encodes them after every generated month, replacing
only the current history slot. Fixed latent normalization, SST support, month,
statics, and CO2 use the public trained-model API, unchanged. No added noise,
annealing, fitting, or changes to the checkpoint. Cycling changes the dynamics;
this short pilot does not establish a stationary climate or ENSO skill.

Save SST, T2M and precipitation snapshots (native physical units) at step zero
and every five steps, plus every month's pre/post-cycle latents and statistics.
Step zero is the decoded encoded-noise current history, NOT the raw noise.
"""
from pathlib import Path
import argparse
import copy
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monthly_diffusion import MonthlyDiffusionV1
from monthly_diffusion.data.sources import open_co2

COMPONENTS = ("atmosphere", "ocean")
VARIABLES = ("sst", "t2m", "mtpr")
BRANCHES = ("cycled", "latent_only")
LABELS = dict(sst="Sea Surface Temperature", t2m="2-meter Temperature", mtpr="Precipitation")
UNITS = dict(sst="°C", t2m="K", mtpr="mm/day")


@torch.inference_mode()
def run_pilot(model, *, co2, first, seed, steps, output):
    forecasts = dict(cycled=model.initialize(co2=co2, members=1, seed=seed, encoded_noise_start=first),
                     latent_only=model.initialize(co2=co2, members=1, seed=seed, gaussian_start=first))
    # Clone the actual encoded state: two separate MPS encoder calls can differ
    # by floating-point reduction roundoff, despite identical input draws.
    initial = copy.deepcopy(forecasts["cycled"].state_dict())
    forecasts["latent_only"].load_state_dict(initial)
    for a, b in zip(forecasts["cycled"].history, forecasts["latent_only"].history):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.save(initial, output/"initial_state.pt")
    snapshots, rows, latent_rows = [], [], []
    clock = time.monotonic()
    for step in range(steps+1):
        for branch, forecast in forecasts.items():
            previous = [h[:, 0].clone() for h in forecast.history]
            if step:
                za, zo = next(forecast)
                for h, before in zip(forecast.history, previous):
                    torch.testing.assert_close(h[:, 1], before, rtol=0, atol=0)
            else:
                values = []
                for h, template in zip(forecast.history, forecast.templates):
                    coords = dict(template.coords)
                    coords.update(time=[forecast.times[0]], member=[0])
                    values.append(xr.DataArray(h[:, 0].cpu().numpy()[None],
                        dims=("time", "member", "node", "channel"), coords=coords,
                        name=template.name, attrs=template.attrs))
                za, zo = values
            # Decode every variable, including inverse centering/scaling and
            # nonlinear transforms. Do not recycle just the three plotted fields.
            atmos, ocean = model.decode(za, zo, co2=co2)
            atmos, ocean = atmos.isel(member=0, drop=True), ocean.isel(member=0, drop=True)
            fields = xr.merge([atmos[["t2m", "mtpr"]], ocean[["sst"]]])
            predicted = [h[:, 0].cpu().clone() for h in forecast.history]
            if branch == "cycled" and step:
                encoded = model.encode(atmos, ocean, co2=co2)
                for i, z in enumerate(encoded):
                    if not np.isfinite(z.values).all():
                        raise FloatingPointError("Nonfinite re-encoded latent")
                    # z has shape [one time, node, channel]; here time is the
                    # single batch member. Previous history/noise/RNG untouched.
                    forecast.history[i][:, 0] = torch.as_tensor(z.values, device=model.device)
                for h, before in zip(forecast.history, previous):
                    torch.testing.assert_close(h[:, 1], before, rtol=0, atol=0)
            latent_rows.append(dict(step=step, branch=branch, time=forecast.times[0],
                predicted=predicted, fed_back=[h[:, 0].cpu().clone() for h in forecast.history]))
            if step % 5 == 0 or step == steps:
                snapshots.append(fields.assign_coords(step=("time", [step])).expand_dims(branch=[branch]))
            row = dict(step=step, branch=branch, time=str(forecast.times[0]))
            for name in VARIABLES:
                field = fields[name].isel(time=0)
                values = field.values
                support = np.array(model.preprocessor.stats[name]["mask"], dtype=bool, copy=True)
                if name == "sst" and model.preprocessor.ocean_support_mask is not None:
                    support &= np.asarray(model.preprocessor.ocean_support_mask, dtype=bool)
                if not np.isfinite(values[support]).all():
                    raise FloatingPointError(f"Nonfinite decoded {name} on valid support")
                weights = np.cos(np.deg2rad(field.lat))
                row[f"{name}_mean"] = float(field.weighted(weights).mean())
                row[f"{name}_min"] = float(field.min())
                row[f"{name}_max"] = float(field.max())
                row[f"{name}_units"] = field.attrs.get("units", "")
            if row["mtpr_min"] < 0:
                raise AssertionError("Inverse-softplus precipitation must be nonnegative")
            for i, component in enumerate(COMPONENTS):
                delta = latent_rows[-1]["fed_back"][i]-predicted[i]
                row[f"{component}_cycle_rms"] = float(delta.square().mean().sqrt())
                row[f"{component}_latent_rms"] = float(forecast.history[i][:, 0].square().mean().sqrt())
            rows.append(row)
        for a, b in zip(forecasts["cycled"].noise, forecasts["latent_only"].noise):
            if a is not None:
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        print(f"Month {step:02d}/{steps}: {str(forecasts['cycled'].times[0])[:7]} | "
              f"{time.monotonic()-clock:.1f}s elapsed", flush=True)
        # Tiny restart/latent records survive interruption; physical snapshots
        # are intentionally limited to the requested fields/dates.
        torch.save(dict(records=latent_rows, states={b:f.state_dict() for b,f in forecasts.items()}),
                   output/"latent_states.pt")
        pd.DataFrame(rows).to_csv(output/"monthly_summary.csv", index=False)
    ds = xr.combine_by_coords(snapshots, combine_attrs="override")
    ds.attrs.update(definition="Decoded predicted states BEFORE same-month re-encoding; step0 is decoded initial encoded noise",
                    conditioning="Constant CO2; calendar advances monthly", co2_ppm=float(co2),
                    history_order="current, previous", seed=seed,
                    cycle="All physical prognostic fields decoded and re-encoded; only current latent replaced")
    ds.to_netcdf(output/"physical_snapshots.nc", encoding={v:dict(zlib=True, complevel=3) for v in VARIABLES})
    return ds, pd.DataFrame(rows)


def plot_results(ds, table, output):
    import cartopy.crs as ccrs
    from cartopy.util import add_cyclic_point
    try:
        from cmocean import cm
        temperature_cmap = cm.thermal
    except ImportError:
        temperature_cmap = "inferno"
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    # Identical scales across dates AND branches, no spatial smoothing.
    limits = {v: (float(ds[v].min()), float(ds[v].max())) for v in VARIABLES}
    limits["mtpr"] = (0., float(np.nanpercentile(ds.mtpr.values*86400., 99.5)))
    for branch in BRANCHES:
        data = ds.sel(branch=branch)
        height = 2.0*len(data.time)+1.4
        fig, axes = plt.subplots(len(data.time), 3, figsize=(12.5, height),
            subplot_kw=dict(projection=ccrs.Robinson()), squeeze=False)
        fig.subplots_adjust(left=.10, right=.985, top=1.-1.15/height,
                            bottom=.075, hspace=.18, wspace=.06)
        for j, name in enumerate(VARIABLES):
            for i, date in enumerate(data.time.values):
                ax = axes[i, j]
                field = data[name].isel(time=i)
                if name == "mtpr":
                    field = field*86400.
                values, lon = add_cyclic_point(field.values, coord=field.lon)
                im = ax.pcolormesh(lon, field.lat, values, transform=ccrs.PlateCarree(),
                    cmap="YlGnBu" if name == "mtpr" else temperature_cmap,
                    vmin=limits[name][0], vmax=limits[name][1], rasterized=True)
                ax.set_global(); ax.set_facecolor("#e5e5e5"); ax.coastlines(linewidth=.35)
                if i == 0:
                    ax.set_title(f"{LABELS[name]} ({UNITS[name]})", pad=9)
                if j == 0:
                    ax.text(-.05, .5, f"Step {int(data.step.isel(time=i)):02d}\n{str(date)[:7]}",
                            transform=ax.transAxes, ha="right", va="center", fontsize=9)
            pos = axes[-1, j].get_position()
            cax = fig.add_axes([pos.x0+.025, .04, pos.width-.05, .011])
            fig.colorbar(im, cax=cax, orientation="horizontal", extend="max" if name == "mtpr" else "neither")
        title = "Coupled prediction → decode → re-encode" if branch == "cycled" else "Control · latent-only coupled prediction"
        fig.suptitle(f"{title}\nEncoded field-noise start · constant CO₂ {ds.attrs['co2_ppm']:.2f} ppm · one trajectory", fontsize=12)
        for ext in ("png", "pdf"):
            fig.savefig(output/f"{branch}_physical_maps.{ext}", dpi=210)
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    for branch, color in zip(BRANCHES, ("#D55E00", "#0072B2")):
        sub = table[table.branch == branch]
        for j, name in enumerate(VARIABLES):
            axes[0, j].plot(sub.step, sub[f"{name}_mean"]*(86400 if name == "mtpr" else 1), color=color, label=branch)
            axes[0, j].set(title=LABELS[name], ylabel=f"Area-weighted mean ({UNITS[name]})")
        for j, component in enumerate(COMPONENTS):
            axes[1, j].plot(sub.step, sub[f"{component}_latent_rms"], color=color, label=branch)
            axes[1, j].set(title=f"{component.title()} latent RMS", ylabel="Fixed normalized coordinates")
    sub = table[table.branch == "cycled"]
    for component in COMPONENTS:
        axes[1, 2].plot(sub.step, sub[f"{component}_cycle_rms"], label=component)
    axes[1, 2].set(title="Decode–re-encode correction", ylabel="Latent RMS change")
    for ax in axes.flat:
        ax.set_xlabel("Generated months"); ax.grid(alpha=.2)
    axes[0, 0].legend(frameon=False); axes[1, 2].legend(frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(output/f"cycling_comparison.{ext}", dpi=210)
    plt.close(fig)
    (output/"plot_metadata.json").write_text(json.dumps(dict(limits=limits,
        precipitation_display="mm/day; 99.5th percentile upper limit pooled across both branches/dates; saturation marked",
        temperatures="Native units; full pooled min/max; no clipping", smoothing="none"), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT/"runs/mdv1-era5-oras5-sst-support")
    p.add_argument("--output", type=Path, default=ROOT/"analysis/cycled_noise_initialization_era5_30months")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--first", default="1800-01")
    p.add_argument("--seed", type=int, default=20260716)
    p.add_argument("--device", default="mps")
    p.add_argument("--plot-only", action="store_true", help="Render saved fields; no checkpoint/GPU/inference")
    args = p.parse_args()
    if args.plot_only:
        plot_results(xr.load_dataset(args.output/"physical_snapshots.nc"),
                     pd.read_csv(args.output/"monthly_summary.csv"), args.output)
        return
    if args.steps < 1:
        raise ValueError("Positive step count required")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable in this process; not silently falling back to CPU")
    torch.set_num_threads(4)
    model = MonthlyDiffusionV1.load(args.run/"model.pt", device=args.device)
    sources = yaml.safe_load((args.run/"data_sources.yaml").read_text())
    co2 = open_co2(sources["co2_netcdf"], expected_sha256=sources.get("co2_sha256"))
    ppm = float(co2.sel(time="1850-01").mean())
    record = dict(run=str(args.run.resolve()), weight_digest=model.weight_digest(), stages=model.stages,
        steps=args.steps, first=args.first, seed=args.seed, device=args.device, members=1,
        co2_ppm=ppm, co2_source=sources["co2_netcdf"], co2_sha256=sources.get("co2_sha256"),
        noise="Original trained terminal-noise amplitudes and AR1 memory; no new field noise",
        interpretation="Initialization pilot only; cycling remains enabled for every generated month")
    args.output.mkdir(parents=True, exist_ok=True)
    config = args.output/"configuration.json"
    if config.exists() and json.loads(config.read_text()) != record:
        raise ValueError("Configuration differs; choose a new output directory")
    config.write_text(json.dumps(record, indent=2))
    if (args.output/"physical_snapshots.nc").exists():
        ds = xr.load_dataset(args.output/"physical_snapshots.nc")
        table = pd.read_csv(args.output/"monthly_summary.csv")
    else:
        print(f"Loaded {args.run.name}: {model.stages}; constant CO2={ppm:.6f} ppm", flush=True)
        ds, table = run_pilot(model, co2=ppm, first=args.first, seed=args.seed, steps=args.steps, output=args.output)
    plot_results(ds, table, args.output)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
