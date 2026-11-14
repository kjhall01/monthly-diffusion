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

# Make the repository workflow helpers available when running this file directly.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from monthly_diffusion.checkpoints import load_checkpoint

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from workflows.legacy_model import MonthlyDiffusionV1
from workflows.data.sources import open_canonical_state, open_co2
from workflows.data.dataloader import month_features, co2_values


from workflows.experiments.compare_encoded_noise import COMPONENTS, monthly_metrics, plot_results

with torch.inference_mode():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run", type=Path, default=ROOT / "runs/mdv1-era5-oras5-sst-support"
    )
    p.add_argument(
        "--output", type=Path, default=ROOT / "analysis/encoded_noise_era5_sst_support"
    )
    p.add_argument("--years", type=int, default=8)
    p.add_argument("--draws", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=20260930)
    args = p.parse_args()
    if args.years < 2 or args.draws < 2 or args.batch_size < 1:
        raise ValueError("Need >=2 years, >=2 draws, positive batch size")
    torch.set_num_threads(4)
    model = MonthlyDiffusionV1.load(args.run / "model.pt", device=args.device)
    model.autoencoder.eval()
    cache = load_checkpoint(args.run / "latents.pt", map_location="cpu", weights_only=False)
    assert cache["ae_digest"] == model.weight_digest(autoencoder_only=True)
    sources = yaml.safe_load((args.run / "data_sources.yaml").read_text())
    dates = pd.DatetimeIndex(cache["training_time"])
    years = np.unique(dates.year)
    years = years[
        np.unique(np.linspace(0, len(years) - 1, args.years).round().astype(int))
    ]
    positions = np.flatnonzero(np.isin(dates.year, years))
    selected_dates = dates[positions]
    assert len(selected_dates) == len(years) * 12
    metadata = dict(
        run=str(args.run.resolve()),
        ae_digest=cache["ae_digest"],
        stages=model.stages,
        selected_years=years.tolist(),
        draws_per_date=args.draws,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        sources=sources,
        noise="IID N(0,1) in transformed, centered, standardized field coordinates",
        masks="Exact per-variable finite masks of matched real source dates; SST support applied",
        conditioning="Matched real source month and CO2; normal checkpoint statics",
        reference="Same selected training dates; all training latents also saved",
        metrics="Unweighted latent coordinates; per-node monthly means and ddof=1 temporal variance",
        caveat="Initialization-only; no test of spin-up or attractor convergence",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    record = args.output / "configuration.json"
    if record.exists() and json.loads(record.read_text()) != metadata:
        raise ValueError("Inputs changed; choose a new output directory")
    record.write_text(json.dumps(metadata, indent=2))
    destination = args.output / "latents.nc"
    if not destination.exists():
        atmos, ocean, _ = open_canonical_state(
            sources["state_zarr"], expected_attrs=sources.get("expected_attrs")
        )
        co2 = open_co2(sources["co2_netcdf"], expected_sha256=sources.get("co2_sha256"))
        times = xr.DataArray(
            selected_dates.values, dims="time", coords=dict(time=selected_dates.values)
        )
        months = month_features(times).to(model.device)
        forcing = torch.tensor(
            (
                (co2_values(co2, times) - model.preprocessor.co2_mean)
                / model.preprocessor.co2_std
            ).values,
            dtype=torch.float32,
            device=model.device,
        )[:, None]
        variables = {}
        tick = time.monotonic()
        for ci, (component, source) in enumerate(zip(COMPONENTS, (atmos, ocean))):
            rng = np.random.default_rng(args.seed + ci)
            direct_rng = np.random.default_rng(args.seed + 100 + ci)
            network = getattr(model.autoencoder, component)
            reference = cache["training"][component].numpy()[positions]
            encoded, zero = [], []
            for start in range(0, len(times), args.batch_size):
                stop = min(start + args.batch_size, len(times))
                physical = source.sel(time=times.values[start:stop])
                normalized = model.preprocessor.encode(physical, component).compute()
                values = np.stack(
                    [
                        normalized[v].transpose("time", "lat", "lon").values
                        for v in model.names[component]
                    ],
                    axis=1,
                )
                valid = np.isfinite(values)
                mask = (
                    torch.tensor(valid[:, 0], device=model.device)
                    if network.mask_aware
                    else None
                )
                context = (
                    months[start:stop],
                    forcing[start:stop],
                )
                if start == 0:
                    replay = model.latent_normalizers[component].normalize(
                        network.encode(
                            torch.tensor(values, device=model.device),
                            *context,
                            node_mask=mask,
                        )
                    )
                    np.testing.assert_allclose(
                        replay.cpu().numpy(),
                        reference[start:stop],
                        rtol=0.005,
                        atol=0.002,
                    )
                    print(component, "cached-reference replay passed", flush=True)
                draws = []
                for draw in range(args.draws + 1):
                    x = (
                        rng.standard_normal(values.shape).astype("float32")
                        if draw < args.draws
                        else np.zeros_like(values)
                    )
                    x[~valid] = np.nan
                    z = network.encode(
                        torch.tensor(x, device=model.device), *context, node_mask=mask
                    )
                    z = model.latent_normalizers[component].normalize(z).cpu().numpy()
                    if not np.isfinite(z).all():
                        raise FloatingPointError("Nonfinite encoded noise")
                    if draw < args.draws:
                        draws.append(z)
                    else:
                        zero.append(z)
                encoded.append(np.stack(draws, axis=1))
                if stop % 12 == 0 or stop == len(times):
                    print(
                        f"{component}: {stop}/{len(times)} dates, {time.monotonic() - tick:.1f}s elapsed",
                        flush=True,
                    )
            noise = np.concatenate(encoded)
            variables[f"real_{component}"] = (("time", "node", "channel"), reference)
            variables[f"all_training_{component}"] = (
                ("training_time", "node", "channel"),
                cache["training"][component].numpy(),
            )
            variables[f"encoded_noise_{component}"] = (
                ("time", "draw", "node", "channel"),
                noise,
            )
            variables[f"encoded_zero_{component}"] = (
                ("time", "node", "channel"),
                np.concatenate(zero),
            )
            variables[f"direct_latent_{component}"] = (
                ("time", "draw", "node", "channel"),
                direct_rng.standard_normal(noise.shape).astype("float32"),
            )
        ds = xr.Dataset(
            variables,
            coords=dict(
                time=times.values,
                training_time=dates.values,
                draw=np.arange(args.draws),
                node=np.arange(reference.shape[-2]),
                channel=np.arange(reference.shape[-1]),
            ),
            attrs=dict(
                ae_digest=cache["ae_digest"], normalized="fixed transition coordinates"
            ),
        )
        ds.to_netcdf(
            destination,
            encoding={v: dict(zlib=True, complevel=2) for v in ds.data_vars},
        )
        atmos.close()
        ocean.close()
    ds = xr.load_dataset(destination)
    rows, conditional = [], []
    for component in COMPONENTS:
        for month in range(1, 13):
            m = ds.time.dt.month.values == month
            real = ds[f"real_{component}"].values[m]
            for kind in ("encoded_noise", "direct_latent", "encoded_zero"):
                candidate = (
                    ds[f"{kind}_{component}"].values[m].reshape(-1, *real.shape[1:])
                )
                for channel in ["all", *range(real.shape[-1])]:
                    r = real if channel == "all" else real[..., channel : channel + 1]
                    c = (
                        candidate
                        if channel == "all"
                        else candidate[..., channel : channel + 1]
                    )
                    rows.append(
                        dict(
                            component=component,
                            month=month,
                            kind=kind,
                            channel=channel,
                            **monthly_metrics(r, c),
                        )
                    )
            noise = ds[f"encoded_noise_{component}"].values[m]
            zero = ds[f"encoded_zero_{component}"].values[m]
            conditional.append(
                dict(
                    component=component,
                    month=month,
                    fixed_condition_noise_rms_std=float(
                        np.sqrt(noise.var(axis=1, ddof=1).mean())
                    ),
                    real_within_month_rms_std=float(
                        np.sqrt(real.var(axis=0, ddof=1).mean())
                    ),
                    noise_minus_zero_rms=float(
                        np.sqrt(np.mean((noise - zero[:, None]) ** 2))
                    ),
                )
            )
    table = pd.DataFrame(rows)
    table.to_csv(args.output / "monthly_metrics.csv", index=False)
    pd.DataFrame(conditional).to_csv(
        args.output / "fixed_condition_noise_spread.csv", index=False
    )
    plot_results(ds, table, args.output)
    print(
        table[table.channel == "all"]
        .groupby(["component", "kind"])[
            ["std_ratio", "mean_pattern_error_in_reference_sd"]
        ]
        .mean()
        .to_string(),
        flush=True,
    )
    print("Saved", args.output, flush=True)
