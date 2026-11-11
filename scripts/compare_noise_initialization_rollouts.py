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

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monthly_diffusion import MonthlyDiffusionV1
from monthly_diffusion.data.sources import open_canonical_state, open_co2
from monthly_diffusion.training.common import atomic_save
from monthly_diffusion.analysis.indices import enso_index
from monthly_diffusion.experiments.diagnose_enso_initialization import (
    fit_readout,
    design,
    metrics,
)


from monthly_diffusion.experiments.compare_noise_initialization_rollouts import (
    COMPONENTS,
    KINDS,
    initial_histories,
    paired_rollouts,
    analyze,
)

with torch.inference_mode():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run", type=Path, default=ROOT / "runs/mdv1-era5-oras5-sst-support"
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "analysis/encoded_vs_direct_noise_enso_rollouts",
    )
    p.add_argument("--years", type=int, default=30)
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.years < 10 or args.members < 2:
        raise ValueError("Need >=10 years and >=2 members")
    torch.set_num_threads(4)
    model = MonthlyDiffusionV1.load(args.run / "model.pt", device=args.device)
    model.autoencoder.eval()
    model.transition.eval()
    cache = torch.load(args.run / "latents.pt", map_location="cpu", weights_only=False)
    assert cache["ae_digest"] == model.weight_digest(autoencoder_only=True)
    sources = yaml.safe_load((args.run / "data_sources.yaml").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        checkpoint=str(args.run.resolve() / "model.pt"),
        checkpoint_sha256=model.weight_digest(),
        stages=model.stages,
        config=model.config.to_dict(),
        sources=sources,
        first="1967-01",
        history=["1966-12", "1966-11"],
        years=args.years,
        members=args.members,
        seed=args.seed,
        device=args.device,
        noise="Paired diffusion innovations, AR histories initially None; package seven-step sampler unchanged",
        initialization="Direct N(0,1) transition coordinates vs encoded IID N(0,1) preprocessed fields",
        readout="Train-only ridge, blocked-year CV selects alpha; monthly intercepts and trend controlled",
        statistics="Primary: own monthly climatology and linear trend removed per member/window; no ensemble mean removal",
        limitations="Short five-member latent-only test; no decoded SST and no proof of attractor convergence",
    )
    record = args.output / "configuration.json"
    if record.exists() and json.loads(record.read_text()) != metadata:
        raise ValueError("Diagnostic inputs changed; use a new output directory")
    record.write_text(json.dumps(metadata, indent=2))
    a, o, _ = open_canonical_state(
        sources["state_zarr"], expected_attrs=sources.get("expected_attrs")
    )
    co2 = open_co2(sources["co2_netcdf"], expected_sha256=sources.get("co2_sha256"))
    target_path = args.output / "physical_nino34.nc"
    if not target_path.exists():
        enso_index(
            o[["sst"]].sel(
                time=slice("1958", "2006"), lat=slice(-5, 5), lon=slice(190, 240)
            ),
            remove_monthly=False,
        ).compute().to_netcdf(target_path)
    target = xr.load_dataarray(target_path)
    probes, skill = {}, []
    for component in COMPONENTS:
        path = args.output / f"{component}_readout.npz"
        if path.exists():
            probe = dict(np.load(path))
        else:
            dates = cache["training_time"]
            x = cache["training"][component].numpy().reshape(len(dates), -1)
            probe = fit_readout(x, target.sel(time=dates).values, dates)
            np.savez(path, **probe)
        probes[component] = probe
        for split in ("training", "validation"):
            dates = cache[split + "_time"]
            x = cache[split][component].numpy().reshape(len(dates), -1)
            prediction = (x - design(dates) @ probe["latent_baseline"]) @ probe["coef"]
            truth = (
                target.sel(time=dates).values
                - design(dates) @ probe["physical_baseline"]
            )
            skill.append(
                dict(
                    component=component,
                    split=split,
                    alpha=float(probe["alpha"]),
                    **metrics(prediction, truth),
                )
            )
    pd.DataFrame(skill).to_csv(args.output / "readout_skill.csv", index=False)
    print(pd.DataFrame(skill).to_string(index=False), flush=True)
    initial_path = args.output / "encoded_history.pt"
    if initial_path.exists():
        initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    else:
        initial = initial_histories(
            model, (a, o), co2, first="1967-01", members=args.members, seed=args.seed
        )
        atomic_save(initial, initial_path)
    a.close()
    o.close()
    paired_rollouts(
        model,
        initial,
        args.output,
        first="1967-01",
        years=args.years,
        members=args.members,
        seed=args.seed,
        co2=model.preprocessor.co2_mean,
    )
    shards = [
        torch.load(p, map_location="cpu", weights_only=False)
        for p in sorted(args.output.glob("year-*.pt"))
    ]
    fields = {}
    for component in COMPONENTS:
        values = np.stack(
            [
                np.concatenate([s["latents"][k][component] for s in shards])
                for k in KINDS
            ]
        )
        fields[f"latent_{component}"] = (
            ("initialization", "time", "member", "node", "channel"),
            values,
        )
        projections = values.reshape(*values.shape[:3], -1) @ probes[component]["coef"]
        fields[f"enso_{component}"] = (
            ("initialization", "time", "member"),
            projections,
        )
    ds = xr.Dataset(
        fields,
        coords=dict(
            initialization=list(KINDS),
            time=np.concatenate([s["time"] for s in shards]),
            member=np.arange(args.members),
        ),
        attrs=dict(
            checkpoint_sha256=model.weight_digest(),
            readout="Raw projection, no extrapolated trend; see metrics for anomaly removal",
        ),
    )
    ds.to_netcdf(
        args.output / "rollout_latents_readouts.nc",
        encoding={v: dict(zlib=True, complevel=2) for v in ds.data_vars},
    )
    analyze(ds, cache, probes, args.output)
    print("Saved", args.output, flush=True)
