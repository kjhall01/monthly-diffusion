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
import json
import sys

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import torch
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monthly_diffusion import MonthlyDiffusionV1
from monthly_diffusion.data.sources import open_co2


from monthly_diffusion.experiments.pilot_cycled_initialization import (
    run_pilot,
    plot_results,
)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--run", type=Path, default=ROOT / "runs/mdv1-era5-oras5-sst-support")
p.add_argument(
    "--output",
    type=Path,
    default=ROOT / "analysis/cycled_noise_initialization_era5_30months",
)
p.add_argument("--steps", type=int, default=30)
p.add_argument("--first", default="1800-01")
p.add_argument("--seed", type=int, default=20260716)
p.add_argument("--device", default="mps")
p.add_argument(
    "--plot-only",
    action="store_true",
    help="Render saved fields; no checkpoint/GPU/inference",
)
args = p.parse_args()
if args.plot_only:
    plot_results(
        xr.load_dataset(args.output / "physical_snapshots.nc"),
        pd.read_csv(args.output / "monthly_summary.csv"),
        args.output,
    )
    raise SystemExit(0)
if args.steps < 1:
    raise ValueError("Positive step count required")
if args.device == "mps" and not torch.backends.mps.is_available():
    raise RuntimeError(
        "MPS unavailable in this process; not silently falling back to CPU"
    )
torch.set_num_threads(4)
model = MonthlyDiffusionV1.load(args.run / "model.pt", device=args.device)
sources = yaml.safe_load((args.run / "data_sources.yaml").read_text())
co2 = open_co2(sources["co2_netcdf"], expected_sha256=sources.get("co2_sha256"))
ppm = float(co2.sel(time="1850-01").mean())
record = dict(
    run=str(args.run.resolve()),
    weight_digest=model.weight_digest(),
    stages=model.stages,
    steps=args.steps,
    first=args.first,
    seed=args.seed,
    device=args.device,
    members=1,
    co2_ppm=ppm,
    co2_source=sources["co2_netcdf"],
    co2_sha256=sources.get("co2_sha256"),
    noise="Original trained terminal-noise amplitudes and AR1 memory; no new field noise",
    interpretation="Initialization pilot only; cycling remains enabled for every generated month",
)
args.output.mkdir(parents=True, exist_ok=True)
config = args.output / "configuration.json"
if config.exists() and json.loads(config.read_text()) != record:
    raise ValueError("Configuration differs; choose a new output directory")
config.write_text(json.dumps(record, indent=2))
if (args.output / "physical_snapshots.nc").exists():
    ds = xr.load_dataset(args.output / "physical_snapshots.nc")
    table = pd.read_csv(args.output / "monthly_summary.csv")
else:
    print(
        f"Loaded {args.run.name}: {model.stages}; constant CO2={ppm:.6f} ppm",
        flush=True,
    )
    ds, table = run_pilot(
        model,
        co2=ppm,
        first=args.first,
        seed=args.seed,
        steps=args.steps,
        output=args.output,
    )
plot_results(ds, table, args.output)
print(f"Saved {args.output}", flush=True)
