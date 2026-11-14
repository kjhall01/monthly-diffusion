"""ENSO/PDO/NAO/QBO audit of the saved, paired initialization experiments.

No new forecasts: fit train-only, validation-checked latent ridge readouts,
then stream the saved *predicted* latents through the unchanged decoder.
All physical indices use package definitions and one training-only PDO EOF.
Primary anomalies subtract fixed training-month means, with no detrending
and no ensemble-mean subtraction. Compare each stochastic trajectory before
averaging statistics, not statistics of the ensemble-mean trajectory.
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
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from workflows.legacy_model import MonthlyDiffusionV1
from workflows.experiments.experiment_cycle_duration import DISPLAY


from workflows.experiments.diagnose_cycling_modes import MODES, COMPONENTS, ARCHIVES, prepare_readouts, decode_saved, assemble, analyze, render, render_seasonal_sensitivity, render_summary

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--run", type=Path, default=ROOT / "runs/mdv1-era5-oras5-sst-support")
p.add_argument("--output", type=Path, default=ROOT / "analysis/cycling_duration_modes")
p.add_argument("--device", default="mps")
p.add_argument("--prepare-only", action="store_true")
p.add_argument("--analysis-only", action="store_true")
args = p.parse_args()
torch.set_num_threads(4)
args.output.mkdir(parents=True, exist_ok=True)
config = json.loads((ARCHIVES["observed"] / "configuration.json").read_text())
cache = load_checkpoint(args.run / "latents.pt", map_location="cpu", weights_only=False)
if cache["ae_digest"] != config["ae_digest"]:
    raise ValueError("Wrong cached autoencoder")
source = yaml.safe_load((args.run / "data_sources.yaml").read_text())
metadata = dict(
    checkpoint=config["checkpoint"],
    weight_digest=config["weight_digest"],
    ae_digest=config["ae_digest"],
    co2_ppm=config["co2_ppm"],
    methods=list(DISPLAY),
    members=5,
    years=30,
    seed=config["seed"],
    inputs={
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for path in set(ARCHIVES.values())
        for p in sorted(path.glob("year-*.pt"))
    },
    source=source,
    training="1966–2006",
    validation="1958–1965",
    primary="Years 11–30, fixed training-month anomalies",
    sensitivity="Own-window monthly means and linear trend removed; per member",
    pdo="Fixed train-only North-Pacific EOF after month/trend removal; no global-SST subtraction, matching package convention",
)
path = args.output / "configuration.json"
if path.exists() and json.loads(path.read_text()) != metadata:
    raise ValueError("Analysis identity changed; use new output")
path.write_text(json.dumps(metadata, indent=2))
if not args.analysis_only:
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no CPU fallback")
    model = MonthlyDiffusionV1.load(args.run / "model.pt", device=args.device)
    if model.weight_digest() != config["weight_digest"]:
        raise ValueError("Wrong checkpoint")
    probes = prepare_readouts(model, cache, source, args.output)
    if args.prepare_only:
        raise SystemExit(0)
    decode_saved(model, cache, config, args.output)
    del model
else:
    probes = {
        (c, m): dict(np.load(args.output / f"probe_{c}_{m}.npz"))
        for c in COMPONENTS
        for m in MODES
    }
ds = assemble(cache, probes, args.output)
table = analyze(ds, args.output)
render(ds, table, args.output)
render_seasonal_sensitivity(ds, args.output)
render_summary(ds, table, args.output)
(args.output / "code_provenance.json").write_text(
    json.dumps(
        {
            str(Path(__file__).name): hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest()
        },
        indent=2,
    )
)
print(f"Finished: {args.output}", flush=True)
