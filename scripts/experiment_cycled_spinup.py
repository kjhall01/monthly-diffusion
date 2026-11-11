"""Paired, resumable test of temporary AE cycling as climate initialization.

Four arms: encoded-noise / 30-month cycling then latent-only / always cycling /
observed encoded Nov-Dec 1958 histories. All use matched monthly innovations,
unchanged AR1 noise, fixed 1850 CO2 for decoding/cycling, and identical calendars.
Observation histories retain their actual-CO2 encoding from the checkpoint's
validation cache; they are coherent modern states, not preindustrial truth.
No retraining. Annual files save EVERY predicted/fed-back latent, physical
Nino3.4 and Tahiti-Darwin pressure indices, and full forecast restart states.
No physical maps are archived. All-field physical decode/re-encode remains
identical to the pilot; only the current history slot is replaced.
"""

from pathlib import Path
import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monthly_diffusion import MonthlyDiffusionV1


from monthly_diffusion.experiments.experiment_cycled_spinup import (
    COMPONENTS,
    METHODS,
    run_experiment,
    analyze,
)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument(
    "--output",
    type=Path,
    default=ROOT / "analysis/cycled_initialization_ensemble_30years",
)
p.add_argument(
    "--pilot",
    type=Path,
    default=ROOT / "analysis/cycled_noise_initialization_era5_30months",
)
p.add_argument(
    "--readouts",
    type=Path,
    default=ROOT / "analysis/encoded_vs_direct_noise_enso_rollouts",
)
p.add_argument("--years", type=int, default=30)
p.add_argument("--members", type=int, default=5)
p.add_argument("--seed", type=int, default=20260716)
p.add_argument("--device", default="mps")
p.add_argument("--no-analysis", action="store_true")
p.add_argument("--analysis-only", action="store_true")
args = p.parse_args()
torch.set_num_threads(4)
source = json.loads((args.pilot / "configuration.json").read_text())
probe_config = json.loads((args.readouts / "configuration.json").read_text())
if source["weight_digest"] != probe_config["checkpoint_sha256"]:
    raise ValueError("Probe checkpoint mismatch")
run = Path(source["run"])
cache = torch.load(run / "latents.pt", map_location="cpu", weights_only=False)
record = dict(
    checkpoint=str(run / "model.pt"),
    weight_digest=source["weight_digest"],
    ae_digest=cache["ae_digest"],
    years=args.years,
    members=args.members,
    seed=args.seed,
    device=args.device,
    first="1800-01",
    cycle_months=30,
    co2_ppm=source["co2_ppm"],
    methods=list(METHODS),
    history_order="current, previous",
    observation_source="Cached actual-CO2 encoded November/December 1958, same checkpoint; months aligned, simulation years relabeled",
    pairing="Identical monthly terminal innovations and AR1 memory across treatments; cycle30/continuous exactly identical until month30",
    storage="All monthly predicted and fed-back latents, physical scalar indices, annual restart states; NO full decoded fields",
    scope="Temporary cycling changes only initialization; continuous cycling changes effective model dynamics",
    anomalies="Fixed training monthly baselines; primary diagnostics retain level shifts; no linear detrending or ensemble-mean subtraction",
)
args.output.mkdir(parents=True, exist_ok=True)
path = args.output / "configuration.json"
if path.exists() and json.loads(path.read_text()) != record:
    raise ValueError("Configuration changed; choose a new output directory")
path.write_text(json.dumps(record, indent=2))
if not args.analysis_only:
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no CPU fallback")
    model = MonthlyDiffusionV1.load(run / "model.pt", device=args.device)
    if (
        model.weight_digest() != source["weight_digest"]
        or model.weight_digest(autoencoder_only=True) != cache["ae_digest"]
    ):
        raise ValueError("Checkpoint changed")
    run_experiment(
        model,
        cache,
        args.output,
        years=args.years,
        members=args.members,
        seed=args.seed,
        first="1800-01",
        ppm=source["co2_ppm"],
    )
    del model
if not args.no_analysis:
    probes = {
        c: np.load(args.readouts / f"{c}_readout.npz")["coef"] for c in COMPONENTS
    }
    analyze(
        args.output,
        cache,
        probes,
        args.pilot / "coupling/encoded_reference_indices.nc",
        30,
    )
