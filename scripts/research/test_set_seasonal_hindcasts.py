"""Current frozen checkpoints: same-trajectory, test-only DJF hindcasts.

Reuses MonthlyDiffusionV1.initialize/decode unchanged. History is always
[current, previous] inside Forecast; callers supply chronological physical ICs.
Only training winters fit reference/model thresholds. All months of the three
scored fields and both full latent states are archived, allowing other fields
to be decoded later without rerunning stochastic transitions.
"""

# Make the repository workflow helpers available when running this file directly.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import fcntl
import shutil

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from workflows.experiments.test_set_seasonal_hindcasts import infer, score, comparison_map, plot

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--config", type=Path, default=ROOT / "examples/test-set-djf-hindcasts.yaml"
)
parser.add_argument(
    "--experiment", choices=("e3sm", "reanalysis", "both"), default="both"
)
parser.add_argument(
    "--stage", choices=("pilot", "infer", "score", "plot", "all"), default="all"
)
parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
args = parser.parse_args()
torch.set_num_threads(4)
config = yaml.safe_load(args.config.read_text())
output = ROOT / config["output"]
output.mkdir(parents=True, exist_ok=True)
with (output / ".lock").open("w") as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another seasonal hindcast writer is active") from None
    (output / "configuration.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    keys = (
        list(config["experiments"]) if args.experiment == "both" else [args.experiment]
    )
    for key in keys:
        if args.stage in ("pilot", "infer", "all"):
            infer(config, key, args.device, pilot=args.stage == "pilot")
        if args.stage in ("score", "all"):
            score(config, key)
        if args.stage in ("plot", "all"):
            plot(config, key)
    if args.stage in ("plot", "all"):
        comparison_map(config)
        for key in keys:
            shutil.copy2(
                __file__, output / key / "provenance" / "completed_pipeline.py"
            )
