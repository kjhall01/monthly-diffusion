"""Read-only checkpoint/data/coastal-SST audit of the matched frozen models.

Distribution statistics use every finite training value, without latitude
weights, matching Preprocessor.fit. Plot KDEs use a fixed uniform sample over
grid cells and months; missing values are excluded, genuine zeros retained.
Nothing here fits or changes a model or its preprocessing.
"""

from pathlib import Path
import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")
import torch
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monthly_diffusion import MonthlyDiffusionV1
from monthly_diffusion.data.sources import open_canonical_state


from monthly_diffusion.experiments.audit_reanalysis_comparability import (
    RUNS,
    architecture,
    distributions,
    coastal_sst,
    spread_and_replay,
    support_audit,
)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", type=Path, default=ROOT / "analysis/era5_e3sm_comparability")
p.add_argument(
    "--stage",
    choices=("all", "data", "coast", "architecture", "spread", "support"),
    default="all",
)
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
models = {
    k: MonthlyDiffusionV1.load(v / "model.pt", device="cpu") for k, v in RUNS.items()
}
sources = {
    k: yaml.safe_load((v / "data_sources.yaml").read_text()) for k, v in RUNS.items()
}
fields = {}
for k, s in sources.items():
    a, o, _ = open_canonical_state(
        s["state_zarr"], expected_attrs=s.get("expected_attrs")
    )
    fields[k] = xr.merge((a, o), compat="no_conflicts")
identity = dict(
    checkpoints={k: m.weight_digest() for k, m in models.items()},
    sources=sources,
    version=1,
)
marker = args.output / "configuration.json"
if marker.exists() and json.loads(marker.read_text()) != identity:
    raise ValueError("Audit identity changed; use a new output directory")
marker.write_text(json.dumps(identity, indent=2))
if args.stage in ("all", "architecture"):
    architecture(models, sources, args.output)
if args.stage in ("all", "data"):
    distributions(models, fields, args.output)
if args.stage in ("all", "coast"):
    coastal_sst(models, fields, sources, args.output)
if args.stage in ("all", "spread"):
    spread_and_replay(models, args.output)
if args.stage in ("all", "support"):
    support_audit(models, args.output)
print("Saved audit:", args.output, flush=True)
