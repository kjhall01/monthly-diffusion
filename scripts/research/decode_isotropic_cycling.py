"""Decode archived finalists to physical indices; never regenerate forecasts.

All prognostic fields are decoded transiently with the public model API and
correct masks; only ENSO/PDO/NAO/QBO scalars are retained. Annual output shards
are resumable, and the selected years/members/methods are recorded explicitly.
"""

from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


from monthly_diffusion.experiments.experiment_isotropic_cycling_sweep import OUTPUT


from monthly_diffusion.experiments.decode_isotropic_cycling import run

# Do not compete for the GPU with the latent experiments. The confirmation
# statistics are written only after both 100-year cohorts have finished.
complete = OUTPUT / "confirmation_pooled/ranking.csv"
while not complete.exists():
    print("Waiting for both independent 100-year latent confirmations", flush=True)
    time.sleep(30)
decision = json.loads((OUTPUT / "selection.json").read_text())
methods = [f"cycle{n}" for n in decision["confirmation_cycles"]] + [
    "observed",
    "observed_independent",
]
for label in ("confirm_january", "confirm_july"):
    # Fixed first-ten subset, never chosen for favorable statistics. Full
    # 20-member latent trajectories are retained for all 100 years.
    run(label, methods, years=(81, 100), member_count=10)
