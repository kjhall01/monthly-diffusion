"""Additional comparisons after the independent cycling experiments finish.

Run from frozen/: python scripts/research/summarize_isotropic_cycling.py
This reads existing trajectories only. It never loads a model for inference.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import xarray as xr
from monthly_diffusion.experiments.diagnose_isotropic_cycling_sweep import (
    OUTPUT,
    compare,
    plot,
    pool_confirmation,
    anomaly_sensitivity,
    joint_and_seasonal,
    paired_separation,
)


names = ("confirm_january", "confirm_july")
for name in (*names, "confirmation_pooled"):
    folder = OUTPUT / name
    ds = xr.load_dataset(folder / "mode_trajectories.nc")
    if len(ds.time) != 1200:
        raise ValueError(f"Incomplete confirmation: {name}")
    joint_and_seasonal(ds, folder, (60, 100))
    anomaly_sensitivity(ds, folder, (60, 100))
    plot(ds, folder, window=(60, 100))
    sensitivity = folder / "spinup_sensitivity"
    sensitivity.mkdir(exist_ok=True)
    compare(
        ds,
        sensitivity,
        windows={
            "years06_25": (5, 25),
            "years11_30": (10, 30),
            "years21_40": (20, 40),
            "years41_60": (40, 60),
        },
    )
    if name in names:
        paired_separation(folder)

# Physical indices exist only in the last 20 years. Do not score the NaNs
# outside that explicitly decoded window or merge January/July calendars.
physical = [OUTPUT / n / "physical_years81_100_members10" for n in names]
if not all((p / "ranking.csv").exists() for p in physical):
    print("Physical decoding not complete yet; latent summaries are ready.")
    raise SystemExit(0)
out = OUTPUT / "physical_confirmation_pooled"
ds = pool_confirmation(
    physical,
    out,
    windows={"years81_100": (80, 100)},
    plot_window=(80, 100),
    cohort_names=list(names),
)
anomaly_sensitivity(ds, out, (80, 100))
rows = []
for name, folder in (("latent", OUTPUT / "confirmation_pooled"), ("physical", out)):
    metrics = pd.read_csv(folder / "agreement_metrics.csv")
    period = "years61_100" if name == "latent" else "years81_100"
    metrics = metrics[metrics.period == period].copy()
    metrics["space"] = name
    rows.append(metrics)
pd.concat(rows, ignore_index=True).to_csv(
    OUTPUT / "confirmation_summary.csv", index=False
)
