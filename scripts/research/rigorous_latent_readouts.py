"""Rigorous frozen-AE readouts for ENSO/E/C/NAO.

Fits blocked-year ridge probes on r1 (1966--2006), then evaluates them on
held-out E3SM members and the r1 validation period.  The probes are diagnostics
only; no model weights are changed.
"""

from __future__ import annotations

# Make the repository workflow helpers available when running this file directly.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from monthly_diffusion.checkpoints import load_checkpoint

import argparse
import gc
import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from workflows.legacy_model import MonthlyDiffusionV1
from workflows.data.sources import open_canonical_state, open_co2


from workflows.experiments.rigorous_latent_readouts import idx, monthly_climatology, anomalies, blocked_probe, score

p = argparse.ArgumentParser()
p.add_argument("--run", type=Path, default=ROOT / "runs/mdv1-e3smv2-r1i1p1f1")
p.add_argument("--data", type=Path, default=ROOT / "examples/e3sm-local.yaml")
p.add_argument(
    "--output", type=Path, default=ROOT / "analysis/rigorous_latent_readouts_epoch159"
)
p.add_argument("--device", default="cpu")
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
source = yaml.safe_load(args.data.read_text())
co2 = open_co2(source["co2_netcdf"], expected_sha256=source.get("co2_sha256"))
model = MonthlyDiffusionV1.load(args.run / "model.pt", device=args.device)
members = ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1", "r4i1p1f1", "r5i1p1f1"]
cache = load_checkpoint(args.run / "latents.pt", map_location="cpu", weights_only=False)
r1_path = source["state_zarr"]
r1a, r1o, _ = open_canonical_state(r1_path, expected_attrs=source.get("expected_attrs"))
r1a_train, r1o_train = (
    r1a.sel(time=slice("1966-01", "2006-12")),
    r1o.sel(time=slice("1966-01", "2006-12")),
)
r1a_val, r1o_val = (
    r1a.sel(time=slice("1958-01", "1965-12")),
    r1o.sel(time=slice("1958-01", "1965-12")),
)
train_dates = pd.DatetimeIndex(r1a_train.time.values)
val_dates = pd.DatetimeIndex(r1a_val.time.values)
encoded_train = (
    cache["training"]["atmosphere"].numpy().reshape(len(train_dates), -1),
    cache["training"]["ocean"].numpy().reshape(len(train_dates), -1),
)
encoded_val = (
    cache["validation"]["atmosphere"].numpy().reshape(len(val_dates), -1),
    cache["validation"]["ocean"].numpy().reshape(len(val_dates), -1),
)
targets_train = idx(r1a_train, r1o_train)
targets_val = idx(r1a_val, r1o_val)
target_names = ["E", "C", "Nino3.4", "SOI_proxy", "NAO"]
target_names = ["E", "C", "Nino3.4", "SOI", "NAO"]
r1a, r1o = encoded_train
r1y = targets_train
yclim = monthly_climatology(r1y, train_dates)
xacl = monthly_climatology(r1a, train_dates)
xocl = monthly_climatology(r1o, train_dates)
rows = []
probes = {}
years = train_dates.year.values
for j, name in enumerate(target_names):
    y = anomalies(r1y[:, j], train_dates, yclim[:, j])
    for component, x, xclim in (("atmosphere", r1a, xacl), ("ocean", r1o, xocl)):
        xa = anomalies(x, train_dates, xclim)
        c, b, alpha, cv = blocked_probe(xa, y, years)
        probes[(name, component)] = (c, b)
        pred = xa @ c + b
        rows.append(
            dict(
                member="r1",
                period="blocked_cv_train",
                mode=name,
                component=component,
                alpha=alpha,
                cv_rmse=cv,
                **score(pred, y),
            )
        )
# The r1 validation interval is already encoded in the frozen cache.
for j, name in enumerate(target_names):
    y = anomalies(targets_val[:, j], val_dates, yclim[:, j])
    for component, x, xclim in (
        ("atmosphere", encoded_val[0], xacl),
        ("ocean", encoded_val[1], xocl),
    ):
        xa = anomalies(x, val_dates, xclim)
        c, b = probes[(name, component)]
        rows.append(
            dict(
                member="r1i1p1f1",
                period="1958-1965",
                mode=name,
                component=component,
                alpha=np.nan,
                cv_rmse=np.nan,
                **score(xa @ c + b, y),
            )
        )
# Encode/evaluate held-out members one at a time so the diagnostic does
# not retain five full decoded source archives or latent arrays in RAM.
for member in members[1:]:
    path = source["state_zarr"].replace("r1i1p1f1", member)
    a, o, _ = open_canonical_state(
        path, expected_attrs={**source.get("expected_attrs", {}), "member_id": member}
    )
    a, o = (
        a.sel(time=slice("1966-01", "2006-12")),
        o.sel(time=slice("1966-01", "2006-12")),
    )
    dates = pd.DatetimeIndex(a.time.values)
    yall = idx(a, o)
    za, zo = model.encode(a, o, co2=co2.sel(time=dates))
    xs = (za.values.reshape(len(dates), -1), zo.values.reshape(len(dates), -1))
    for j, name in enumerate(target_names):
        y = anomalies(yall[:, j], dates, yclim[:, j])
        for component, x, xclim in (
            ("atmosphere", xs[0], xacl),
            ("ocean", xs[1], xocl),
        ):
            xa = anomalies(x, dates, xclim)
            c, b = probes[(name, component)]
            rows.append(
                dict(
                    member=member,
                    period="1966-2006",
                    mode=name,
                    component=component,
                    alpha=np.nan,
                    cv_rmse=np.nan,
                    **score(xa @ c + b, y),
                )
            )
    print(f"encoded/evaluated {member}", flush=True)
    a.close()
    o.close()
    del a, o, za, zo, xs
    gc.collect()
out = pd.DataFrame(rows)
out.to_csv(args.output / "summary.csv", index=False)
xr.Dataset(
    {"target": (("time", "mode"), targets_train)},
    coords={"time": train_dates, "mode": target_names},
).to_netcdf(args.output / "physical_indices_train.nc")
xr.Dataset(
    {"target": (("time", "mode"), targets_val)},
    coords={"time": val_dates, "mode": target_names},
).to_netcdf(args.output / "physical_indices_validation.nc")
print(out.to_string(index=False))
print(f"Wrote {args.output}")
