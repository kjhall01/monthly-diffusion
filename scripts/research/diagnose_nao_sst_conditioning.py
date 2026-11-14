"""Test whether decoder conditioning creates an NAO-like SST pattern."""

from __future__ import annotations
# Make the repository workflow helpers available when running this file directly.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from monthly_diffusion.checkpoints import load_checkpoint

import argparse
import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from workflows.legacy_model import MonthlyDiffusionV1
from workflows.data.sources import open_canonical_state, open_co2


from workflows.experiments.diagnose_nao_sst_conditioning import nao_index, regression, compare

p = argparse.ArgumentParser()
p.add_argument("--run", type=Path, default=ROOT / "runs/mdv1-e3smv2-r1i1p1f1")
p.add_argument("--data", type=Path, default=ROOT / "examples/e3sm-local.yaml")
p.add_argument(
    "--output", type=Path, default=ROOT / "analysis/nao_sst_conditioning_epoch159"
)
p.add_argument("--device", default="cpu")
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
source = yaml.safe_load(args.data.read_text())
a, o, s = open_canonical_state(
    source["state_zarr"], expected_attrs=source.get("expected_attrs")
)
co2 = open_co2(source["co2_netcdf"], expected_sha256=source.get("co2_sha256"))
a, o = a.sel(time=slice("1966-01", "2006-12")), o.sel(time=slice("1966-01", "2006-12"))
dates = pd.DatetimeIndex(a.time.values)
model = MonthlyDiffusionV1.load(args.run / "model.pt", device=args.device)
cache = load_checkpoint(args.run / "latents.pt", map_location="cpu", weights_only=False)
z_a = xr.DataArray(
    cache["training"]["atmosphere"].numpy(),
    dims=("time", "node", "channel"),
    coords={"time": dates, "node": np.arange(162), "channel": np.arange(4)},
)
z_o = xr.DataArray(
    cache["training"]["ocean"].numpy(),
    dims=("time", "node", "channel"),
    coords={"time": dates, "node": np.arange(162), "channel": np.arange(4)},
)
physical_index = nao_index(a)
reference = regression(o.sst, physical_index)
mask = o.sst.isel(time=0).notnull()
mean_co2 = float(co2.sel(time=dates).mean())
conditions = {
    "normal": (z_a, z_o, co2.sel(time=dates)),
    "fixed_co2": (
        z_a,
        z_o,
        xr.DataArray(
            np.full(len(dates), mean_co2),
            dims="time",
            coords={"time": dates},
            name="co2",
        ),
    ),
}
# Unique synthetic timestamps are required by the CO2/month lookup; their
# years are bookkeeping only and every decoder input carries July month.
july = pd.date_range("1966-07-01", periods=len(dates), freq="YS")
conditions["fixed_month"] = (
    z_a.assign_coords(time=july),
    z_o.assign_coords(time=july),
    co2.sel(time=dates).assign_coords(time=july),
)
conditions["fixed_co2_month"] = (
    z_a.assign_coords(time=july),
    z_o.assign_coords(time=july),
    xr.DataArray(
        np.full(len(dates), mean_co2), dims="time", coords={"time": july}, name="co2"
    ),
)
rows = []
maps = {}
for name, (za, zo, c) in conditions.items():
    _, decoded_o = model.decode(ocean=zo, co2=c)
    decoded_o = decoded_o.assign_coords(time=dates)
    pattern = regression(decoded_o.sst, physical_index)
    maps[name] = pattern
    rows.append(dict(condition=name, **compare(pattern, reference, mask)))
    print(name, rows[-1], flush=True)
pd.DataFrame(rows).to_csv(args.output / "summary.csv", index=False)
ds = xr.Dataset({"reference": reference, **{k: v for k, v in maps.items()}})
ds.to_netcdf(args.output / "nao_sst_regression_maps.nc")
with plt.rc_context({"font.size": 9}):
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.2), constrained_layout=True)
    fields = {"E3SM reference": reference, **maps}
    vmax = max(float(abs(v).max()) for v in fields.values())
    for ax, (name, v) in zip(axes, fields.items()):
        im = ax.pcolormesh(
            v.lon, v.lat, v, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto"
        )
        ax.set_title(name)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
    fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        label="SST regression on NAO (K per index unit)",
        shrink=0.8,
    )
    fig.savefig(args.output / "nao_sst_conditioning.png", dpi=220)
    fig.savefig(args.output / "nao_sst_conditioning.pdf")
    plt.close(fig)
print(f"Wrote {args.output}")
