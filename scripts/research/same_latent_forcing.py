"""Same-latent decoder CO2-response audit for a frozen MonthlyDiffusionV1 AE.

The physical states are encoded at their historical CO2, then decoded twice
with the identical latent, month, and static fields while only the absolute
CO2 coordinate is replaced by month-matched early and late-period means.
This is a decoder-only forced-response diagnostic; no transition rollout is
performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import xarray as xr
import yaml

from monthly_diffusion import MonthlyDiffusionV1
from monthly_diffusion.data.sources import open_canonical_state, open_co2
from monthly_diffusion.data.preprocessing import LatentNormalizer
from monthly_diffusion.analysis.spatial import weighted_mean


from monthly_diffusion.experiments.same_latent_forcing import decode_response, plot_maps

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, default=Path("runs/mdv1-e3smv2-r1i1p1f1"))
parser.add_argument("--data", type=Path, default=Path("examples/e3sm-local.yaml"))
parser.add_argument("--output", type=Path, default=Path("analysis/same_latent_forcing"))
parser.add_argument(
    "--autoencoder-checkpoint",
    type=Path,
    default=None,
    help="Optional stage autoencoder/best.pt or latest.pt to use before AE stage completion",
)
parser.add_argument("--early", type=int, nargs=2, default=(1850, 1880))
parser.add_argument("--late", type=int, nargs=2, default=(2070, 2100))
parser.add_argument("--device", default="cpu")
parser.add_argument("--threads", type=int, default=4)
args = parser.parse_args()
torch.set_num_threads(args.threads)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
output = args.output
output.mkdir(parents=True, exist_ok=True)
source = yaml.safe_load(args.data.read_text())
atmos, ocean, _ = open_canonical_state(
    source["state_zarr"], expected_attrs=source.get("expected_attrs")
)
co2 = open_co2(source["co2_netcdf"], expected_sha256=source.get("co2_sha256"))
model = MonthlyDiffusionV1.load(args.run / "model.pt", device=args.device)
checkpoint_epoch = None
if args.autoencoder_checkpoint is not None:
    payload = torch.load(
        args.autoencoder_checkpoint, map_location="cpu", weights_only=False
    )
    model.autoencoder.load_state_dict(payload["model"])
    checkpoint_epoch = int(payload.get("epoch", -1))
    logging.info(
        "Loaded interim autoencoder checkpoint %s (epoch %s)",
        args.autoencoder_checkpoint,
        checkpoint_epoch,
    )
if not model.latent_normalizers:
    # The parent model checkpoint is written before diffusion begins, so an
    # interim AE has no fitted transition normalizers yet.  Identity units
    # are exact for this decoder-only paired comparison: the same raw z is
    # decoded twice, and no transition score is being evaluated.
    model.latent_normalizers = {}
    for component in ("atmosphere", "ocean"):
        normalizer = LatentNormalizer()
        normalizer.mean = torch.zeros(model.config.latent_channels)
        normalizer.std = torch.ones(model.config.latent_channels)
        model.latent_normalizers[component] = normalizer
    logging.info(
        "No fitted latent statistics; using identity units for interim same-latent audit"
    )
early_ds, late_ds, response, early_co2, late_co2 = decode_response(
    model,
    atmos.sel(time=slice("1966-01", "2006-12")),
    ocean.sel(time=slice("1966-01", "2006-12")),
    co2,
    tuple(args.early),
    tuple(args.late),
)
archive = xr.merge(
    [
        early_ds.rename({v: f"decoded_early_{v}" for v in early_ds.data_vars}),
        late_ds.rename({v: f"decoded_late_{v}" for v in late_ds.data_vars}),
        response.rename({v: f"response_{v}" for v in response.data_vars}),
    ]
)
archive.attrs.update(response.attrs)
archive["co2_early"] = early_co2.assign_attrs(units="ppm")
archive["co2_late"] = late_co2.assign_attrs(units="ppm")
archive.to_netcdf(output / "same_latent_response.nc")
rows = []
for name, field in response.data_vars.items():
    mean_map = field.mean("time", skipna=True)
    rows.append(
        {
            "variable": name,
            "area_weighted_mean_response": float(weighted_mean(mean_map).values),
            "spatial_rms_response": float(
                np.sqrt((mean_map**2).mean(skipna=True).values)
            ),
        }
    )
import pandas as pd

pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
plot_maps(response, output, tuple(args.early), tuple(args.late))
metadata = {
    "run": str(args.run.resolve()),
    "checkpoint": str((args.run / "model.pt").resolve()),
    "autoencoder_checkpoint": None
    if args.autoencoder_checkpoint is None
    else str(args.autoencoder_checkpoint.resolve()),
    "autoencoder_checkpoint_epoch": checkpoint_epoch,
    "device": args.device,
    "early": args.early,
    "late": args.late,
    "n_samples": int(response.sizes["time"]),
    "co2_path": str(Path(source["co2_netcdf"]).resolve()),
    "definition": response.attrs["definition"],
    "source_member": atmos.attrs.get("member_id"),
    "state_path": str(Path(source["state_zarr"]).resolve()),
    "source_expected_attrs": source.get("expected_attrs"),
    "co2_early_mean_ppm": float(early_co2.mean()),
    "co2_late_mean_ppm": float(late_co2.mean()),
    "sample_dates": [str(t) for t in response.time.values],
    "annual_average": "equal weights across 36 paired monthly states (12 months in each of 3 years)",
}
identity_path = args.autoencoder_checkpoint or args.run / "model.pt"
with identity_path.open("rb") as stream:
    metadata["weights_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
(output / "metadata.json").write_text(json.dumps(metadata, indent=2))
logging.info("Wrote %s", output / "same_latent_response.nc")
