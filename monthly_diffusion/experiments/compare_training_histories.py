"""Read-only comparison of two staged MDv1 runs, including loss attribution.

Histories contain online training losses and EMA validation losses: their gap
is descriptive, not an exactly matched evaluation of one set of weights.
Each run also uses its own training-fitted physical and latent normalization.
Optional validation replay uses the selected AE's immutable latent cache and
the original per-month SST message mask. It decomposes, but never changes,
the reconstruction objective into cells inside/outside that support.
"""

from pathlib import Path
import argparse
import hashlib
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[2]
from monthly_diffusion import MonthlyDiffusionV1
from monthly_diffusion.data.sources import open_canonical_state, open_co2
from monthly_diffusion.data.dataloader import co2_values, month_features
from monthly_diffusion.modules.autoencoder import reconstruction_mse

RUNS = {
    "e3sm": ROOT / "runs/mdv1-e3smv2-r1i1p1f1",
    "era": ROOT / "runs/mdv1-era5-oras5",
}
LABELS = {"e3sm": "E3SMv2", "era": "ERA5/ORAS5"}
COLORS = {"e3sm": "#333333", "era": "#0072B2"}


def save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=190, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def validation_support(output, histories, runs):
    """Replay all 96 validation months, four at a time, in objective units.

    Inside/outside contributions use the ORIGINAL full valid-area denominator;
    their sum equals the logged per-field loss. Within-region MSE is also saved,
    but is not additive. Physical RMSE is exact only for identity transforms.
    """
    rows = []
    for key, run in runs.items():
        model = MonthlyDiffusionV1.load(run / "model.pt", device="cpu")
        cache = torch.load(run / "latents.pt", map_location="cpu", weights_only=False)
        assert cache["ae_digest"] == model.weight_digest(autoencoder_only=True)
        sources = yaml.safe_load((run / "data_sources.yaml").read_text())
        a, o, _ = open_canonical_state(
            sources["state_zarr"], expected_attrs=sources.get("expected_attrs")
        )
        dates = np.asarray(cache["validation_time"])
        fields = {"atmosphere": a.sel(time=dates), "ocean": o.sel(time=dates)}
        co2 = open_co2(sources["co2_netcdf"], expected_sha256=sources.get("co2_sha256"))
        static = model.preprocessor.static_tensor(model.statics)[None]
        lat = torch.tensor(model.statics.lat.values, dtype=torch.float32)
        weight = np.cos(np.deg2rad(model.statics.lat.values))[None, None, :, None]
        months = month_features(fields["ocean"].time)
        forcing = torch.tensor(
            (
                (co2_values(co2, fields["ocean"].time) - model.preprocessor.co2_mean)
                / model.preprocessor.co2_std
            ).values,
            dtype=torch.float32,
        )[:, None]
        best = histories[key, "autoencoder"].loc[
            histories[key, "autoencoder"].val_loss.idxmin()
        ]
        maps = {}
        for component in ("atmosphere", "ocean"):
            specs = [v for v in model.config.variables if v.component == component]
            n = len(specs)
            sums = {
                s: np.zeros(n)
                for s in (
                    "total",
                    "inside",
                    "outside",
                    "outside_area",
                    "inside_mse",
                    "outside_mse",
                )
            }
            error_sum = np.zeros((n, len(lat), len(model.statics.lon)))
            counts = np.zeros_like(error_sum)
            outside_counts = np.zeros_like(error_sum)
            for start in range(0, len(dates), 4):
                sl = slice(start, start + 4)
                normalized = model.preprocessor.encode(
                    fields[component].isel(time=sl), component
                ).compute()
                target = torch.from_numpy(
                    np.stack(
                        [
                            normalized[v.name].transpose("time", "lat", "lon").values
                            for v in specs
                        ],
                        1,
                    )
                )
                mask = (
                    torch.isfinite(target[:, 0])
                    if component == "ocean" and model.config.ocean_mask_aware
                    else None
                )
                z = model.latent_normalizers[component].denormalize(
                    cache["validation"][component][sl]
                )
                prediction = getattr(model.autoencoder, component).decode(
                    z,
                    static.expand(len(z), -1, -1, -1),
                    months[sl],
                    forcing[sl],
                    node_mask=mask,
                )
                valid = np.isfinite(target.numpy())
                squared = np.where(
                    valid,
                    (prediction.numpy() - target.numpy()).astype("float64") ** 2,
                    0.0,
                )
                outside = (
                    np.zeros_like(valid)
                    if mask is None
                    else valid & ~mask.numpy()[:, None]
                )
                denominator = (weight * valid).sum((-2, -1))
                outside_area = (weight * outside).sum((-2, -1))
                outside_num = (weight * squared * outside).sum((-2, -1))
                total_num = (weight * squared).sum((-2, -1))
                total = total_num / denominator
                np.testing.assert_allclose(
                    total.mean(0),
                    reconstruction_mse(prediction, target, lat).numpy(),
                    rtol=3e-6,
                    atol=1e-7,
                )
                sums["total"] += total.sum(0)
                sums["outside"] += (outside_num / denominator).sum(0)
                sums["inside"] += ((total_num - outside_num) / denominator).sum(0)
                sums["outside_area"] += (outside_area / denominator).sum(0)
                sums["inside_mse"] += (
                    (total_num - outside_num) / (denominator - outside_area)
                ).sum(0)
                sums["outside_mse"] += np.divide(
                    outside_num,
                    outside_area,
                    out=np.zeros_like(outside_num),
                    where=outside_area > 0,
                ).sum(0)
                error_sum += squared.sum(0)
                counts += valid.sum(0)
                outside_counts += outside.sum(0)
                if start % 32 == 0:
                    print(
                        key,
                        component,
                        "validation months",
                        start,
                        "/",
                        len(dates),
                        flush=True,
                    )
            for i, spec in enumerate(specs):
                result = {k: float(v[i] / len(dates)) for k, v in sums.items()}
                logged = float(best[f"val_{component}_field{i}"])
                # The archive was generated on MPS, replay is CPU. Permit tiny
                # device differences, but reject a wrong checkpoint or mask.
                np.testing.assert_allclose(
                    result["total"], logged, rtol=0.003, atol=2e-5
                )
                np.testing.assert_allclose(
                    result["inside"] + result["outside"], result["total"], rtol=1e-10
                )
                rows.append(
                    dict(
                        dataset=key,
                        component=component,
                        variable=spec.name,
                        logged=logged,
                        weight=spec.weight,
                        outside_fraction_of_error=result["outside"] / result["total"],
                        identity_physical_rmse=(
                            np.sqrt(result["total"])
                            * model.preprocessor.stats[spec.name]["std"]
                            if spec.transform == "identity"
                            else np.nan
                        ),
                        **result,
                    )
                )
                maps[spec.name + "_mse"] = (
                    ("lat", "lon"),
                    np.divide(
                        error_sum[i],
                        counts[i],
                        out=np.full_like(counts[i], np.nan),
                        where=counts[i] > 0,
                    ),
                )
                maps[spec.name + "_outside_support_fraction"] = (
                    ("lat", "lon"),
                    np.divide(
                        outside_counts[i],
                        counts[i],
                        out=np.full_like(counts[i], np.nan),
                        where=counts[i] > 0,
                    ),
                )
        xr.Dataset(
            maps,
            coords=dict(lat=model.statics.lat, lon=model.statics.lon),
            attrs=dict(
                period="1958-1965",
                definition="time-mean squared error in model training coordinates",
                checkpoint_sha256=model.weight_digest(),
            ),
        ).to_netcdf(output / f"{key}_validation_errors.nc")
        pd.DataFrame(rows).to_csv(output / "validation_support.csv", index=False)
        del model, cache
