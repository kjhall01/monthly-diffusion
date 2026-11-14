"""Paired ENSO consistency after a saved decode/re-encode initialization pilot.

No forecasts are regenerated. Reuse immutable training-fitted latent probes.
Decode saved predicted latents to compute physical Nino3.4 and a gridded SOI
proxy (Tahiti-box minus Darwin-box MSLP). Decode encoded TRAINING latents at
the same fixed CO2 to supply decoder-matched monthly climatologies and a fair
autoencoded physical relationship. Also retain the actual source relationship.
SOI proxy is monthly pressure-difference anomaly / training anomaly SD, not
station SOI. Reverse its sign for comparisons: El Nino positive on both axes.

No rollout-specific climatology, linear detrending, ensemble-mean subtraction,
or fitted rollout recalibration. Correlation centers only for its definition.
Report offsets, spread and departures from the TRAINING regression as well as
correlation. Thirty months/one member cannot establish coupling or causality.
"""

# Make the repository workflow helpers available when running this file directly.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from monthly_diffusion.checkpoints import load_checkpoint

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from workflows.legacy_model import MonthlyDiffusionV1
from workflows.analysis.indices import regional_mean
from workflows.data.sources import open_canonical_state


from workflows.experiments.diagnose_cycled_coupling import COMPONENTS, fixed_anomalies, pair_metrics, lag_curve, physical_indices, decode_indices, plot_comparison

p = argparse.ArgumentParser(description=__doc__)
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
p.add_argument("--device", default="mps")
args = p.parse_args()
torch.set_num_threads(4)
output = args.pilot / "coupling"
output.mkdir(exist_ok=True)
config = json.loads((args.pilot / "configuration.json").read_text())
probe_config = json.loads((args.readouts / "configuration.json").read_text())
if config["weight_digest"] != probe_config["checkpoint_sha256"]:
    raise ValueError("Pilot/probe checkpoint mismatch")
run = Path(config["run"])
cache = load_checkpoint(run / "latents.pt", map_location="cpu", weights_only=False)
saved = load_checkpoint(
    args.pilot / "latent_states.pt", map_location="cpu", weights_only=False
)
train_dates = np.asarray(cache["training_time"])
training = {c: cache["training"][c].numpy().astype(float) for c in COMPONENTS}
selected = [r for r in saved["records"] if r["branch"] == "cycled" and r["step"] > 0]
dates = np.asarray([r["time"] for r in selected])
raw = {}
for name, branch, stage in (
    ("cycled_predicted", "cycled", "predicted"),
    ("cycled_fed_back", "cycled", "fed_back"),
    ("latent_only", "latent_only", "predicted"),
):
    rows = [r for r in saved["records"] if r["branch"] == branch and r["step"] > 0]
    np.testing.assert_array_equal([r["time"] for r in rows], dates)
    raw[name] = {
        c: np.stack([r[stage][i][0].numpy() for r in rows])
        for i, c in enumerate(COMPONENTS)
    }
probes = {c: np.load(args.readouts / f"{c}_readout.npz")["coef"] for c in COMPONENTS}
latent, ref_readouts = {}, {}
for name, values in raw.items():
    projected = {}
    for c in COMPONENTS:
        z, ref = fixed_anomalies(values[c], dates, training[c], train_dates)
        projected[c] = z.reshape(len(z), -1) @ probes[c]
        ref_readouts[c] = ref.reshape(len(ref), -1) @ probes[c]
    latent[name] = np.column_stack((projected["ocean"], projected["atmosphere"]))
latent_ref = np.column_stack((ref_readouts["ocean"], ref_readouts["atmosphere"]))
validation = []
for c in ("ocean", "atmosphere"):
    z, _ = fixed_anomalies(
        cache["validation"][c].numpy(),
        cache["validation_time"],
        training[c],
        train_dates,
    )
    validation.append(z.reshape(len(z), -1) @ probes[c])
validation = np.column_stack(validation)
xr.Dataset(
    dict(
        training=(("training_time", "component"), latent_ref),
        validation=(("validation_time", "component"), validation),
    ),
    coords=dict(
        training_time=train_dates,
        validation_time=cache["validation_time"],
        component=["ocean", "atmosphere"],
    ),
).to_netcdf(output / "latent_reference_indices.nc")
xr.Dataset(
    {name: (("time", "component"), pair) for name, pair in latent.items()},
    coords=dict(time=dates, component=["ocean", "atmosphere"]),
).to_netcdf(output / "latent_indices.nc")
path = output / "decoded_indices.nc"
refpath = output / "encoded_reference_indices.nc"
if not path.exists() or not refpath.exists():
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no silent CPU fallback for decoding")
    model = MonthlyDiffusionV1.load(run / "model.pt", device=args.device)
    if (
        model.weight_digest() != config["weight_digest"]
        or model.weight_digest(autoencoder_only=True) != cache["ae_digest"]
    ):
        raise ValueError("Checkpoint/cache changed after pilot")
    if not refpath.exists():
        print(
            "Decoding training latents at pilot CO2 for a decoder-matched reference",
            flush=True,
        )
        decode_indices(model, training, train_dates, config["co2_ppm"]).to_netcdf(
            refpath
        )
    if not path.exists():
        ds = xr.concat(
            [
                decode_indices(model, raw[b], dates, config["co2_ppm"]).expand_dims(
                    branch=[b]
                )
                for b in ("cycled_predicted", "latent_only")
            ],
            dim="branch",
        )
        ds.to_netcdf(path)
    del model
d, refds = xr.load_dataset(path), xr.load_dataset(refpath)
# Replay the already archived map snapshots: proves the index decodes use
# the saved PREDICTED latents, not the subsequent re-encoded states.
snapshots = xr.open_dataset(args.pilot / "physical_snapshots.nc")
shared = np.intersect1d(snapshots.time.values, dates)
for name, branch in (("cycled_predicted", "cycled"), ("latent_only", "latent_only")):
    nino = regional_mean(
        snapshots.sst.sel(branch=branch, time=shared),
        lat_bounds=(-5.0, 5.0),
        lon_bounds=(190.0, 240.0),
    )
    np.testing.assert_allclose(
        nino.values, d.nino34.sel(branch=name, time=shared).values, rtol=0, atol=1.0e-4
    )
physical = {}
rp = np.column_stack((refds.nino34.values, -refds.pressure_difference.values))
_, physical_ref = fixed_anomalies(rp, train_dates, rp, train_dates)
for name in d.branch.values:
    b = d.sel(branch=name)
    pair = np.column_stack((b.nino34.values, -b.pressure_difference.values))
    physical[str(name)], _ = fixed_anomalies(pair, dates, rp, train_dates)
# Source fields provide an independent physical relationship, not another
# learned readout. Reduce to regional scalars BEFORE materializing data.
sourcepath = output / "actual_reference_indices.nc"
sources = yaml.safe_load((run / "data_sources.yaml").read_text())
if not sourcepath.exists():
    a, o, _ = open_canonical_state(
        sources["state_zarr"], expected_attrs=sources.get("expected_attrs")
    )
    physical_indices(
        a.sel(time=slice("1966", "2006")), o.sel(time=slice("1966", "2006"))
    ).to_netcdf(sourcepath)
actual = xr.load_dataset(sourcepath)
ar = np.column_stack((actual.nino34.values, -actual.pressure_difference.values))
_, actual_pair = fixed_anomalies(ar, actual.time.values, ar, actual.time.values)
metrics, window_rows, lag_rows, sensitivities = [], [], [], []
for kind, reference, candidates in (
    ("latent", latent_ref, latent),
    ("physical", physical_ref, physical),
):
    starts = [
        i
        for i in range(len(train_dates) - len(dates) + 1)
        if pd.Timestamp(train_dates[i]).month == pd.Timestamp(dates[0]).month
    ]
    windows = [reference[i : i + len(dates)] for i in starts]
    for i, window in zip(starts, windows):
        window_rows.append(
            dict(
                kind=kind, start=str(train_dates[i]), **pair_metrics(window, reference)
            )
        )
    for name, pair in {"encoded_reference": reference, **candidates}.items():
        metrics.append(dict(kind=kind, series=name, **pair_metrics(pair, reference)))
        lag_rows.extend(
            dict(kind=kind, series=name, lag_months=lag, correlation=value)
            for lag, value in zip(range(-6, 7), lag_curve(pair))
        )
    for skip in (0, 6, 12):
        for name, pair in candidates.items():
            sensitivities.append(
                dict(
                    kind=kind,
                    series=name,
                    first_step=skip + 1,
                    last_step=len(dates),
                    **pair_metrics(pair[skip:], reference),
                )
            )
    plot_comparison(reference, candidates, train_dates, dates, kind, output, windows)
metrics.append(
    dict(
        kind="physical",
        series="actual_source_reference",
        **pair_metrics(actual_pair, physical_ref),
    )
)
metrics.append(
    dict(
        kind="latent",
        series="encoded_validation_reference",
        **pair_metrics(validation, latent_ref),
    )
)
pd.DataFrame(metrics).to_csv(output / "coupling_metrics.csv", index=False)
pd.DataFrame(window_rows).to_csv(output / "matched_reference_windows.csv", index=False)
pd.DataFrame(lag_rows).to_csv(output / "lag_correlations.csv", index=False)
pd.DataFrame(sensitivities).to_csv(
    output / "initial_adjustment_sensitivity.csv", index=False
)
record = dict(
    config,
    readouts=str(args.readouts.resolve()),
    physical_reference="Encoded training states decoded at same fixed CO2 as pilot",
    anomalies="Fixed training monthly means; no trend/window climatology/ensemble mean removed",
    pressure_proxy="Area-weighted Tahiti [-20,-10;205,216] minus Darwin [-18,-8;126,136], Pa; reversed for plots",
    physical_plot_scaling="Each index divided by its own encoded-reference anomaly SD; SOI is a gridded proxy, not station SOI",
    limitations="One 30-month trajectory; training-reference readouts fitted to same SST target; correlation is not feedback/causality",
    replay_check="Regional SST index matches all saved physical snapshots within 1e-4 degrees C",
    initial_adjustment_sensitivity="Also report months 7–30 and 13–30, retaining the SAME fixed training climatologies and regressions",
    reference_window_band="5th–95th percentiles of overlapping same-calendar-start 30-month training windows; NOT confidence interval",
)
(output / "configuration.json").write_text(json.dumps(record, indent=2))
print(pd.DataFrame(metrics).to_string(index=False), flush=True)
print(f"Saved {output}", flush=True)
