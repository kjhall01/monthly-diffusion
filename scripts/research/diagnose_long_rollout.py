#!/usr/bin/env python3
"""Run the frozen package's bounded-memory long-rollout evaluation."""

# Make the repository workflow helpers available when running this file directly.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import fcntl
import json
import logging
import os
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("config", type=Path)
p.add_argument(
    "--stage", choices=("all", "physical", "plots", "decoder"), default="all"
)
p.add_argument("--device", default=None, help="Device for paired decoder probe only")
p.add_argument(
    "--wait",
    action="store_true",
    help="Wait for a running diagnostic stage before resuming",
)
p.add_argument(
    "--forecast",
    type=Path,
    help="Explicit alternate forecast (e.g. constant-CO2 control)",
)
p.add_argument(
    "--output", type=Path, help="Separate diagnostic output for an alternate forecast"
)
p.add_argument("--scenario-label", default=None)
p.add_argument(
    "--skip-decoder",
    action="store_true",
    help="Skip repeated checkpoint-only gain probes for a control run",
)
args = p.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
# Bound BLAS/OpenMP workers before importing NumPy/Torch.
for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ[name] = "4"
from workflows.analysis.archives import ClimateArchive
from workflows.analysis.long_rollout import fit_pdo, extract_indices, variable_products
import numpy as np
import xarray as xr
from workflows.analysis.long_rollout_plots import plot_long_rollout
from workflows.analysis.decoder_gain import decoder_diagnostics
import torch

torch.set_num_threads(4)
config = yaml.safe_load(args.config.read_text())
if args.forecast is not None:
    if args.output is None:
        p.error(
            "--forecast requires --output so the original diagnostics are preserved"
        )
    config["forecast"] = str(args.forecast)
if args.output is not None:
    config["output"] = str(args.output)
if args.scenario_label is not None:
    config["scenario_label"] = args.scenario_label
for key in ("forecast", "checkpoint", "training_source", "co2_netcdf", "output"):
    config[key] = str((ROOT / config[key]).resolve())
config["references"] = [str((ROOT / path).resolve()) for path in config["references"]]
Path(config["output"]).mkdir(parents=True, exist_ok=True)
lock = open(Path(config["output"]) / "diagnostics.lock", "a")
if args.wait:
    logging.info("Waiting for exclusive access to %s", config["output"])
try:
    fcntl.flock(lock, fcntl.LOCK_EX | (0 if args.wait else fcntl.LOCK_NB))
except BlockingIOError:
    raise RuntimeError(
        "Another diagnostic process is already writing this output directory"
    ) from None
marker = Path(config["output"]) / "configuration.json"
if marker.exists() and json.loads(marker.read_text())["config"] != config:
    raise ValueError("Diagnostic configuration changed; choose a new output directory")
if args.stage in ("all", "physical"):
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    forecast = ClimateArchive([config["forecast"]], forecast=True)
    # The model's fixed TRAINING SST mask governs both sides of comparison.
    # Reference-specific missing cells inside that domain remain missing.
    reference = ClimateArchive(
        config["references"],
        ocean_support=forecast.ocean_support,
        ocean_variables=forecast.ocean_variables,
    )
    reanalysis = config.get("reference_kind", "held_out_ensemble") == "reanalysis"
    if reanalysis and len(reference.members) != 1:
        raise ValueError("Reanalysis comparisons require one observed trajectory")
    if not reanalysis and (
        config.get("training_member", "r1i1p1f1") in reference.members
        or len(reference.members) < 2
    ):
        raise ValueError("Use at least two HELD-OUT reference members")
    if not np.array_equal(forecast.lat, reference.lat) or not np.array_equal(
        forecast.lon, reference.lon
    ):
        raise ValueError("Reference and model must share the same grid")
    from workflows.data.dataloader import validate_units
    from monthly_diffusion.preprocessing import Variable

    validate_units(
        xr.Dataset(
            {
                v: xr.DataArray(attrs={"units": reference.units.get(v, "")})
                for v in config["variables"]
            }
        ),
        [Variable(v, "", forecast.units[v]) for v in config["variables"]],
    )
    periods = {k: config[k] for k in ("early", "training", "late")}
    selections = []
    for name, archive in (("reference", reference), ("model", forecast)):
        start, end = config.get(
            "reference_evaluation" if name == "reference" else "evaluation",
            config.get("evaluation", [1851, 2100]),
        )
        keep = np.flatnonzero(
            (archive.time.dt.year >= start) & (archive.time.dt.year <= end)
        )
        expected = np.arange(start * 12 + 1, (end + 1) * 12 + 1)
        if not np.array_equal(archive.keys[keep], expected):
            raise ValueError(
                "Evaluation requires all twelve months of each requested year"
            )
        required = (
            [periods["training"]]
            if reanalysis and name == "reference"
            else periods.values()
        )
        if any(lo < start or hi > end for lo, hi in required):
            raise ValueError("Climatology periods must lie inside evaluation coverage")
        selections.append(keep)
    # Run identity binds caches to the scientific choices, all references and
    # the archived checkpoint, without loading the model or resuming inference.
    identity = dict(
        config=config,
        forecast=forecast.record,
        reference_members=list(reference.members),
        version=2 if reanalysis else 1,
    )
    marker = output / "configuration.json"
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise ValueError(
            "Diagnostic configuration changed; choose a new output directory"
        )
    marker.write_text(json.dumps(identity, indent=2))
    block_months = config.get("block_months", 12)
    if block_months < 1 or block_months > 24:
        raise ValueError("block_months must be 1..24 to bound memory")
    anomaly_fits = dict(reference=None, model=None)
    if reanalysis:
        if (
            config.get("anomaly_method") != "training_monthly_linear_detrend"
            or config.get("index_period") != config["training"]
        ):
            raise ValueError(
                "Reanalysis diagnostics require explicit matched training-period anomaly comparisons"
            )
        from workflows.analysis.trajectory_anomalies import TrainingTrajectoryAnomalies

        for name, archive in (("reference", reference), ("model", forecast)):
            anomaly_fits[name] = TrainingTrajectoryAnomalies(
                archive, periods["training"], output / name / "anomalies", block_months
            )
    basis = fit_pdo(
        reference, periods["training"], block_months, output, anomaly_fits["reference"]
    )
    for name, archive, selection in zip(
        ("reference", "model"), (reference, forecast), selections
    ):
        root = output / name
        root.mkdir(exist_ok=True)
        index_selection = selection
        if "index_period" in config:
            lo, hi = config["index_period"]
            index_selection = selection[
                (archive.time.dt.year.values[selection] >= lo)
                & (archive.time.dt.year.values[selection] <= hi)
            ]
        indices = extract_indices(
            archive, index_selection, basis, root, block_months, anomaly_fits[name]
        )
        archive_periods = (
            {"training": periods["training"]}
            if reanalysis and name == "reference"
            else periods
        )
        for variable in config["variables"]:
            variable_products(
                archive,
                selection,
                indices,
                variable,
                archive_periods,
                root,
                block_months,
                anomaly_fits[name],
            )
if args.stage in ("all", "plots"):
    plot_long_rollout(config["output"], config)
if args.stage in ("all", "decoder") and not args.skip_decoder:
    decoder_diagnostics(config, config["output"], device=args.device)
logging.info("Completed %s diagnostics: %s", args.stage, config["output"])
