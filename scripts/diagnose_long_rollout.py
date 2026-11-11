#!/usr/bin/env python3
"""Run the frozen package's bounded-memory long-rollout evaluation."""
import argparse
import fcntl
import json
import logging
import os
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/".matplotlib"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("--stage", choices=("all", "physical", "plots", "decoder"), default="all")
    p.add_argument("--device", default=None, help="Device for paired decoder probe only")
    p.add_argument("--wait", action="store_true", help="Wait for a running diagnostic stage before resuming")
    p.add_argument("--forecast", type=Path, help="Explicit alternate forecast (e.g. constant-CO2 control)")
    p.add_argument("--output", type=Path, help="Separate diagnostic output for an alternate forecast")
    p.add_argument("--scenario-label", default=None)
    p.add_argument("--skip-decoder", action="store_true", help="Skip repeated checkpoint-only gain probes for a control run")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    # Bound BLAS/OpenMP workers before importing NumPy/Torch.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "4"
    from monthly_diffusion.analysis.long_rollout import run_physical_diagnostics
    from monthly_diffusion.analysis.long_rollout_plots import plot_long_rollout
    from monthly_diffusion.analysis.decoder_gain import decoder_diagnostics
    import torch
    torch.set_num_threads(4)
    config = yaml.safe_load(args.config.read_text())
    if args.forecast is not None:
        if args.output is None:
            p.error("--forecast requires --output so the original diagnostics are preserved")
        config["forecast"] = str(args.forecast)
    if args.output is not None:
        config["output"] = str(args.output)
    if args.scenario_label is not None:
        config["scenario_label"] = args.scenario_label
    for key in ("forecast", "checkpoint", "training_source", "co2_netcdf", "output"):
        config[key] = str((ROOT/config[key]).resolve())
    config["references"] = [str((ROOT/path).resolve()) for path in config["references"]]
    Path(config["output"]).mkdir(parents=True, exist_ok=True)
    lock = open(Path(config["output"])/"diagnostics.lock", "a")
    if args.wait:
        logging.info("Waiting for exclusive access to %s", config["output"])
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | (0 if args.wait else fcntl.LOCK_NB))
    except BlockingIOError:
        raise RuntimeError("Another diagnostic process is already writing this output directory") from None
    marker = Path(config["output"])/"configuration.json"
    if marker.exists() and json.loads(marker.read_text())["config"] != config:
        raise ValueError("Diagnostic configuration changed; choose a new output directory")
    if args.stage in ("all", "physical"):
        run_physical_diagnostics(config, config["output"])
    if args.stage in ("all", "plots"):
        plot_long_rollout(config["output"], config)
    if args.stage in ("all", "decoder") and not args.skip_decoder:
        decoder_diagnostics(config, config["output"], device=args.device)
    logging.info("Completed %s diagnostics: %s", args.stage, config["output"])


if __name__ == "__main__":
    main()
