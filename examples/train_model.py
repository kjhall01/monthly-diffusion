"""Explicit E3SM/ERA5 example: same split, staged architecture, no hidden downloads.

After `pip install -e .`, run:
    python examples/train_model.py examples/e3sm.yaml --output runs/e3sm
This is intentionally a readable script, not a framework CLI.
"""
import argparse
import json
import logging
from pathlib import Path
import yaml
import torch
from monthly_diffusion import MonthlyDiffusionV1, ModelConfig
from monthly_diffusion.data.sources import open_canonical_state, open_co2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_yaml", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("model.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--load-in-memory", action="store_true",
                        help="Keep only train/validation physical fields in CPU RAM (about 3.6 GiB at 1 degree)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    torch.set_num_threads(args.threads)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable; refusing a silent CPU run")
    logging.info("Opening pinned source from %s", args.data_yaml)
    source = yaml.safe_load(args.data_yaml.read_text())
    atmosphere, ocean, statics = open_canonical_state(source["state_zarr"], expected_attrs=source.get("expected_attrs"))
    co2 = open_co2(source["co2_netcdf"], expected_sha256=source.get("co2_sha256"))
    args.output.mkdir(parents=True, exist_ok=True)
    # Persist the source record beside model/config/history, not in the package.
    (args.output/"data_sources.yaml").write_text(yaml.safe_dump(source, sort_keys=False))
    model = MonthlyDiffusionV1(statics=statics, config=args.config, device=args.device)
    train = slice("1966-01", "2006-12")
    validation = slice("1958-01", "1965-12")
    logging.info("Device %s | training 1966-2006 (%d months) | validation 1958-1965 (%d months)",
                 model.device, atmosphere.sel(time=train).sizes["time"],
                 atmosphere.sel(time=validation).sizes["time"])
    logging.info("Preparing geometry and training-only normalization; output %s", args.output)
    train_atmos, train_ocean = atmosphere.sel(time=train), ocean.sel(time=train)
    val_atmos, val_ocean = atmosphere.sel(time=validation), ocean.sel(time=validation)
    if args.load_in_memory:
        logging.info("Loading training and validation fields into CPU RAM; test/future states remain lazy")
        for dataset in (train_atmos, train_ocean, val_atmos, val_ocean):
            dataset.load()
    # Test 2007--2014 is deliberately not supplied to either trainer.
    model.train(train_atmos, train_ocean, co2=co2,
                validation=(val_atmos, val_ocean),
                directory=args.output, resume=args.resume)


if __name__ == "__main__":
    main()
