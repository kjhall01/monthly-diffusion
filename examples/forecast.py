"""Stream a chosen trained model; source/forcing paths stay in an editable YAML."""

import argparse
import logging
from pathlib import Path
import torch
import yaml
import xarray as xr
from monthly_diffusion.inference.metadata import forecast_metadata
from monthly_diffusion import MonthlyDiffusionV1
from monthly_diffusion.data.dataloader import co2_values
from monthly_diffusion.inference.forecast import Forecast
from monthly_diffusion.inference.archive import (
    prepare_archive,
    write_month,
)
from monthly_diffusion.inference.spinup import spin_up
from monthly_diffusion.data.sources import (
    open_canonical_state,
    open_co2,
    hold_first_co2_back_to,
)


p = argparse.ArgumentParser()
p.add_argument("checkpoint", type=Path)
p.add_argument("data_yaml", type=Path)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--months", type=int, default=120)
p.add_argument("--members", type=int, default=5)
p.add_argument(
    "--initialization",
    default=None,
    help="Current month YYYY-MM; prior month is also required",
)
p.add_argument(
    "--gaussian-start",
    default=None,
    help="First generated month YYYY-MM; initialize both normalized latent histories with independent N(0,1)",
)
p.add_argument(
    "--encoded-noise-start",
    default=None,
    help="First generated month YYYY-MM; encode independent N(0,1) preprocessed fields with train-fitted masks and history-month conditioning",
)
p.add_argument(
    "--stochastic-start",
    default=None,
    help="First generated month YYYY-MM; training-local moment-matched field noise plus one predict/decode/re-encode cycle",
)
p.add_argument(
    "--spinup-months",
    type=int,
    default=0,
    help="Latent-only months to discard BEFORE the --months saved months",
)
p.add_argument(
    "--save-spinup-latents",
    action="store_true",
    help="Retain every spin-up latent in resumable annual blocks; do not decode spin-up",
)
p.add_argument(
    "--hold-first-co2-back-to",
    default=None,
    help="Explicit constant prefix using the first prescribed CO2 value",
)
p.add_argument("--constant-co2", type=float, default=None)
p.add_argument(
    "--constant-initial-co2",
    action="store_true",
    help="Keep actual CO2 for both history months; hold future CO2 at the current month's value",
)
p.add_argument("--resume", action="store_true")
p.add_argument(
    "--extend",
    action="store_true",
    help="With --resume, allow a longer horizon only if the checkpoint, initialization and entire previous forcing prefix match",
)
p.add_argument("--device", default=None)
p.add_argument("--threads", type=int, default=4)
p.add_argument(
    "--require-trained", action="store_true", help="Refuse an incomplete staged model"
)
p.add_argument("--seed", type=int, default=20260716)
args = p.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
torch.set_num_threads(args.threads)
if args.device == "mps" and not torch.backends.mps.is_available():
    raise RuntimeError("MPS requested but unavailable; not falling back to CPU")
if args.constant_initial_co2 and args.constant_co2 is not None:
    p.error("Choose --constant-initial-co2 or --constant-co2, not both")
if (
    sum(
        v is not None
        for v in (
            args.gaussian_start,
            args.encoded_noise_start,
            args.stochastic_start,
            args.initialization,
        )
    )
    > 1
):
    p.error(
        "Choose one of --gaussian-start, --encoded-noise-start, --stochastic-start or physical --initialization"
    )
if args.save_spinup_latents and args.spinup_months < 1:
    p.error("--save-spinup-latents requires positive --spinup-months")
if args.months < 1 or args.spinup_months < 0:
    p.error("months must be positive and spinup_months nonnegative")
if args.extend and not args.resume:
    p.error("--extend requires --resume")
noise_start = args.stochastic_start or args.encoded_noise_start or args.gaussian_start
source = yaml.safe_load(args.data_yaml.read_text())
co2 = (
    args.constant_co2
    if args.constant_co2 is not None
    else open_co2(source["co2_netcdf"], expected_sha256=source.get("co2_sha256"))
)
if args.hold_first_co2_back_to:
    if args.constant_co2 is not None:
        p.error("A scalar constant CO2 does not require a prefix")
    co2 = hold_first_co2_back_to(co2, args.hold_first_co2_back_to)
    logging.info(
        "Explicit spin-up prefix: %s through %s at %.4f ppm",
        args.hold_first_co2_back_to,
        co2.attrs["spinup_prefix_end"],
        co2.attrs["spinup_prefix_ppm"],
    )
if noise_start:
    a, o = None, None
else:
    a, o, _ = open_canonical_state(
        source["state_zarr"], expected_attrs=source.get("expected_attrs")
    )
    if args.initialization:
        a, o = (
            a.sel(time=slice(None, args.initialization)).isel(time=slice(-2, None)),
            o.sel(time=slice(None, args.initialization)).isel(time=slice(-2, None)),
        )
    else:
        a, o = a.isel(time=slice(0, 2)), o.isel(time=slice(0, 2))
    if args.initialization and (
        not a.sizes["time"] or str(a.time.values[-1])[:7] != args.initialization
    ):
        raise ValueError("Requested initialization month is unavailable")
if args.constant_initial_co2:
    if noise_start:
        ppm = float(co2.sel(time=noise_start).values[0])
        co2 = co2.copy(data=co2.values * 0 + ppm)
    else:
        ppm = float(co2_values(co2, a.time).isel(time=-1))
        current = int(a.time.dt.year[-1]) * 12 + int(a.time.dt.month[-1])
        co2 = co2.where(co2.time.dt.year * 12 + co2.time.dt.month <= current, ppm)
    logging.info("Control: future CO2 fixed at %.4f ppm", ppm)
model = MonthlyDiffusionV1.load(args.checkpoint, device=args.device)
if args.require_trained and not all(
    k in model.stages for k in ("autoencoder", "diffusion")
):
    raise RuntimeError("Finish both training stages before starting the long rollout")
logging.info(
    "Forecast device %s | selected stages %s | %d members × %d months",
    model.device,
    model.stages,
    args.members,
    args.months,
)
# Encode two observed history months, or draw the requested noise histories.
forecast = Forecast(
    model,
    a,
    o,
    co2=co2,
    members=args.members,
    seed=args.seed,
    gaussian_start=args.gaussian_start,
    encoded_noise_start=args.encoded_noise_start,
    stochastic_start=args.stochastic_start,
)

# Check forcing coverage and archive identity before drawing any forecast months.
metadata = forecast_metadata(
    forecast,
    months=args.months,
    spinup_months=args.spinup_months,
    gaussian_start=args.gaussian_start,
    encoded_noise_start=args.encoded_noise_start,
    stochastic_start=args.stochastic_start,
    save_spinup_latents=args.save_spinup_latents,
)
prepare_archive(
    forecast,
    args.output,
    metadata,
    resume=args.resume,
    extend=args.extend,
    spinup_months=args.spinup_months,
    save_spinup_latents=args.save_spinup_latents,
    save_initial_state=noise_start is not None or args.spinup_months > 0,
)

# Spin-up keeps history and noise continuous, but discards the physical output.
spin_up(
    forecast,
    args.spinup_months,
    path=args.output,
    save_latents=args.save_spinup_latents,
)

# Predict one month, decode to physical fields, then save fields + restart state.
# On resume, forecast.step already points to the last completed month.
for step in range(forecast.step, args.spinup_months + args.months):
    atmosphere_latent, ocean_latent = next(forecast)
    fields = xr.merge(
        model.decode(atmosphere_latent, ocean_latent, co2=co2), compat="no_conflicts"
    )
    fields = fields.assign_coords(co2=co2_values(co2, fields.time))
    fields.attrs.update(
        checkpoint_sha256=metadata["checkpoint_sha256"],
        seed=args.seed,
        initialization_time=metadata["initialization_time"],
        history_order="current, previous",
    )
    write_month(fields, forecast, args.output, spinup_months=args.spinup_months)
    saved_months = forecast.step - args.spinup_months
    if saved_months % 12 == 0 or saved_months == args.months:
        logging.info(
            "Saved forecast month %d/%d (%s)",
            saved_months,
            args.months,
            str(fields.time.values[0])[:10],
        )
