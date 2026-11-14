"""Generate a resumable physical trajectory from the selected production models."""

# Make the repository workflow helpers available when running this file directly.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workflows.run_files import check_script_arguments
import numpy as np
import torch
import xarray as xr
from monthly_diffusion.loading import load_models
from workflows.data.dataloader import co2_values
from workflows.data.run_data import load_reference_data, load_run_co2
from workflows.data.sources import hold_first_co2_back_to
from monthly_diffusion.rollout import Forecast
from workflows.inference.metadata import forecast_metadata
from workflows.inference.archive import prepare_archive, write_month
from workflows.inference.spinup import spin_up

check_script_arguments()

RUN_DIR = Path(__file__).resolve().parent / "runs/test_gridpoint_no_co2"
OUTPUT = RUN_DIR / "ar_rollout"
START = "1851-01"  # first generated month, including any discarded spin-up
MONTHS = 3012  # number of saved months; 1850--2100 with no spin-up
MEMBERS = 5
GUIDANCE_SCALE = 1.0  # 0: CO2-dropped branch; 1: ordinary conditional; >1: CFG
INITIALIZATION = "observed"  # stochastic, encoded_noise, gaussian, or observed
SPINUP_MONTHS = 0
SAVE_SPINUP_LATENTS = False
CONSTANT_CO2 = None  # e.g. 284.27 for a constant-forcing experiment
CONSTANT_INITIAL_CO2 = (
    False  # retain observed history forcing; hold the future constant
)
HOLD_FIRST_CO2_BACK_TO = "1850-11"  # prescribed prefix if the source starts in 1850
RESUME = False
EXTEND = False
SEED = 20260716
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
THREADS = 4

if INITIALIZATION not in ("stochastic", "encoded_noise", "gaussian", "observed"):
    raise ValueError("Unknown INITIALIZATION")
if MONTHS < 1 or SPINUP_MONTHS < 0 or (EXTEND and not RESUME):
    raise ValueError(
        "Require positive MONTHS, nonnegative spin-up, and RESUME for EXTEND"
    )
if SAVE_SPINUP_LATENTS and not SPINUP_MONTHS:
    raise ValueError("SAVE_SPINUP_LATENTS requires positive SPINUP_MONTHS")
if CONSTANT_INITIAL_CO2 and CONSTANT_CO2 is not None:
    raise ValueError("Choose only one constant-forcing option")
torch.set_num_threads(THREADS)
co2 = CONSTANT_CO2 if CONSTANT_CO2 is not None else load_run_co2(RUN_DIR)
if (
    CONSTANT_CO2 is None
    and HOLD_FIRST_CO2_BACK_TO
    and str(co2.time.values[0])[:7] > HOLD_FIRST_CO2_BACK_TO
):
    co2 = hold_first_co2_back_to(co2, HOLD_FIRST_CO2_BACK_TO)
    print(
        f"CO2 prefix: {HOLD_FIRST_CO2_BACK_TO} through {co2.attrs['spinup_prefix_end']} at {co2.attrs['spinup_prefix_ppm']} ppm"
    )

atmosphere, ocean = None, None
if INITIALIZATION == "observed":
    atmosphere, ocean, _ = load_reference_data(RUN_DIR)
    history_months = [
        str(np.datetime64(START, "M") - np.timedelta64(k, "M")) for k in (2, 1)
    ]
    atmosphere = atmosphere.sel(time=slice(*history_months))
    ocean = ocean.sel(time=slice(*history_months))
    if (
        atmosphere.sizes["time"] != 2
        or str(atmosphere.time.values[-1])[:7] != history_months[-1]
    ):
        raise ValueError(
            "Observed initialization requires the two months preceding START"
        )
if CONSTANT_INITIAL_CO2:
    if INITIALIZATION == "observed":
        ppm = float(co2_values(co2, atmosphere.time).isel(time=-1))
        current = int(atmosphere.time.dt.year[-1]) * 12 + int(
            atmosphere.time.dt.month[-1]
        )
        co2 = co2.where(co2.time.dt.year * 12 + co2.time.dt.month <= current, ppm)
    else:
        ppm = float(co2.sel(time=START).values[0])
        co2 = co2.copy(data=co2.values * 0 + ppm)

model = load_models(RUN_DIR, device=DEVICE)
forecast = Forecast(
    model,
    atmosphere,
    ocean,
    co2=co2,
    members=MEMBERS,
    seed=SEED,
    guidance_scale=GUIDANCE_SCALE,
    gaussian_start=START if INITIALIZATION == "gaussian" else None,
    encoded_noise_start=START if INITIALIZATION == "encoded_noise" else None,
    stochastic_start=START if INITIALIZATION == "stochastic" else None,
)
metadata = forecast_metadata(
    forecast,
    months=MONTHS,
    spinup_months=SPINUP_MONTHS,
    gaussian_start=START if INITIALIZATION == "gaussian" else None,
    encoded_noise_start=START if INITIALIZATION == "encoded_noise" else None,
    stochastic_start=START if INITIALIZATION == "stochastic" else None,
    save_spinup_latents=SAVE_SPINUP_LATENTS,
)
prepare_archive(
    forecast,
    OUTPUT,
    metadata,
    resume=RESUME,
    extend=EXTEND,
    spinup_months=SPINUP_MONTHS,
    save_spinup_latents=SAVE_SPINUP_LATENTS,
    save_initial_state=INITIALIZATION != "observed" or SPINUP_MONTHS > 0,
)

# Resuming restores history, persistent noise, and RNG before advancing spin-up.
spin_up(forecast, SPINUP_MONTHS, path=OUTPUT, save_latents=SAVE_SPINUP_LATENTS)

# Predict, decode, then atomically save each month and the matching restart state.
for step in range(forecast.step, SPINUP_MONTHS + MONTHS):
    atmosphere_latent, ocean_latent = next(forecast)
    fields = xr.merge(
        model.decode(atmosphere_latent, ocean_latent, co2=co2), compat="no_conflicts"
    )
    fields = fields.assign_coords(co2=co2_values(co2, fields.time))
    fields.attrs.update(
        checkpoint_sha256=metadata["checkpoint_sha256"],
        seed=SEED,
        initialization_time=metadata["initialization_time"],
        history_order="current, previous",
    )
    if model.transition.condition_on_co2:
        fields.attrs["guidance_scale"] = GUIDANCE_SCALE
    write_month(fields, forecast, OUTPUT, spinup_months=SPINUP_MONTHS)
    saved_months = forecast.step - SPINUP_MONTHS
    if saved_months % 12 == 0 or saved_months == MONTHS:
        print(f"Saved month {saved_months}/{MONTHS}: {str(fields.time.values[0])[:7]}")
