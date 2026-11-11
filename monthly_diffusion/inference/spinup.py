"""Advance discarded months, optionally retaining annual latent restart blocks."""

from pathlib import Path
import logging
import numpy as np
from ..checkpoints import atomic_save


def spin_up(forecast, months, *, path=None, save_latents=False):
    """Keep history and noise continuous across spin-up and the saved forecast."""
    if path is not None:
        path = Path(path)
        spinup_path = path / "spinup_latents"
    spinup_times, spinup_a, spinup_o = [], [], []
    while forecast.step < months:
        a, o = next(
            forecast
        )  # Latent-only, except the optional first-month initialization cycle.
        if save_latents:
            spinup_times.append(a.time.values[0])
            spinup_a.append(a.values[0])
            spinup_o.append(o.values[0])
        if forecast.step % 12 == 0 or forecast.step == months:
            if save_latents:
                atomic_save(
                    dict(
                        time=np.asarray(spinup_times),
                        latents=dict(
                            atmosphere=np.stack(spinup_a), ocean=np.stack(spinup_o)
                        ),
                        state=forecast.state_dict(),
                    ),
                    spinup_path / f"year-{(forecast.step + 11) // 12:04d}.pt",
                )
                spinup_times, spinup_a, spinup_o = [], [], []
            elif path is not None:
                atomic_save(forecast.state_dict(), path / "spinup_state.pt")
            logging.getLogger(__name__).info(
                "Latent-only spin-up %d/%d months", forecast.step, months
            )
