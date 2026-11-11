"""Advance coupled latent histories one month at a time."""

import numpy as np
import torch
import xarray as xr
from ..data.dataloader import next_month, month_features, co2_values
from ..random_state import rng_state, restore_rng
from .initialization import initialize_history


class Forecast:
    """A monthly latent iterator; each call retains the two latest history states."""

    def __init__(
        self,
        model,
        atmos,
        ocean,
        *,
        co2,
        members,
        seed,
        gaussian_start=None,
        encoded_noise_start=None,
        stochastic_start=None,
    ):
        self.model = model
        self.co2 = co2
        self.members = members
        self.seed = seed
        (self.history, self.templates, self.times, self.cycle_months, self.rng) = (
            initialize_history(
                model,
                atmos,
                ocean,
                co2=co2,
                members=members,
                seed=seed,
                gaussian_start=gaussian_start,
                encoded_noise_start=encoded_noise_start,
                stochastic_start=stochastic_start,
            )
        )
        self.noise = (None, None)
        self.step = 0

    def __iter__(self):
        return self

    @torch.no_grad()
    def __next__(self):
        model = self.model
        time = next_month(self.times[0])
        time_array = xr.DataArray([time], dims="time", coords={"time": [time]})
        co2_values(
            self.co2, time_array
        )  # fail before consuming RNG if forcing is missing
        dates = xr.DataArray(
            self.times + [time], dims="time", coords={"time": self.times + [time]}
        )
        calendar = (
            month_features(dates)
            .reshape(1, 6)
            .expand(self.members, -1)
            .to(model.device)
        )
        ah = torch.cat((self.history[0][:, 0], self.history[0][:, 1]), -1)
        oh = torch.cat((self.history[1][:, 0], self.history[1][:, 1]), -1)
        external_rng = rng_state(model.device)
        restore_rng(self.rng)
        try:
            model.transition.eval()
            a, o, na, no = model.transition.sample(
                calendar, ah, oh, previous_noise=self.noise
            )
            self.rng = rng_state(model.device)
        finally:
            restore_rng(external_rng)
        self.noise = na, no
        self.history = [
            torch.stack((a, self.history[0][:, 0]), 1),
            torch.stack((o, self.history[1][:, 0]), 1),
        ]
        self.times = [time, self.times[0]]
        self.step += 1
        outputs = []
        for value, template in zip((a, o), self.templates):
            coords = {k: v for k, v in template.coords.items() if k != "time"}
            coords.update(time=[time], member=np.arange(self.members))
            outputs.append(
                xr.DataArray(
                    value.cpu().numpy()[None],
                    dims=("time", "member", "node", "channel"),
                    coords=coords,
                    name=template.name,
                    attrs=template.attrs,
                )
            )
        if self.step <= self.cycle_months:
            # One physical projection, exactly as tested: only replace current,
            # not previous, and keep diffusion/AR1 noise and RNG continuous.
            fields = model.decode(*outputs, co2=self.co2)
            projected = [[], []]
            for member in range(self.members):
                encoded = model.encode(
                    *(ds.isel(member=member, drop=True) for ds in fields), co2=self.co2
                )
                for i, z in enumerate(encoded):
                    if not np.isfinite(z.values).all():
                        raise FloatingPointError(
                            "Initialization cycle produced nonfinite latents"
                        )
                    projected[i].append(z.values[0])
            for i in range(2):
                self.history[i][:, 0] = torch.tensor(
                    np.stack(projected[i]), device=model.device
                )
        return tuple(outputs)

    def state_dict(self):
        state = dict(
            history=[v.cpu() for v in self.history],
            noise=[None if n is None else n.cpu() for n in self.noise],
            times=self.times,
            step=self.step,
            rng=self.rng,
        )
        if self.cycle_months:
            state["cycle_months"] = self.cycle_months
        return state

    def load_state_dict(self, state):
        self.history = [v.to(self.model.device) for v in state["history"]]
        self.noise = tuple(
            None if v is None else v.to(self.model.device) for v in state["noise"]
        )
        self.times, self.step, self.rng = state["times"], state["step"], state["rng"]
        self.cycle_months = state.get("cycle_months", 0)
