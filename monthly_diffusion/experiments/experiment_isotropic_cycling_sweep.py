"""Controlled cycling-duration sweep; immutable checkpoint, resumable latents.

Use standard-normal *input fields*, encode them, cycle only during an initial
finite interval, then use the unmodified latent transition. Real-data controls
use multiple calendar-matched validation histories, not one chosen ENSO phase.
Only latent trajectories/restarts are archived; finalists can be decoded later.
Run from frozen/: python scripts/experiment_isotropic_cycling_sweep.py
"""

from pathlib import Path
import copy
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import pandas as pd
import torch
import xarray as xr

from monthly_diffusion import MonthlyDiffusionV1
from monthly_diffusion.training.common import atomic_save, rng_state, restore_rng
from monthly_diffusion.data.dataloader import next_month, month_features

OUTPUT = ROOT / "analysis/isotropic_cycling_convergence"
CHECKPOINT = ROOT / "runs/mdv1-era5-oras5-sst-support/model.pt"
AUDIT = ROOT / "analysis/cycling_duration_modes"
COMPONENTS = ("atmosphere", "ocean")
MODES = ("ENSO", "PDO", "NAO", "QBO")
CO2 = 284.2665710449219
CYCLES = (0, 1, 3, 6, 12, 30, 60, 120)


def observed_pairs(cache, first, members, seed):
    """Balanced sampling of adjacent actual histories with matching calendar."""
    dates = np.asarray(cache["validation_time"]).astype("datetime64[M]")
    current = np.datetime64(first, "M") - np.timedelta64(1, "M")
    month = pd.Timestamp(current).month
    candidates = [
        i
        for i in range(1, len(dates))
        if pd.Timestamp(dates[i]).month == month
        and dates[i] - dates[i - 1] == np.timedelta64(1, "M")
    ]
    if not candidates:
        raise ValueError("No adjacent real histories with matching calendar")
    rng = np.random.default_rng(seed)
    indices = np.concatenate(
        [rng.permutation(candidates) for _ in range(1 + members // len(candidates))]
    )[:members]
    return np.stack([indices, indices - 1], axis=1)


@torch.inference_mode()
def advance_batched(forecasts):
    """Same package sampler, batching independent arms after cycling finishes.

    Draw each arm's terminal noise with its OWN archived RNG/AR1 state before
    batching. No draws, time levels, calendar features, or solver steps change.
    The package sample(..., noise=...) then evaluates all arms together.
    """
    model = next(iter(forecasts.values())).model
    histories, calendars, noises = [[], []], [], [[], []]
    external_rng = rng_state(model.device)
    try:
        for f in forecasts.values():
            if f.step < f.cycle_months:
                raise ValueError("Batched latent-only step called during cycling")
            target = next_month(f.times[0])
            dates = xr.DataArray(
                f.times + [target], dims="time", coords={"time": f.times + [target]}
            )
            calendars.append(
                month_features(dates)
                .reshape(1, 6)
                .expand(f.members, -1)
                .to(model.device)
            )
            ah, oh = [torch.cat((h[:, 0], h[:, 1]), -1) for h in f.history]
            histories[0].append(ah)
            histories[1].append(oh)
            restore_rng(f.rng)
            na, no = model.transition.terminal_noise(ah, oh, f.noise)
            f.rng = rng_state(model.device)
            noises[0].append(na)
            noises[1].append(no)
    finally:
        restore_rng(external_rng)
    model.transition.eval()
    a, o, _, _ = model.transition.sample(
        torch.cat(calendars),
        torch.cat(histories[0]),
        torch.cat(histories[1]),
        noise=tuple(torch.cat(n) for n in noises),
    )
    outputs, offset = {}, 0
    for j, (method, f) in enumerate(forecasts.items()):
        target = next_month(f.times[0])
        values = [v[offset : offset + f.members] for v in (a, o)]
        f.history = [torch.stack((v, h[:, 0]), 1) for v, h in zip(values, f.history)]
        f.noise = (noises[0][j], noises[1][j])
        f.times, f.step = [target, f.times[0]], f.step + 1
        z = []
        for value, template in zip(values, f.templates):
            coords = {k: v for k, v in template.coords.items() if k != "time"}
            coords.update(time=[target], member=np.arange(f.members))
            z.append(
                xr.DataArray(
                    value.cpu().numpy()[None],
                    dims=("time", "member", "node", "channel"),
                    coords=coords,
                    name=template.name,
                    attrs=template.attrs,
                )
            )
        outputs[method] = z
        offset += f.members
    return outputs


@torch.inference_mode()
def run(
    model,
    label,
    *,
    cycles=CYCLES,
    members=10,
    years=60,
    first="1800-01",
    seed=20261009,
    stop_after_years=None,
):
    output = OUTPUT / label
    output.mkdir(parents=True, exist_ok=True)
    methods = [f"cycle{n}" for n in cycles] + ["observed", "observed_independent"]
    metadata = dict(
        checkpoint=str(CHECKPOINT),
        weight_digest=model.weight_digest(),
        ae_digest=model.weight_digest(autoencoder_only=True),
        seed=seed,
        first=first,
        cycles=list(cycles),
        members=members,
        co2_ppm=CO2,
        methods=methods,
        initialization="masked independent N(0,1) normalized physical-input fields -> encode",
        pairing="identical diffusion/terminal-AR1 innovations; independent observed control uses seed+7300",
        observation_source="balanced adjacent validation histories, own encoded CO2, matched calendar, relabeled year",
        history_order="current, previous",
        archive="predicted and fed-back latents + complete annual restart",
    )
    record = output / "configuration.json"
    if record.exists() and json.loads(record.read_text()) != metadata:
        raise ValueError("Experiment identity changed; use another output folder")
    record.write_text(json.dumps(metadata, indent=2))
    cache = torch.load(
        CHECKPOINT.parent / "latents.pt", map_location="cpu", weights_only=False
    )
    if cache["ae_digest"] != metadata["ae_digest"]:
        raise ValueError("Cached real histories use another autoencoder")

    forecasts = {}
    init_path = output / "initial_states.pt"
    if init_path.exists():
        saved_initial = torch.load(init_path, map_location="cpu", weights_only=False)
    else:
        iso = model.initialize(
            co2=CO2, members=members, seed=seed, encoded_noise_start=first
        )
        saved_initial = {m: copy.deepcopy(iso.state_dict()) for m in methods}
        initial_dates = []
        for j, method in enumerate(("observed", "observed_independent")):
            pair = observed_pairs(cache, first, members, seed + 101 + j)
            state = saved_initial[method]
            for i, c in enumerate(COMPONENTS):
                state["history"][i] = cache["validation"][c][pair].clone()
            if j:
                independent = model.initialize(
                    co2=CO2, members=members, seed=seed + 7300, gaussian_start=first
                )
                state["rng"] = independent.rng
            dates = np.asarray(cache["validation_time"])[pair]
            for member, d in enumerate(dates):
                initial_dates.append(
                    dict(
                        method=method,
                        member=member,
                        current=str(d[0]),
                        previous=str(d[1]),
                    )
                )
        pd.DataFrame(initial_dates).to_csv(
            output / "observed_initial_dates.csv", index=False
        )
        atomic_save(saved_initial, init_path)
    for method in methods:
        f = model.initialize(co2=CO2, members=members, seed=seed, gaussian_start=first)
        f.load_state_dict(copy.deepcopy(saved_initial[method]))
        f.cycle_months = int(method[5:]) if method.startswith("cycle") else 0
        forecasts[method] = f

    shards = sorted(output.glob("year-*.pt"))
    if [p.name for p in shards] != [
        f"year-{y:03d}.pt" for y in range(1, len(shards) + 1)
    ]:
        raise ValueError("Missing annual restart block")
    if shards:
        block = torch.load(shards[-1], map_location="cpu", weights_only=False)
        for method, f in forecasts.items():
            f.load_state_dict(block["states"][method])
            if f.step != 12 * len(shards):
                raise ValueError("Restart step disagrees with archived dates")
    longest = f"cycle{max(cycles)}"
    execution = [longest] + [m for m in methods if m != longest]
    last_year = min(years, stop_after_years) if stop_after_years else years
    started = time.monotonic()
    for year in range(len(shards) + 1, last_year + 1):
        year_started = time.monotonic()
        predicted = {c: [] for c in COMPONENTS}
        fed_back = {c: [] for c in COMPONENTS}
        dates, corrections = [], []
        for month in range(12):
            step = (year - 1) * 12 + month + 1
            expected = (
                np.datetime64(first, "M") + np.timedelta64(step - 1, "M")
            ).astype("datetime64[ns]")
            monthly = {}
            old_histories = {
                m: [h[:, 0].clone() for h in f.history] for m, f in forecasts.items()
            }
            batched = advance_batched(forecasts) if step > max(cycles) else None
            for method in execution:
                f = forecasts[method]
                duration = int(method[5:]) if method.startswith("cycle") else 0
                previous = old_histories[method]
                if batched is not None:
                    z = batched[method]
                elif (
                    method != longest
                    and method.startswith("cycle")
                    and step <= duration
                ):
                    # Exactly common prefix, computed once, not independently
                    # re-encoded and perturbed by GPU roundoff for every arm.
                    f.load_state_dict(copy.deepcopy(forecasts[longest].state_dict()))
                    f.cycle_months = duration
                    z = monthly[longest]["z"]
                else:
                    z = next(f)
                if f.times[0] != expected or f.step != step:
                    raise ValueError("Month skipped or history time misordered")
                before, after = [], []
                for i in range(2):
                    torch.testing.assert_close(
                        f.history[i][:, 1], previous[i], atol=0, rtol=0
                    )
                    before.append(z[i].values[0].copy())
                    after.append(f.history[i][:, 0].cpu().numpy().copy())
                    if (
                        not np.isfinite(before[-1]).all()
                        or not np.isfinite(after[-1]).all()
                    ):
                        raise FloatingPointError("Nonfinite latent")
                    if step > duration:
                        np.testing.assert_array_equal(before[-1], after[-1])
                monthly[method] = dict(z=z, before=before, after=after)
            base = forecasts["observed"]
            for method, f in forecasts.items():
                if method == "observed_independent":
                    continue
                for n, target in zip(f.noise, base.noise):
                    torch.testing.assert_close(n, target, atol=0, rtol=0)
                for key in ("torch", "mps"):
                    if key in f.rng:
                        torch.testing.assert_close(
                            f.rng[key], base.rng[key], atol=0, rtol=0
                        )
            for i, c in enumerate(COMPONENTS):
                predicted[c].append(
                    np.stack([monthly[m]["before"][i] for m in methods])
                )
                fed_back[c].append(np.stack([monthly[m]["after"][i] for m in methods]))
            corrections.append(
                np.stack(
                    [
                        np.stack(
                            [
                                np.sqrt(
                                    np.mean(
                                        (
                                            monthly[m]["before"][i]
                                            - monthly[m]["after"][i]
                                        )
                                        ** 2,
                                        axis=(1, 2),
                                    )
                                )
                                for i in range(2)
                            ],
                            axis=-1,
                        )
                        for m in methods
                    ]
                )
            )
            dates.append(expected)
        atomic_save(
            dict(
                time=np.asarray(dates),
                methods=methods,
                predicted={c: np.stack(v) for c, v in predicted.items()},
                fed_back={c: np.stack(v) for c, v in fed_back.items()},
                correction=np.stack(corrections),
                states={m: f.state_dict() for m, f in forecasts.items()},
            ),
            output / f"year-{year:03d}.pt",
        )
        elapsed = time.monotonic() - started
        speed = elapsed / (year - len(shards))
        current_speed = time.monotonic() - year_started
        print(
            f"{label} year {year}/{years} | {current_speed:.1f}s/year current ({speed:.1f} cumulative) | ETA {(years - year) * current_speed / 60:.1f}min | MPS {torch.mps.current_allocated_memory() / 2**30:.2f} GiB",
            flush=True,
        )
    return output
