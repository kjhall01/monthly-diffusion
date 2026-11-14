"""Shared cycling prefix, forecast dates, and exact annual restart."""

import numpy as np
import torch

from test_cycled_initialization import CycleModel

from workflows.experiments import experiment_isotropic_cycling_sweep as sweep


def test_real_histories_are_adjacent_calendar_matched_and_balanced():
    dates = np.arange("1958-01", "1966-01", dtype="datetime64[M]")
    cache = {"validation_time": dates}
    for first in ("1800-01", "1800-07"):
        pairs = sweep.observed_pairs(cache, first, 16, 44)
        np.testing.assert_array_equal(pairs[:, 0]-pairs[:, 1], 1)
        target = (np.datetime64(first, "M")-np.timedelta64(1, "M")).astype(int) % 12
        np.testing.assert_array_equal(dates[pairs[:, 0]].astype(int) % 12, target)
        _, counts = np.unique(pairs[:, 0], return_counts=True)
        assert counts.max()-counts.min() <= 1


def test_sweep_prefix_cutoffs_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "OUTPUT", tmp_path)
    monkeypatch.setattr(sweep, "CHECKPOINT", tmp_path / "model.pt")
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 0)
    dates = np.arange("1958-01", "1966-01", dtype="datetime64[M]")
    cache = dict(ae_digest="same", validation_time=dates,
        validation={c: torch.arange(96*4).reshape(96,4,1).float()/100 for c in sweep.COMPONENTS})
    torch.save(cache, tmp_path / "latents.pt")
    model = CycleModel()
    model.weight_digest = lambda autoencoder_only=False: "same"
    # The tiny test transition has no explicit-noise batching API; test the
    # sweep/restarts separately from the real-sampler MPS equivalence canary.
    monkeypatch.setattr(sweep, "advance_batched", lambda fs: {m: next(f) for m,f in fs.items()})
    options = dict(cycles=(0,1,3,6), members=2, years=2, seed=81)
    sweep.run(model, "whole", **options)
    sweep.run(model, "resumed", **options, stop_after_years=1)
    sweep.run(model, "resumed", **options)
    for y in (1,2):
        a = torch.load(tmp_path/"whole"/f"year-{y:03d}.pt", weights_only=False)
        b = torch.load(tmp_path/"resumed"/f"year-{y:03d}.pt", weights_only=False)
        for c in sweep.COMPONENTS:
            np.testing.assert_array_equal(a["predicted"][c], b["predicted"][c])
            np.testing.assert_array_equal(a["fed_back"][c], b["fed_back"][c])
            if y == 1:
                for i, duration in enumerate(options["cycles"]):
                    np.testing.assert_array_equal(a["fed_back"][c][:duration, i], a["fed_back"][c][:duration, 3])
        for i, method in enumerate(a["methods"]):
            duration = int(method[5:]) if method.startswith("cycle") else 0
            months = np.arange((y-1)*12+1, y*12+1)
            np.testing.assert_array_equal(a["correction"][months > duration, i], 0)
