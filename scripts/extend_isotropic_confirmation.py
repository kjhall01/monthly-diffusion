"""One planned precision extension, preserving the completed 100-year results.

No candidate selection or tolerance changes. Continue all four arms for both
seasons to 200 years. Reuse immutable initial/annual states through symlinks;
new years and summaries live in separate directories. Wait for the physical
decode job to release the GPU before continuing inference.
"""

from pathlib import Path
import sys
import time
import json

sys.path.insert(0, str(Path(__file__).resolve().parent))
from monthly_diffusion.experiments import (
    experiment_isotropic_cycling_sweep as experiment,
)
from monthly_diffusion.experiments import (
    diagnose_isotropic_cycling_sweep as diagnostics,
)

names = ("confirm_january", "confirm_july")
while not all(
    (experiment.OUTPUT / n / "physical_years81_100_members10/ranking.csv").exists()
    for n in names
):
    time.sleep(30)
print(
    "Physical-index decoding complete; continuing both latent confirmations to 200 years",
    flush=True,
)
if not experiment.torch.backends.mps.is_available():
    raise RuntimeError("MPS required")
experiment.torch.set_num_threads(2)
decision = json.loads((experiment.OUTPUT / "selection.json").read_text())
model = experiment.MonthlyDiffusionV1.load(experiment.CHECKPOINT, device="mps")
if model.weight_digest() != decision["weight_digest"]:
    raise ValueError("Checkpoint changed")
windows = {
    "years61_200": (60, 200),
    "years101_200": (100, 200),
    "years161_200": (160, 200),
}
outputs = []
for name, first, seed in (
    (names[0], "1800-01", 20261019),
    (names[1], "1800-07", 20261029),
):
    source = experiment.OUTPUT / name
    label = name + "_200years"
    destination = experiment.OUTPUT / label
    destination.mkdir(exist_ok=True)
    for filename in [
        "initial_states.pt",
        "observed_initial_dates.csv",
        *[f"year-{y:03d}.pt" for y in range(1, 101)],
    ]:
        target = destination / filename
        if not (source / filename).exists():
            raise FileNotFoundError(source / filename)
        if not target.exists():
            target.symlink_to(Path("..") / name / filename)
    experiment.run(
        model,
        label,
        cycles=tuple(decision["confirmation_cycles"]),
        members=20,
        years=200,
        first=first,
        seed=seed,
    )
    ds = diagnostics.assemble(destination)
    diagnostics.compare(ds, destination, windows=windows)
    diagnostics.plot(ds, destination, window=(100, 200))
    diagnostics.convergence(ds, destination)
    diagnostics.bulk_latents(destination, {"years101_200": (100, 200)})
    outputs.append(destination)
destination = experiment.OUTPUT / "confirmation_200years_pooled"
ds = diagnostics.pool_confirmation(
    outputs, destination, windows=windows, plot_window=(100, 200)
)
diagnostics.anomaly_sensitivity(ds, destination, (100, 200))
diagnostics.joint_and_seasonal(ds, destination, (100, 200))
diagnostics.convergence(ds, destination)
