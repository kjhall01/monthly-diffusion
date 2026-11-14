"""Short synthetic compute canary; never reads data or modifies real checkpoints.

Run each mode in a separate process for interpretable peak RSS. The joint mode
recreates the old workload (two online history encodes, three detached target
encodes, current reconstruction, three-step diffusion). Targets use identical
weights under no_grad; no second full AE is needed for a timing experiment.
"""

# Make the repository workflow helpers available when running this file directly.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workflows.training.losses import rollout_loss
import argparse
import json
import resource
import time
import numpy as np
import torch
import xarray as xr
from workflows.legacy_model import MonthlyDiffusionV1
from workflows.training.losses import reconstruction_mse


from workflows.devices import synchronize

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", choices=("autoencoder", "joint", "diffusion", "encode"), required=True
)
parser.add_argument("--device", default="mps")
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--repeats", type=int, default=2)
parser.add_argument("--resolution", type=int, default=1)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(4)
device = torch.device(args.device)
if device.type == "mps" and not torch.backends.mps.is_available():
    raise RuntimeError(
        "MPS is not available in this process; do not silently benchmark CPU"
    )
nlat, nlon = 180 // args.resolution, 360 // args.resolution
statics = xr.Dataset(
    dict(
        phi_s=(("lat", "lon"), np.zeros((nlat, nlon), dtype="float32")),
        lsm=(("lat", "lon"), np.zeros((nlat, nlon), dtype="float32")),
    ),
    coords=dict(
        lat=np.linspace(-90 + 90 / nlat, 90 - 90 / nlat, nlat),
        lon=np.linspace(180 / nlon, 360 - 180 / nlon, nlon),
    ),
)
torch.manual_seed(20260716)
model = MonthlyDiffusionV1(statics=statics, device=device)
# Synthetic topography is already zero; no physical-data fit is needed here.
model.preprocessor.topo_mean, model.preprocessor.topo_std = 0.0, 1.0
model._build()
b = args.batch_size
ae, transition = model.autoencoder, model.transition
optimizer = torch.optim.AdamW(
    list(ae.parameters()) + list(transition.parameters()), lr=0.001, eps=1e-6
)
a = torch.randn(b, 16, nlat, nlon, device=device)
o = torch.randn(b, 9, nlat, nlon, device=device)
month = torch.randn(b, 5, 2, device=device)
co2 = torch.randn(b, 5, 1, device=device)
lat = torch.tensor(statics.lat.values, dtype=torch.float32, device=device)
weights = {
    comp: torch.tensor(
        [v.weight for v in model.config.variables if v.component == comp], device=device
    )
    for comp in ("atmosphere", "ocean")
}
times = []
for repeat in range(args.repeats + 1):
    optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    start = time.perf_counter()
    if args.mode in ("autoencoder", "joint"):
        loss, latent = 0.0, {}
        for name, data in (("atmosphere", a), ("ocean", o)):
            branch = getattr(ae, name)
            prediction, z = branch(data, month[:, 1], co2[:, 1])
            loss = (
                loss + (reconstruction_mse(prediction, data, lat) * weights[name]).sum()
            )
            if args.mode == "joint":
                previous = branch.encode(data, month[:, 0], co2[:, 0]).detach()
                with torch.no_grad():
                    targets = [
                        branch.encode(data, month[:, i], co2[:, i])
                        for i in range(2, 5)
                    ]
                latent[name] = torch.stack([previous, z.detach(), *targets], 1)
        if args.mode == "joint":
            loss = loss + rollout_loss(transition, dict(**latent, month=month)['atmosphere'], dict(**latent, month=month)['ocean'], dict(**latent, month=month)['month']).metrics()["loss"]
        loss.backward()
        optimizer.step()
    elif args.mode == "encode":
        with torch.no_grad():
            ae.atmosphere.encode(a, month[:, 1], co2[:, 1])
            ae.ocean.encode(o, month[:, 1], co2[:, 1])
    else:
        loss = rollout_loss(transition, torch.randn(b, 5, 162, 4, device=device), torch.randn(b, 5, 162, 4, device=device), month).metrics()["loss"]
        loss.backward()
        optimizer.step()
    synchronize(device)
    seconds = time.perf_counter() - start
    if repeat:
        times.append(seconds)
    print(
        f"{args.mode}: {'warmup' if repeat == 0 else 'measured'} {seconds:.3f}s",
        flush=True,
    )
result = dict(
    mode=args.mode,
    device=str(device),
    batch_size=b,
    resolution_degrees=args.resolution,
    seconds_per_batch=times,
    mean_seconds=float(np.mean(times)),
    training_epoch_compute_seconds=float(
        np.mean(times)
        * np.ceil((488 if args.mode in ("joint", "diffusion") else 492) / b)
    ),
    peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    parameters_autoencoder=sum(p.numel() for p in ae.parameters()),
    parameters_transition=sum(p.numel() for p in transition.parameters()),
    notes="Synthetic float32 compute only. Excludes data I/O, validation, preprocessing and checkpoint writes. macOS peak RSS bytes.",
)
if device.type == "mps":
    result.update(
        mps_current_bytes=torch.mps.current_allocated_memory(),
        mps_driver_bytes=torch.mps.driver_allocated_memory(),
    )
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
