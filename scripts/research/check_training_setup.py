"""One-year real-data smoke test; never writes a training checkpoint.

Verify the pinned source, full split coverage, optional matched model config,
transforms, masked reconstruction gradients and three-step diffusion gradients.
The one-year statistics and randomly initialized weights are disposable: full
training independently fits its statistics on 1966--2006 and starts from seed.
"""

# Make the repository workflow helpers available when running this file directly.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workflows.training.losses import rollout_loss
import argparse
import json
import logging
import resource
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from workflows.legacy_model import MonthlyDiffusionV1
from workflows.legacy_records import read_settings
from workflows.data.sources import open_canonical_state, open_co2
from workflows.data.dataloader import PhysicalStates, co2_values, month_features, validate_months
from monthly_diffusion.preprocessing import inverse_transform, transform
from workflows.training.losses import reconstruction_mse


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--reference-config", type=Path)
parser.add_argument("--device", default="cpu")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
torch.set_num_threads(4)
if args.device == "mps" and not torch.backends.mps.is_available():
    raise RuntimeError("MPS requested but unavailable; refusing CPU fallback")
cfg, training = read_settings(args.config)
if args.reference_config:
    assert cfg.to_dict() == read_settings(args.reference_config)[0].to_dict(), (
        "Model configurations differ"
    )
source = yaml.safe_load(args.source.read_text())
a, o, statics = open_canonical_state(
    source["state_zarr"], expected_attrs=source.get("expected_attrs")
)
co2 = open_co2(source["co2_netcdf"], expected_sha256=source.get("co2_sha256"))
validate_months(a.time)
assert np.array_equal(a.time, o.time)
co2_values(co2, a.time)  # full archive must be covered, including 2025
for lo, hi, n in ((1966, 2006, 492), (1958, 1965, 96), (2007, 2014, 96)):
    part = a.sel(time=slice(str(lo), str(hi)))
    assert part.sizes["time"] == n, f"Incomplete {lo}--{hi} split"
a, o = a.sel(time="1966").load(), o.sel(time="1966").load()
model = MonthlyDiffusionV1(statics=statics, settings=cfg, device=args.device)
model.prepare(a, o, co2=co2)
for graph in model.geometry.down_maps:
    assert torch.bincount(graph.edge_set.receiver, minlength=graph.n_target).min() > 0
assert model.autoencoder.ocean.mask_aware == cfg.ocean_mask_aware
initial_digest = model.weight_digest()
report = dict(
    source=source,
    config=cfg.to_dict(),
    device=str(model.device),
    pilot_year=1966,
    note="One-year smoke test only; no fitted statistics or weights reused for training",
    parameters_autoencoder=sum(p.numel() for p in model.autoencoder.parameters()),
    parameters_transition=sum(p.numel() for p in model.transition.parameters()),
    initial_weight_digest=initial_digest,
    variables={},
)
if cfg.ocean_support == "sst":
    report["ocean_support"] = dict(
        policy="training_sst",
        cells=int(model.preprocessor.ocean_support_mask.sum()),
        excluded_finite_cells={
            v.name: int(
                (
                    np.isfinite(o[v.name]).any("time")
                    & ~model.preprocessor.ocean_support_mask
                ).sum()
            )
            for v in cfg.variables
            if v.component == "ocean"
        },
    )
for component, fields in (("atmosphere", a), ("ocean", o)):
    if component == "ocean":
        fields = model.preprocessor.mask_ocean(fields)
    scaled = model.preprocessor.encode(fields, component)
    decoded = model.preprocessor.decode(scaled, component)
    for spec in cfg.variables:
        if spec.component != component:
            continue
        raw, normalized = fields[spec.name], scaled[spec.name]
        assert np.isfinite(raw).equals(np.isfinite(normalized)), spec.name
        # Epsilon guards intentionally perturb physical boundary values.
        guarded = inverse_transform(transform(raw, spec), spec)
        error = abs(decoded[spec.name] - guarded)
        tolerance = 1.0e-4 * max(spec.scale, float(abs(guarded).max()))
        assert float(error.max()) <= tolerance, spec.name
        report["variables"][spec.name] = dict(
            units=raw.attrs["units"],
            normalization_std=model.preprocessor.stats[spec.name]["std"],
            valid_fraction=float(np.isfinite(raw).mean()),
            max_roundtrip_error=float(error.max()),
        )
dataset = PhysicalStates(a, o, model.preprocessor, co2_values(co2, a.time))
batch = next(iter(DataLoader(dataset, batch_size=training["atmosphere"]["batch_size"])))
batch = {k: v.to(model.device) for k, v in batch.items()}
b = len(batch["month"])
lat = torch.tensor(statics.lat.values, device=model.device)
loss = 0.0
started = time.monotonic()
for component in ("atmosphere", "ocean"):
    prediction, z = getattr(model.autoencoder, component)(
        batch[component], batch["month"], batch["co2"]
    )
    assert torch.isfinite(prediction).all() and torch.isfinite(z).all()
    per_field = reconstruction_mse(prediction, batch[component], lat)
    weights = per_field.new_tensor(
        [v.weight for v in cfg.variables if v.component == component]
    )
    loss = loss + (per_field * weights).sum()
loss.backward()
assert all(
    p.grad is None or torch.isfinite(p.grad).all()
    for p in model.autoencoder.parameters()
)
report.update(
    reconstruction_loss=float(loss.detach()),
    ae_forward_backward_seconds=time.monotonic() - started,
)
model.autoencoder.zero_grad(set_to_none=True)
del prediction, z, loss, per_field, batch
# The diffusion shape, ordering, masks and objective are unchanged by the
# reanalysis adapter. Exercise the real three-step loss with full-size
# synthetic normalized latents, without fitting a toy transition.
c = cfg.latent_channels
nodes = model.geometry.meshes[-1].n_nodes
h = training["diffusion"]["rollout_steps"]
sequence = dict(
    atmosphere=torch.randn(b, h + 2, nodes, c, device=model.device),
    ocean=torch.randn(b, h + 2, nodes, c, device=model.device),
    month=month_features(a.time.isel(time=slice(0, h + 2)))
    .to(model.device)[None]
    .expand(b, -1, -1),
)
terms = rollout_loss(model.transition, sequence['atmosphere'], sequence['ocean'], sequence['month']).metrics()
assert all(torch.isfinite(v).all() for v in terms.values())
terms["loss"].backward()
assert all(
    p.grad is None or torch.isfinite(p.grad).all()
    for p in model.transition.parameters()
)
report["transition_terms"] = {k: float(v.detach()) for k, v in terms.items()}
assert model.weight_digest() == initial_digest, "Smoke test must not update weights"
report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
    1 if sys.platform == "darwin" else 1024
)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2))
logging.info(
    "Passed source, config, transforms, reconstruction and diffusion gradients: %s",
    args.output,
)
