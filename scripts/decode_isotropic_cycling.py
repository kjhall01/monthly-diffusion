"""Decode archived finalists to physical indices; never regenerate forecasts.

All prognostic fields are decoded transiently with the public model API and
correct masks; only ENSO/PDO/NAO/QBO scalars are retained. Annual output shards
are resumable, and the selected years/members/methods are recorded explicitly.
"""
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))

import numpy as np
import torch
import xarray as xr

from monthly_diffusion import MonthlyDiffusionV1
from diagnose_cycling_modes import index_fields, fixed_month_anomalies
from experiment_isotropic_cycling_sweep import OUTPUT, CHECKPOINT, AUDIT, COMPONENTS, MODES, CO2
from diagnose_isotropic_cycling_sweep import assemble,compare,plot


@torch.inference_mode()
def run(label, methods, years=(41,60), member_count=None):
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS required for physical decode")
    torch.set_num_threads(2)
    folder=OUTPUT/label
    config=json.loads((folder/"configuration.json").read_text())
    members=config["members"] if member_count is None else member_count
    if members>config["members"]:
        raise ValueError("Requested unavailable members")
    destination=folder/f"physical_years{years[0]}_{years[1]}_members{members}"
    destination.mkdir(exist_ok=True)
    metadata=dict(methods=list(methods),years=list(years),member_indices=list(range(members)),
                  weight_digest=config["weight_digest"],co2_ppm=CO2,indices="Package definitions; fixed training PDO EOF")
    record=destination/"configuration.json"
    if record.exists() and json.loads(record.read_text())!=metadata:
        raise ValueError("Decode selection changed; choose another output")
    record.write_text(json.dumps(metadata,indent=2))
    model=MonthlyDiffusionV1.load(CHECKPOINT,device="mps")
    if model.weight_digest()!=metadata["weight_digest"]:
        raise ValueError("Checkpoint differs from archived forecasts")
    loading=xr.load_dataarray(AUDIT/"pdo_loading.nc")
    for year in range(years[0],years[1]+1):
        path=destination/f"year-{year:03d}.nc"
        if path.exists():
            continue
        block=torch.load(folder/f"year-{year:03d}.pt",map_location="cpu",weights_only=False)
        indices=[block["methods"].index(m) for m in methods]
        results=[]
        for month,date in enumerate(block["time"]):
            z=[]
            for c in COMPONENTS:
                values=block["predicted"][c][month,indices,:members]
                z.append(xr.DataArray(values.reshape(1,-1,*values.shape[-2:]),
                    dims=("time","member","node","channel"),coords=dict(time=[date],member=np.arange(len(methods)*members))))
            a,o=model.decode(*z,co2=CO2)
            scalar=index_fields(a,o,loading)
            results.append(np.stack([scalar[m].values[0].reshape(len(methods),members) for m in MODES],axis=-1))
            del a,o,scalar
        ds=xr.Dataset(dict(raw=(("time","method","member","mode"),np.stack(results))),
                      coords=dict(time=block["time"],method=list(methods),member=np.arange(members),mode=list(MODES)))
        if not np.isfinite(ds.raw).all():
            raise FloatingPointError("Nonfinite decoded index")
        partial=path.with_suffix(".partial.nc")
        ds.to_netcdf(partial)
        partial.replace(path)
        print(f"Decoded {label} year {year}/{years[1]} · {members} members × {len(methods)} methods",flush=True)
    del model
    torch.mps.empty_cache()
    raw=xr.concat([xr.load_dataset(destination/f"year-{y:03d}.nc") for y in range(years[0],years[1]+1)],"time")
    ref=xr.load_dataset(AUDIT/"decoded_reference_indices.nc")
    values,ref_values=fixed_month_anomalies(raw.raw.values,raw.time.values,np.column_stack([ref[m] for m in MODES]),ref.time.values)
    ds=assemble(folder).sel({"method":list(methods),"member":np.arange(members)})
    ds["physical"]=xr.DataArray(values,dims=("time","method","member","mode"),coords=raw.raw.coords)
    ds["physical_reference"]=(("training_time","mode"),ref_values)
    ds.attrs.update(physical_decoded_years=f"{years[0]}–{years[1]}",
                    unavailable_physical_months="Not decoded; NaN outside selected window")
    ds.to_netcdf(destination/"mode_trajectories.nc")
    compare(ds,destination,windows={f"years{years[0]}_{years[1]}":(years[0]-1,years[1])})
    plot(ds,destination,methods=list(methods),window=(years[0]-1,years[1]))
    (destination/"code_sha256.txt").write_text(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()+"\n")
    return destination


def main():
    # Do not compete for the GPU with the latent experiments. The confirmation
    # statistics are written only after both 100-year cohorts have finished.
    complete=OUTPUT/"confirmation_pooled/ranking.csv"
    while not complete.exists():
        print("Waiting for both independent 100-year latent confirmations",flush=True)
        time.sleep(30)
    decision=json.loads((OUTPUT/"selection.json").read_text())
    methods=[f"cycle{n}" for n in decision["confirmation_cycles"]]+["observed","observed_independent"]
    for label in ("confirm_january","confirm_july"):
        # Fixed first-ten subset, never chosen for favorable statistics. Full
        # 20-member latent trajectories are retained for all 100 years.
        run(label,methods,years=(81,100),member_count=10)


if __name__=="__main__":
    main()
