"""Read-only, one-year preprocessing smoke check against a final source archive."""

import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from monthly_diffusion.data.configs import mdv1_variables
from monthly_diffusion.data.sources import open_canonical_state, open_co2
from monthly_diffusion.data.preprocessing import Preprocessor
from monthly_diffusion.data.dataloader import (
    co2_values,
    validate_fields,
    validate_units,
)


parser = argparse.ArgumentParser()
parser.add_argument("source_yaml", type=Path)
parser.add_argument("--state-zarr", required=True)
parser.add_argument("--co2-netcdf", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
source = yaml.safe_load(args.source_yaml.read_text())
a, o, s = open_canonical_state(args.state_zarr, expected_attrs=source["expected_attrs"])
a, o = a.sel(time=slice("1966-01", "1966-12")), o.sel(time=slice("1966-01", "1966-12"))
co2 = open_co2(args.co2_netcdf, expected_sha256=source["co2_sha256"])
specs = mdv1_variables()
for name, ds in (("atmosphere", a), ("ocean", o)):
    selected = [v for v in specs if v.component == name]
    validate_fields(ds, [v.name for v in selected], s)
    validate_units(ds, selected)
pre = Preprocessor(specs)
pre.fit(a, o, s, co2_values(co2, a.time))
result = dict(source=args.state_zarr, months=len(a.time), variables={})
for component, ds in (("atmosphere", a), ("ocean", o)):
    scaled = pre.encode(ds, component).compute()
    decoded = pre.decode(scaled, component)
    for name in scaled:
        values = scaled[name].values
        assert not np.isinf(values).any()
        result["variables"][name] = dict(
            std=float(scaled[name].astype("float64").std()),
            finite_cells=int(np.isfinite(values).sum()),
            max_roundtrip_error=float(abs(decoded[name] - ds[name]).max().compute()),
            units=ds[name].attrs["units"],
        )
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
