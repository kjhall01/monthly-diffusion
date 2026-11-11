"""Audit whether the ocean branch has land-crossing geometric pathways.

This is deliberately a geometry/data audit, not a model diagnostic.  It uses
SST validity as a common ocean-support mask, propagates that support through
the actual grid-to-mesh and mesh-down maps, and reports mixed/land-supported
same-mesh edges and example Pacific--Atlantic shortest paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


from monthly_diffusion.experiments.audit_ocean_connectivity import audit

parser = argparse.ArgumentParser()
parser.add_argument("--one-degree", type=Path, required=True)
parser.add_argument("--five-degree", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
results = [
    audit("1deg", args.one_degree, (4, 3, 2), 180, 360, (17, 23, 11)),
    audit("5deg", args.five_degree, (3, 2), 36, 72, (17, 23, 11)),
]
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
