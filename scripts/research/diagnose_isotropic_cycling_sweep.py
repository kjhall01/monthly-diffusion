"""Compare finite cycling treatments with real-initialized rollout controls.

Read annual latent shards in bounded memory. No readout is refitted. Confidence
intervals resample whole members, never individual months or grid cells.
"""

# Make the repository workflow helpers available when running this file directly.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import torch

from workflows.experiments.experiment_isotropic_cycling_sweep import OUTPUT


from workflows.experiments.diagnose_isotropic_cycling_sweep import assemble, compare, plot, convergence, bulk_latents

label = "screen_january"

torch.set_num_threads(2)
folder = OUTPUT / label
ds = assemble(folder)
compare(ds, folder)
plot(ds, folder)
convergence(ds, folder)
if len(ds.time) >= 720:
    bulk_latents(folder, {"years21_60": (20, 60), "years41_60": (40, 60)})
