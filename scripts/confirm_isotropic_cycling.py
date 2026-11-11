"""Independent January/July confirmation after the recorded duration selection.

Run: python scripts/confirm_isotropic_cycling.py
Only reads the selection; does not optimize durations on the confirmation runs.
"""
from pathlib import Path
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import experiment_isotropic_cycling_sweep as experiment
import diagnose_isotropic_cycling_sweep as diagnostics


def main():
    decision=json.loads((experiment.OUTPUT/"selection.json").read_text())
    cycles=tuple(decision["confirmation_cycles"])
    if not experiment.torch.backends.mps.is_available():
        raise RuntimeError("MPS required")
    experiment.torch.set_num_threads(2)
    model=experiment.MonthlyDiffusionV1.load(experiment.CHECKPOINT,device="mps")
    if model.weight_digest()!=decision["weight_digest"]:
        raise ValueError("Checkpoint changed after selection")
    for label,first,seed in (("confirm_january","1800-01",20261019),
                             ("confirm_july","1800-07",20261029)):
        folder=experiment.run(model,label,cycles=cycles,members=20,years=100,first=first,seed=seed)
        ds=diagnostics.assemble(folder)
        diagnostics.compare(ds,folder,windows={"years21_60":(20,60),"years61_100":(60,100),"years81_100":(80,100)})
        diagnostics.plot(ds,folder,window=(60,100))
        diagnostics.convergence(ds,folder)
        diagnostics.bulk_latents(folder,{"years21_60":(20,60),"years61_100":(60,100)})
    diagnostics.pool_confirmation([experiment.OUTPUT/"confirm_january",experiment.OUTPUT/"confirm_july"],
                                  experiment.OUTPUT/"confirmation_pooled")


if __name__=="__main__":
    main()
