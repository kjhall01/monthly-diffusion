"""Export an existing trained model with training-only initialization moments.

No network is retrained. The source checkpoint is never overwritten. A future
model.train() fits and saves these same statistics automatically.
"""
import argparse
import logging
from pathlib import Path
import torch
import yaml
from monthly_diffusion import MonthlyDiffusionV1
from monthly_diffusion.data.sources import open_canonical_state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint",type=Path)
    p.add_argument("data_yaml",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--threads",type=int,default=4)
    args=p.parse_args()
    if args.output.exists() or args.output.resolve()==args.checkpoint.resolve():
        raise FileExistsError("Choose a new export filename; existing checkpoints are preserved")
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(message)s")
    torch.set_num_threads(args.threads)
    model=MonthlyDiffusionV1.load(args.checkpoint,device="cpu")
    before=model.weight_digest()
    source=yaml.safe_load(args.data_yaml.read_text())
    a,o,_=open_canonical_state(source["state_zarr"],expected_attrs=source.get("expected_attrs"))
    model.fit_stochastic_initialization(a,o)
    assert model.weight_digest()==before
    model.save(args.output)
    a.close(); o.close()
    logging.info("Saved %s; unchanged weights %s",args.output,before)


if __name__=="__main__":main()
