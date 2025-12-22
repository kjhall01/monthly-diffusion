# monthly-diffusion
latent diffusion for forced monthly climate prediction 

# Install
The commands listed below, in order, should set up the python environment you need to train and run monthly-diffusion models. If you are on NCAR HPC (Derecho) or otherwise using CUDA, substitute `cuda_training_environment.yaml` for `mps_environment.yaml`.  Core dependencies for model training/running are `pytorch torch-harmonics xarray pandas scipy` and for the cmor-ization script additionally `xesmf metpy`. Also whatever GPU backend you wnat to use. 

```
git clone https://github.com/kjhall01/monthly-diffusion.git
cd monthly-diffusion
conda env create -f mps_environment.yaml
conda activate main
pip install -e .
```

# Set Up
Point the training script to the directory where you have your training data / forcing files / static files. On Glade storage, you can use `/glade/work/khall/ERA5/AIMIP-Data/`, and this should be the default for the library implemented here, but you'll need to edit `scripts/train_conditional_pvae_on_era5.py` to reflect your data location. The files were created on Derecho with the notebooks listed in "monthly-diffusion/data-pipeline" if you want to see that. 

Edit the name of your model (`*.pth` pattern - it is saved as a directory with multiple files) and the model/training hyperparameters in `train_conditional_pvae_on_era5.py` to reflect whatever you want them to, then train the model via: 

```
cd scripts
python train_conditional_pvae_on_era5.py
```

This should set your model training. On Derecho, you can verify GPU usage by running the following command in another terminal: 

```
watch -n0.1 nvidia-smi
```

# Perform AIMIP Runs
Run a trained model with `scripts/run_aimp_runs.py`. You'll need to update the name of the model to reflect the one you've trained. This script will run a 46.25 autoregressive SST-forced run using ERA5 forcings created for AIMIP (not by me), then It will at 2K to the SST forcing field, do the same, then +4K to the SST field, then do the same. The Results will be saved in `f"scripts/observed_forcings-{model_path[:-4]}"` which includes the name of your model (everythiup other than '.pth').  Forced run results will be in `f"scripts/p2k_forcings-{model_path[:-4]}"` or `f"scripts/p4k_forcings-{model_path[:-4]}"` respectively.

# CMORize and Check Results
Since AIMIP requires data to be in a CMOR-compliant format, we need to CMOR-ize the model output (regrid to a CMOR-compliant grid, since ours is ... not compliant with much). You can accomplish this with the `scripts/cmorize*.py` scripts. (they are copies with the path names changed so I could run cmorization in parallel as a lazy person). These files rely on templates provided by those hosting the AIMIP data, which can be found on glade storage at `/glade/work/khall/ERA5/AIMIP-Data/templates/*`. 

Once CMOR-ization has completed  you can use the `scripts/analyze_cmorized.ipynb` notebook to evaluate your model output in some basic ways (you'll need to change the paths to reflect your model name / current date). You could also directly analyze the model output un-cmorized with some of the other legacy scripts in `scripts`, if you wanted, but I don't really use those anymore and can't guarantee anything. 





