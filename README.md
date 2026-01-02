# Monthly Diffusion: Toward AI Emulation of Future Climates
####  Kyle Hall (kylehall@umd.edu); Maria Molina
#### UMD Pareto Group (https://mariajmolina.github.io/) 

Monthly Diffusion v0.0.1 (MD1) implements v-prediction [1] type conditional diffusion in the latent space of a conditional variational autoencoder [2, 3, 4, 5]. The architecture of the Encoder/Decoder/Denoiser networks are based on Spherical Fourier Neural Operators [6]. They use spatial conditional RMS norms (akin to FiLM [7] or other conditional normalizations) and low-rank tensor operators (akin to fully-connected linear layers in SHT space) to explicitly allow cross-wavenumber "energy cascades". 

MD1 is an initial version of a model architecture which may subsequently be used in Kyle Hall's doctoral dissertation to study internal variability and the forced response of the earth system to anthropogenic forcings (which were not used for AIMIP). MD1 operates on a monthly-averaged atmospheric state vector derived from ERA5 monthly mean fields (derived = regridded), directly targetting long-timescale modeling and slow climate processes, and dramatically decreasing the computational expense of model training and long autoregressive climate runs. I have been able to run the training code at 1.5 degree nominal horizontal resolution on an Apple M3 with 8GB RAM (but not for long enough to converge). We trained on Derecho A1000 GPUs for the AIMIP run- on a single A1000, it takes about 2 hours wall-clock time for the model to converge. On M3, maybe over the weekend? ~30min/epoch, but that's still unproven. The 46.25 year autoregressive runs are of trivial cost ~ minutes.

### Architecture 
In our parlance, the encoder parametrizes (under parameters $\theta$) the bayesian posterior distribution $p_\theta(z | x, c)$ where $x$ is the atmospheric state, and $c$ is a conditioning tensor. It is modeled as a multivariate gaussian with a diagonal covariance matrix, with parameters $\mu$ and $\sigma$. A random sample of the posterior is generated with the 'reparametrization trick' $z = \mu + \epsilon * \sigma$ where $\epsilon \in \mathcal{N}(0, 1)$, as in most other VAE architectures. We push $p_\theta$ towards an isotropic multivariate gaussian prior $p(z)$ by implementing the standard Kulback-Liebler Divergence loss term, $D_{kl}(p_\theta | \mathcal{N}(0, 1) )$. The decoder then parametrizes (under parameters $\phi$) the likelihood distribution, $q_\phi(x | z, c)$. The compression ratio of the latent is a hyperparameter to be tuned.

The denoiser then performs conditional denoising through v-prediction, in order to predict the mean of the latent distribution at time $t+1$, given the state of the latent at time $t$, and the conditions at time $t$, parametrizing under $\gamma$ the distribution $p_\gamma(\mu_{t+1} | z_t, c_t)$.  Anecdotally, this parametrization strikes a balance between overfitting and autoregressive stability. Conditioning on $z_t$ instead of $\mu_t$ effectively 'augments' the training data (which is very necessary, due to the small number of training samples), and targetting $\mu_{t+1}$ instead of $z_{t+1}$ stabilizes the autoregression (again, anecdotally) because then prediction errors (if unbiased) can be interpreted of as small random samples of $p_\theta$  at time $t+1$, which our network is already trained to see. 

Our goal with this conditional VAE architecture is to free the VAE latent space to focus on capturing latent factors unassociated with the conditioning tensor $c$, which is in essence a skip-connection. In the case of the AIMIP protocol runs, that would mean that the latent space should capture variability unassociated with SST conditions, or "atmospheric" modes of variability. These "atmospheric" modes of variability are the true state vector, in that they are the fields that evolve autoregressively (as opposed to being specified by the forcing values). The conditioning tensor $c$ is provided to the denoiser in order to account for potential impacts of the forcing states on the *evolution* of the latent "atmospheric" modes. That being said, we have not yet thoroughly studied the degree to which MD1's latent space is disentangled from the conditioning tensor. We can conclude that it is to some extent disentangled because, as demonstrated by the analysis notebooks, the North Atlantic Oscillation (NAO, a mode of atmospheric variability which is, ostensibly, at least less-coupled to SST than ENSO or PDO are) indices exhibited by MD1 ensemble members diverge with autoregression over time whereas the ENSO indices do not. 

### Training
We train our encoder/decoder and denoiser jointly, allowing errors from denoising to backpropagate through the encoder in hopes that the encoder will learn a smooth latent manifold which is easy to model for v-prediction. In contrast, most other latent diffusion models are trained in two stages- encoder/decoder first, then diffusion on frozen latents. Our denoiser learns to model latent population parameters - two scalars $\mu$ and $\sigma$ for standardizing the latents before diffusion and un-standardizing them afterwards. As such our loss function has the following terms: 

Symbol | Formula | Name
--- | ----| ----
$L_1$ | $E\[ (X_t - \hat{X}_t)^2\]$ | Reconstruction MSE 
$L_2$ | $E\[ (v - \hat{v})^2]$ | $v$-prediction MSE
$L_3$ | $D_{kl}(p_\theta(z_t \| x_t, c_t) , \mathcal{N}(0, 1) )$ | KL Divergence
$L_4$ | $(E\[ z_t\] - \mu_{pop})^2]$ | Latent Population Mean MSE 
$L_5$ | ($E\[ (E\[ z_t\] - z_t)^2 \] - \sigma_{pop})^2$ | Latent Population Std. Dev. MSE 

Our full weighted loss function is then:

$Loss = \lambda_1 L_1 + \lambda_2 L_2 + \lambda_3 ( L_3 + L_4 + L_5)$

Which is minimized by the AdamW optimizer in pytorch with a weight decay of 1e-3 and an initial learning rate of 1e-3. I've been using ~ $\lambda_1 = 1, \lambda_2 = 0.5, \lambda_3 = 5e-3$ but haven't done any systematic hyperparameter tuning.

Here $x$ represents the required AIMIP reporting fields: U/V/T/Q on seven pressure levels (1000, 850, 700, 500, 250, 100, and 50 hPa), Geopotential Height at 500 hPa (Z500) as well as the required 2D variables SKT, T2M, T2D, U10m, V10m, MTPR, PS. We include Mean Sea Level Pressure (MSLP) (for no particular reason). $c$ includes both the physical forcing fields provided by AIMIP (Sea Surface Temp, Sea Ice Concentration, and a Land-Sea Mask), a learned seasonality embedding based on the month of the year. Speculatively, providing this type of $c$ to the encoder / decoder networks this way should allow the latent to focus on capturing variance unassociated with the forcing terms (SST, Ice) and unassociated with a seasonal cycle, effectively targetting weakly-coupled and atmospheric modes of variability such as the North Atlantic Oscillation (NAO). 

# Conceptual Model Run Procedure

1. Encode: $p_\theta(z_t | x_t, c_t) = \mathcal{N}(\mu_t, \sigma_t^2)$
2. Sample: $z_t = \mu_t + \epsilon * \sigma_t$ where $\epsilon \in \mathcal{N}(0, 1)$
3. Conditional Denoising using a cosine-beta noise schedule to define $\bar{\alpha}, \alpha, \sigma^2, \beta$ at each of $T$ noise levels indexed by $t$
   ```
   n = N(0, 1)
   for t in 15 ... 1:
     v = denoiser(n  given {t/15, z_t0, c_t0})
     eps = sqrt(alpha_bar_t) * v + sqrt( 1 - alpha_bar_t) * n

     mu = 1/sqrt(alpha_t) * ( x - beta_t / torch.sqrt(1 - alpha_bar_t) * eps )

     # add back noise at next noise level
     if t > 1:
       n = mu + N(0, 1) * sigma^2

   z_t1 = (mu * z_pop_stddev) + z_pop_mean
   ```

   vaguely following [8, 9, 10] with the additional help of generative ai (GPT5.1, Github Copilot).

   For autoregression, repeat by swapping $\hat{z_{t+1}}$ in for $z_t$ and $c_{t+1}$ in for $c_t$. (decoding is not necessary for autoregression as time-stepping happens in latent space)

5. Decode: $q_\phi(\hat{x_{t+1}} | \hat{z_{t+1}}, c_{t+1})$ according to forcing value at time $t+1$

It is reasonable to use the forcing at time $t+1$ to condition the decoding of the predicted latent because the atmosphere in theory responds quickly to the state of the ocean. In a higher-frequency model, presumably the atmosphere "sees" the new oceanic forcings on 1 Jan, and responds to them all month long- mostly no longer responding to oceanic forcing from 1-31 Dec. This model is not meant to be a forecast model, but you could run a more true-to-form forecast by forcing the decoding with the climatological mean SST, for example. 


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

# Evaluate Test Set Reconstruction

You can evaluate test set reconstruction skill with `scripts/evaluate_test_set_skill.py`.

# References
- [1] https://arxiv.org/pdf/2202.00512
- [2] https://arxiv.org/abs/1906.02691
- [3] https://arxiv.org/abs/1804.03599
- [4] https://arxiv.org/abs/1312.6114
- [5] https://papers.nips.cc/paper_files/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html
- [6] https://arxiv.org/abs/2306.03838
- [7] https://arxiv.org/abs/1709.07871
- [8] https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
- [9] https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
- [10] https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/diffusion/ddpm.py
   

