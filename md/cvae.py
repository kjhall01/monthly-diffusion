
import torch.nn as nn 
import torch 
from .data_utils import intermediate_shapes, gaussian_crps
from .encoder import VariationalEncoder 
from .decoder import Decoder 
from .preprocessing import undo_preprocess_by_variables
from .diffusion import Diffusion
import pprint 
from torch.utils.data import DataLoader, TensorDataset
import numpy as np 
from torch.distributions import kl_divergence
from pathlib import Path 
import gc
from .saveload import save_model
import pandas as pd 
from pandas.tseries.offsets import DateOffset
from pathlib import Path 
from .saveload import save_model
import math
import uuid 


## seasonality embedding 
class MonthlySpatialTimeEmbedding(nn.Module):
    def __init__(self, C, H, W, basis=8, hidden=64):
        super().__init__()
        self.H, self.W, self.C = H, W, C

        # Learned low-dimensional time embedding
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, basis)  # coefficients for spatial basis
        )

        # Learnable spatial basis: (basis, C, H, W)
        self.spatial_basis = nn.Parameter(
            torch.randn(basis, basis, H, W) * 0.02
        )

    def forward(self, month):
        m = month.float()
        sincos = torch.hstack([
            torch.sin(2 * math.pi * m / 12),
            torch.cos(2 * math.pi * m / 12)
        ])
        coeffs = self.mlp(sincos)      # (basis,)
      #  print(sincos.shape, coeffs.shape) 
        # Weighted sum of basis fields
        emb = torch.einsum('be, echw -> bchw', coeffs, self.spatial_basis)
        return emb

#return 1, 5e-3, 0.5

    
class CVAE(nn.Module):
    """
        Nearly the simplest feed-forward neural network you can imagine.
    """
    def __init__(self, 
        forcing_dim=0, 
        data_dimensions=(0,0,0) , # this is CHW
        activation=nn.GELU, 
        device=torch.device('mps'),
        kl_weight = 5e-3,
        recon_weight = 1,
        prediction_weight = 0.5,
        n_statics = 8,
        statics = None,
        hidden_channels=[8],
        squeeze_factors = [ 2],
        variable_names = None,
        seasonality_dim = 8,

        encoder_sfno_embed_dim = 128,
        encoder_rank = 32,
        encoder_conditioning_rank = 4,
        encoder_conditioning_operator_type = 'spectral_fc',
    
        decoder_sfno_embed_dim = 128,
        decoder_rank = 32,
        decoder_conditioning_rank = 4,
        decoder_conditioning_operator_type = 'spectral_fc',

        diffusion_sfno_embed_dim = 128,
        diffusion_rank = 32,
        diffusion_conditioning_rank = 4,
        diffusion_conditioning_operator_type = 'spectral_fc',

    ):
        super(CVAE, self).__init__()

        self.forcing_dim = forcing_dim
        self.data_dimensions = data_dimensions
        self.activation = activation
        self.kl_weight = kl_weight
        self.recon_weight = recon_weight
        self.prediction_weight = prediction_weight

        self.n_statics = n_statics 
        self.statics = statics.to(device)
        self.hidden_channels = hidden_channels
        self.squeeze_factors = squeeze_factors
        self.variable_names = variable_names
        self.seasonality_dim = seasonality_dim

        self.encoder_sfno_embed_dim = encoder_sfno_embed_dim
        self.encoder_rank = encoder_rank
        self.encoder_conditioning_rank = encoder_conditioning_rank
        self.encoder_conditioning_operator_type = encoder_conditioning_operator_type
    
        self.decoder_sfno_embed_dim = decoder_sfno_embed_dim
        self.decoder_rank = decoder_rank
        self.decoder_conditioning_rank = decoder_conditioning_rank
        self.decoder_conditioning_operator_type = decoder_conditioning_operator_type

        self.diffusion_sfno_embed_dim = diffusion_sfno_embed_dim
        self.diffusion_rank = diffusion_rank
        self.diffusion_conditioning_rank = diffusion_conditioning_rank 
        self.diffusion_conditioning_operator_type = diffusion_conditioning_operator_type

        intermediate_sizes = intermediate_shapes( 
            data_dimensions = self.data_dimensions,
            hidden_layers = self.hidden_channels,
            squeeze_factors = self.squeeze_factors,
        )
        self.latent_shape = intermediate_sizes[0]

        print('INTERMEDIATE COMPRESSION RATIOS:')
        print([intermediate_sizes[i][0]*intermediate_sizes[i][1] * intermediate_sizes[i][2] / (data_dimensions[0] * data_dimensions[1] * data_dimensions[2]) for i in range(len(intermediate_sizes))])

        print('N FORCING CHANNELS: ', forcing_dim) 
        print('N SEASONALITY CHANNELS: ', seasonality_dim)

        self.encoder = VariationalEncoder(
            intermediate_sizes, 
            forcing_dim + seasonality_dim, 
            n_statics = self.n_statics,
            activation = activation,
            varnames= self.variable_names,
            statics = self.statics,
            rank = self.encoder_rank, 
            conditioning_rank = self.encoder_conditioning_rank,
            conditioning_operator_type = self.encoder_conditioning_operator_type,
            sfno_embed_dim=self.encoder_sfno_embed_dim
        )

        self.decoder = Decoder(
            intermediate_sizes, 
            forcing_dim + seasonality_dim, 
            activation = activation,
            sfno_embed_dim = self.decoder_sfno_embed_dim,
            n_statics = self.n_statics,
            statics = self.encoder.statics,
            rank = self.decoder_rank,
            conditioning_rank = self.decoder_conditioning_rank,
            conditioning_operator_type = self.decoder_conditioning_operator_type
        )

        self.predictor = Diffusion(
            [ intermediate_sizes[0] for _ in range(1) ], 
            forcing_dim + seasonality_dim, 
            n_statics=0,
            activation = activation,
            sfno_embed_dim=self.diffusion_sfno_embed_dim,
            device = device,
            rank = self.diffusion_rank,
            conditioning_rank = self.diffusion_conditioning_rank,
            conditioning_operator_type = self.diffusion_conditioning_operator_type
        )

        print("LATENT SPACE DIMENSIONS: ", intermediate_sizes[0])
        self.seasonality_embedding = MonthlySpatialTimeEmbedding(
            *self.data_dimensions, 
            basis=self.seasonality_dim
        )

        self.device = device 
        self.to(self.device)

    def reparametrize(self, mu, logvar, eps=None):
        std = torch.exp(0.5 * logvar)
        if eps is None:
            eps = torch.randn_like(std).to(mu.device).detach()
        elif eps == 'gauss':
            eps = gen_legendre_gauss_noise(std, self.predictor.first_layer.condition_resample)
        else:
            eps = eps 
        return mu + eps * std

    def compute_loss(self, f_t0, x_t0, m_t0, f_t1, x_t1, m_t1, denoise=False, validation=False):
        if validation:
            assert not torch.is_grad_enabled(), "Compute_loss running in grad mode!"

        loss_terms = {} 

        # embed seasonality 
        m_t0 = self.seasonality_embedding(m_t0) 
        f_t0 = torch.cat([f_t0, m_t0], dim=1) 

        m_t1 = self.seasonality_embedding(m_t1) 
        f_t1 = torch.cat([f_t1, m_t1], dim=1) 

        # encode
        mu_prime_t0, lv_prime_t0, cs = self.encoder(x_t0, f_t0)
        z_t0 = self.reparametrize(mu_prime_t0, lv_prime_t0)

        kldiv_loss =( -0.5 * (1 + lv_prime_t0 - mu_prime_t0.pow(2) - lv_prime_t0.exp()) ).mean()
        loss_terms['KL Divergence'] = kldiv_loss

        # compute reconstruction loss  
        xhat_t0 = self.decoder(z_t0, cs)
        reconstruction_loss = torch.nn.functional.mse_loss(xhat_t0, x_t0).mean()
        loss_terms['Reconstruction'] = reconstruction_loss 

        if validation:
            # to track how the model is learning, calculate loss w/ scrambled forcings vs w/ scrambled latents
            f_t0_random = torch.randn_like(f_t0).to(f_t0.device) 

            # scrambled forcings 
            mu_prime_t0_frandom, lv_prime_t0_frandom, cs_random_f = self.encoder(x_t0, f_t0_random)
            reconstruction_with_scrambled_forcings = self.decoder(z_t0, cs_random_f)
            recon_loss_with_scrambled_forcings = torch.nn.functional.mse_loss(reconstruction_with_scrambled_forcings, x_t0).mean() 
            loss_terms['Reconstruction (randomized forcing)'] = recon_loss_with_scrambled_forcings

            # scrambled latents
            reconstruction_with_scrambled_latents = self.decoder(torch.randn_like(z_t0).to(z_t0.device), cs)
            recon_loss_with_scrambled_latents = torch.nn.functional.mse_loss(reconstruction_with_scrambled_latents, x_t0).mean() 
            loss_terms['Reconstruction (randomized latents)'] = recon_loss_with_scrambled_latents

            # persistence loss 
            persistence_mse = torch.nn.functional.mse_loss(x_t0, x_t1).mean()
            loss_terms['Persistence MSE'] = persistence_mse 

        mu_prime_t1, lv_prime_t1, cs_t1 = self.encoder(x_t1, f_t1)
        noised_samples, rand_noise, t = self.predictor.noise_data(mu_prime_t1)

        pred_score = self.predictor.score_net(z_t0, noised_samples, cs[-1], t)
        denoising_loss = torch.nn.functional.mse_loss(pred_score, rand_noise).mean()
        loss_terms['Denoising Loss'] = denoising_loss 

        diffusion_mean_loss = torch.nn.functional.mse_loss( self.predictor.latent_mu.squeeze(), mu_prime_t0.mean()).mean() 
        diffusion_lv_loss = torch.nn.functional.mse_loss( torch.exp(self.predictor.latent_log_sigma.squeeze()), mu_prime_t0.std()).mean()

        loss_terms['Latent Population Mean Loss'] = diffusion_mean_loss 
        loss_terms['Latent Population LgVar Loss'] = diffusion_lv_loss 

        if denoise:
            with torch.no_grad():
                predictions = self.predictor.sample(
                    mu_prime_t0,
                    cs[-1]
                ).detach()
                prediction_loss = torch.nn.functional.mse_loss(predictions, mu_prime_t1 ).mean()
                loss_terms['Diffusion Sampling MSE in Latent Space'] = prediction_loss

                decoded_diffusion_samples = self.decoder(predictions, cs_t1)
                decoded_diffusion_sample_loss = torch.nn.functional.mse_loss(decoded_diffusion_samples, x_t1).mean()
                loss_terms['Diffusion Sampling MSE in Physical Space'] = decoded_diffusion_sample_loss

        return loss_terms

    def train_model(self,
            training_data,
            validation_data,
            lr=1e-4,
            num_epochs=300,
            batch_size=16,
            do_diffusion_every_n=20
        ):

        xtrain_t0 = training_data['x']
        xtrain_t1 = training_data['y']
        xtrain_forcing_t0 = training_data['x_forcing']
        xtrain_forcing_t1 = training_data['y_forcing']
        xtrain_months_t0 = training_data['x_months']
        xtrain_months_t1 = training_data['y_months']

        xval_t0 = validation_data['x']
        xval_t1 = validation_data['y']
        xval_forcing_t0 = validation_data['x_forcing']
        xval_forcing_t1 = validation_data['y_forcing']
        xval_months_t0 = validation_data['x_months']
        xval_months_t1 = validation_data['y_months']

        train_dataset = TensorDataset(
            torch.tensor(xtrain_forcing_t0, dtype=torch.float32).to(self.device), 
            torch.tensor(xtrain_t0, dtype=torch.float32).to(self.device),
            torch.tensor(xtrain_months_t0, dtype=torch.float32).to(self.device),
            torch.tensor(xtrain_forcing_t1, dtype=torch.float32).to(self.device), 
            torch.tensor(xtrain_t1, dtype=torch.float32).to(self.device),
            torch.tensor(xtrain_months_t1, dtype=torch.float32).to(self.device)
        )
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(
            torch.tensor(xval_forcing_t0, dtype=torch.float32).to(self.device), 
            torch.tensor(xval_t0, dtype=torch.float32).to(self.device),
            torch.tensor(xval_months_t0, dtype=torch.float32).to(self.device),
            torch.tensor(xval_forcing_t1, dtype=torch.float32).to(self.device), 
            torch.tensor(xval_t1, dtype=torch.float32).to(self.device),
            torch.tensor(xval_months_t1, dtype=torch.float32).to(self.device)
        )
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)

        print(f"Encoder trainable parameters: {sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)}")
        print(f"Decoder trainable parameters: {sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)}")
        print(f"Predictor trainable parameters: {sum(p.numel() for p in self.predictor.parameters() if p.requires_grad)}")
        print(f"Seasonality trainable parameters: {sum(p.numel() for p in self.seasonality_embedding.parameters() if p.requires_grad)}")

        # checkpointing 
        best_recon_pred, best_pred_pred = 999, 999 
        best_recon_recon, best_pred_recon = 999, 999 
        best_recon_same, best_pred_same = 999, 999 

        this_session = str(uuid.uuid4())
        print(f"This session id: {this_session}")

        for epoch in range(num_epochs):
            do_denoising = epoch % do_diffusion_every_n == 2 

            self.train()

            loss_tracking = {
                'KL Divergence': 0,
                'Reconstruction': 0,
                'Denoising Loss': 0,
                'Latent Population Mean Loss': 0,
                'Latent Population LgVar Loss': 0
            }
            sz = 0 

            if do_denoising:
                loss_tracking['Diffusion Sampling MSE in Latent Space'] = 0
                loss_tracking['Diffusion Sampling MSE in Physical Space'] = 0


            for t_batch_t0, x_batch_t0, m_batch_t0, t_batch_t1, x_batch_t1, m_batch_t1 in train_dataloader:
                optimizer.zero_grad()

                loss_terms = self.compute_loss(
                    t_batch_t0, x_batch_t0, m_batch_t0, t_batch_t1, x_batch_t1, m_batch_t0, 
                    denoise=do_denoising,
                    validation=False 
                )


                for term in loss_terms.keys():
                    loss_tracking[term] += loss_terms[term].item() * x_batch_t0.size(0)

                loss = loss_terms['Reconstruction'] * self.recon_weight + \
                    loss_terms['KL Divergence'] * self.kl_weight + \
                    loss_terms['Denoising Loss'] * self.prediction_weight + \
                    loss_terms['Latent Population Mean Loss'] * self.prediction_weight + \
                    loss_terms['Latent Population LgVar Loss'] * self.prediction_weight 

                loss.backward()

                sz += x_batch_t0.size(0)
                optimizer.step()

            for term in loss_tracking.keys():
                loss_tracking[term] = loss_tracking[term] / sz 

            self.eval()
            val_loss_tracking = {
                'KL Divergence': 0,
                'Reconstruction': 0,
                'Denoising Loss': 0,
                'Latent Population Mean Loss': 0,
                'Latent Population LgVar Loss': 0,
                'Reconstruction (randomized forcing)': 0,
                'Reconstruction (randomized latents)': 0,
                'Persistence MSE': 0
            }

            if do_denoising:
                val_loss_tracking['Diffusion Sampling MSE in Latent Space'] = 0
                val_loss_tracking['Diffusion Sampling MSE in Physical Space'] = 0

            sz = 0 
            with torch.no_grad():
                for t_val_t0, x_val_t0, m_val_t0, t_val_t1, x_val_t1, m_val_t1 in val_dataloader:

                    loss_terms = self.compute_loss(
                        t_val_t0, x_val_t0, m_val_t0, t_val_t1, x_val_t1, m_val_t1,
                        denoise=do_denoising, 
                        validation=True
                    )

                    for term in loss_terms.keys():
                        val_loss_tracking[term] += loss_terms[term].item() * x_val_t0.size(0)

                    sz += x_val_t0.size(0)


            for term in loss_terms.keys():
                val_loss_tracking[term] = val_loss_tracking[term] / sz 

        
            if val_loss_tracking['Reconstruction'] < best_recon_same and val_loss_tracking['Denoising Loss'] < best_pred_same:
                best_recon_same = val_loss_tracking['Reconstruction']
                best_pred_same = val_loss_tracking['Denoising Loss']
                save_model(self, f'.best_overall_checkpoint{this_session}.pth')

            if val_loss_tracking['Reconstruction'] <best_recon_recon:
                best_recon_recon = val_loss_tracking['Reconstruction']
                best_pred_recon = val_loss_tracking['Denoising Loss']
                save_model(self, f'.best_recon_checkpoint{this_session}.pth')

            if val_loss_tracking['Denoising Loss'] < best_pred_pred:
                best_pred_pred = val_loss_tracking['Denoising Loss']
                best_recon_pred = val_loss_tracking['Reconstruction']
                save_model(self, f'.best_pred_checkpoint{this_session}.pth')


            print(f"---------------- EPOCH: {epoch} ------------------------")
            print("                 Train         Val")
            for key in val_loss_tracking.keys():
                if key in list(loss_tracking.keys()):
                    print(f"  {key}:  {loss_tracking[key]:>0.4f}  {val_loss_tracking[key]:>0.4f}") 
                else:
                    print(f"    {key}:  {val_loss_tracking[key]:>0.4f}") 

            print(f"Best Overall Model - recon: {best_recon_same}  pred: {best_pred_same}") 
            print(f"Best Recon Model - recon: {best_recon_recon}  pred: {best_pred_recon}") 
            print(f"Best Pred Model - recon: {best_recon_pred}  pred: {best_pred_pred}") 
            print()
        return f'.best_overall_checkpoint{this_session}.pth', f'.best_recon_checkpoint{this_session}.pth', f'.best_pred_checkpoint{this_session}.pth'


    def forced_run(self, 
           ic, 
           forcings, 
           months, 
           ensemble_size=10, 
           output_template=None, 
           output_files=Path('rollout'), 
           use_noise=True, 
           dict_of_params=None, 
           mask=None, 
           group_levels=True
        ):

        output_files = Path(output_files)
        if not output_files.is_dir():
            output_files.mkdir(exist_ok=True, parents=True)

        self.eval()
        with torch.no_grad():
            static_embed = self.statics

            forcings = torch.tensor(forcings, dtype=torch.float32).to(self.device)
            months = torch.tensor(months, dtype=torch.float32).to(self.device)
            ic = torch.tensor(ic, dtype=torch.float32).to(self.device)

            physical_state = torch.vstack([ic for _ in range(ensemble_size)]).to(self.device)
            forcing_step = torch.vstack([forcings[0,...].unsqueeze(0) for _ in range(ensemble_size)])

            month_step = self.seasonality_embedding(months[0,...].unsqueeze(0))
            month_step = torch.vstack([ month_step for _ in range(ensemble_size)])
            forcing_step2 = torch.cat([forcing_step, month_step], dim= 1)

            mu_state, logsigma2_state, cs = self.encoder(physical_state, forcing_step2)

            decoded_physical_states = self.decoder(mu_state, cs).cpu().detach().numpy() 
            decoded_physical_states = output_template.isel(time=slice(None, ensemble_size)).copy(data=decoded_physical_states)
            decoded_physical_states = decoded_physical_states.rename({'time': 'member'})
            decoded_physical_states = decoded_physical_states.assign_coords({'member': np.arange(ensemble_size) + 1})
            decoded_physical_states = decoded_physical_states.expand_dims('time')
            decoded_physical_states = decoded_physical_states.assign_coords({'time': [pd.Timestamp(output_template.time.values[0])]})

            decoded_physical_states = decoded_physical_states * mask
            decoded_physical_states = undo_preprocess_by_variables(
                decoded_physical_states, 
                dict_to_undo = dict_of_params,
                group_levels = group_levels
            )
            decoded_physical_states.to_netcdf(output_files / f'ensemble_rollout_step0.nc')

            for step in range(1, forcings.shape[0]):
                print('step: ', step)

                mu_state = self.predictor(mu_state, cs[-1])
                forcing_step =  (torch.ones_like(forcing_step) * forcings[step, ...].unsqueeze(0)).to(self.device) 
                month_step = self.seasonality_embedding(months[step,...].unsqueeze(0))
                month_step = torch.vstack([ month_step for _ in range(ensemble_size)])
                forcing_step2 = torch.cat([forcing_step, month_step], dim= 1)
                cs = self.encoder.embed_conditions(forcing_step2)

                decoded_physical_states = self.decoder(mu_state, cs).cpu().detach().numpy() 
                decoded_physical_states = output_template.isel(time=slice(None, ensemble_size)).copy(data=decoded_physical_states)
                decoded_physical_states = decoded_physical_states.rename({'time': 'member'}).assign_coords({'member': np.arange(ensemble_size) + 1})
                decoded_physical_states = decoded_physical_states.expand_dims('time').assign_coords({'time': [pd.Timestamp(output_template.time.values[0]) + DateOffset(months=step)]})
                decoded_physical_states = decoded_physical_states * mask
                decoded_physical_states = undo_preprocess_by_variables(
                    decoded_physical_states, 
                    dict_to_undo=dict_of_params,
                    group_levels= group_levels

                )

                sst = decoded_physical_states.sel(varlev='SKTSFC')
                weights = np.cos(np.deg2rad(sst.lat))
                weights.name = "weights"
                sst = sst.weighted(weights).mean(['lat', 'lon'])
                print(sst)
                decoded_physical_states.to_netcdf(output_files / f'ensemble_rollout_step{step}.nc')

            print(f'saved forced run results to {output_files}!')
            return None 