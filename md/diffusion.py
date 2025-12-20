
import torch.nn as nn 
from .blocks import ConcatCondishBlock, JustCondishBlock
import torch 
import math 
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    """
    Standard diffusion-style timestep embedding:
      t -> sinusoidal embedding -> MLP(SiLU) -> linear(C)
      Output is broadcast to spatial shape (B, C, H, W).
    """
    def __init__(self, channels, emb_dim=256, T=30, spatial_shape=()):
        super().__init__()
        self.channels = channels

        # The usual sinusoidal embedding size (must be even)
        self.emb_dim = emb_dim

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, channels),
        )
        self.T = T 
        self.spatial_shape = spatial_shape

    def forward(self, t):
        """
        t: (B,) or scalar; timestep in [0, T)
        T: total number of diffusion steps
        spatial_shape: (B, C, H, W)
        returns: (B, C, H, W) tensor of timestep embeddings
        """
        C, H, W = self.spatial_shape
        B = t.shape[0]

        # ---- 1. Sinusoidal embedding (like in DDPM) ----
        half_dim = self.emb_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=t.device) / (half_dim - 1)
        )  # (half_dim,)
        args = t.view(-1, 1) * freqs.view(1, -1)  # (B, half_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, emb_dim)
        # ---- 2. MLP with SiLU ----
        emb = self.mlp(emb)  # (B, C)

        # ---- 3. Broadcast spatially ----
        emb = emb.view(B, self.channels, 1, 1) * torch.ones(1, 1, H, W).to(emb.device)
        return emb


def build_cosine_schedule(T, s=0.008, device="mps", dtype=torch.float32):

    # t grid from 0..T inclusive
    t = torch.linspace(0, T, T+1, device=device, dtype=dtype)

    # Cosine schedule f(t)
    def f(t):
        return torch.cos(((t / T + s) / (1 + s)) * math.pi * 0.5) ** 2

    # Normalize to make alpha_bar[0] = 1
    f0 = f(torch.zeros(1, device=device, dtype=dtype))
    alpha_bar = f(t) / f0  # shape [T+1]

    # Compute per-step alphas and betas
    # alpha_bar[t] = prod_{s=1..t} alpha[s]
    alpha_bar_next = alpha_bar[1:]           # ᾱ_t
    alpha_bar_prev = alpha_bar[:-1]          # ᾱ_{t-1}

    alpha = alpha_bar_next / alpha_bar_prev  # α_t
    beta = 1.0 - alpha                        # β_t

    # Clip only extremely small values for numerical stability
    beta = beta.clamp(min=1e-12, max=0.999)

    # Recompute alpha (in case clipping affected β)
    alpha = 1.0 - beta

    # Build alpha_bar (cumulative product)
    alpha_bar = torch.cumprod(alpha, dim=0)

    # Build shifted alpha_bar_prev with ᾱ_0 = 1.0
    alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)

    # Posterior variance for q(x_{t-1} | x_t, x_0)
    posterior_variance = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)

    return {
        "beta": beta,                                   # [T]
        "alpha": alpha,                                 # [T]
        "alpha_bar": alpha_bar,                         # [T]
        "alpha_bar_prev": alpha_bar_prev,               # [T]
        "posterior_variance": posterior_variance        # [T]
    }


class Diffusion(nn.Module):
    def __init__(self, 
            intermediate_sizes, 
            forcing_dim, 
            n_statics,
            activation = nn.GELU,
            sfno_embed_dim=32,
            T=15,
            device=torch.device('mps'),
            s = 0.008,
            time_channels = 5,
            rank = 16,
            conditioning_operator_type = "driscoll-healy",
            conditioning_hidden_channels = 4,
            conditioning_rank = 4
        ):
        super(Diffusion, self).__init__()

        # intermediate sizes is initially small-to-big
        intermediate_sizes_r = intermediate_sizes

        if len(intermediate_sizes_r) == 1: 
            intermediate_sizes_r = [intermediate_sizes_r[0] for _ in range(2)]

        in_shape, out_shape = intermediate_sizes_r[0], intermediate_sizes_r[1]

        self.time_channels = time_channels 
        self.rank = rank
        self.sfno_embed_dim = sfno_embed_dim

        self.first_layer = ConcatCondishBlock(
            in_shape,
            out_shape,
            n_conditioning_channels=in_shape[0] + forcing_dim + self.time_channels,
            n_static_channels=n_statics,
            activation=activation,
            is_sfno_block = False if self.sfno_embed_dim == 0 else True, 
            sfno_hidden_layer= self.sfno_embed_dim,
            rank = self.rank,
            conditioning_operator_type = conditioning_operator_type, 
            conditioning_hidden_channels = conditioning_hidden_channels,
            conditioning_rank = conditioning_rank
        )

        intermediate_sizes_r = intermediate_sizes_r[1:]

        layers = []
        for in_shape, out_shape in zip(intermediate_sizes_r[:-1], intermediate_sizes_r[1:]):
            layers.append(
                JustCondishBlock(
                    in_shape,
                    out_shape,
                    n_conditioning_channels=in_shape[0] +forcing_dim + self.time_channels,
                    activation=activation,
                    is_sfno_block = False if sfno_embed_dim == 0 else True, 
                    sfno_hidden_layer= self.sfno_embed_dim,
                    has_resampled_condition = True,
                    rank = self.rank,
                    conditioning_operator_type = conditioning_operator_type, 
                    conditioning_hidden_channels = conditioning_hidden_channels,
                    conditioning_rank = conditioning_rank
                )
            )

        self.layers = nn.ModuleList(layers)
        self.device = device
        self.T = T

        sched = build_cosine_schedule(self.T, device=self.device)
        self.beta = sched["beta"]
        self.alpha = sched["alpha"]
        self.alpha_bar = sched["alpha_bar"]
        self.alpha_bar_prev = sched["alpha_bar_prev"]
        self.posterior_variance = sched["posterior_variance"]

        self.latent_mu = nn.Parameter(torch.zeros(1)).to(self.device)
        self.latent_log_sigma = nn.Parameter(torch.zeros(1)).to(self.device)
        self.time_embed = TimeEmbedding(self.time_channels, emb_dim=32, T = T, spatial_shape = intermediate_sizes_r[0])

    def normalize_latent(self, z):
        return (z - self.latent_mu) / torch.exp(self.latent_log_sigma)

    def unnormalize_latent(self, z):
        return z * torch.exp(self.latent_log_sigma) + self.latent_mu

    def score_net(self, xt0, xnoise, c, t):
        xt0 = self.normalize_latent(xt0)
        t = self.time_embed(t)
        c = torch.cat([xt0, c, t], dim=1)

        xnoise, c_r = self.first_layer(xnoise, c, statics=None, c_resampled=c )
        for layer in self.layers: 
            xnoise, c_r = layer(xnoise, c, c_is_resampled=True)
        return  xnoise

    def noise_data(self, x):
        x = self.normalize_latent(x)
        t = torch.randint(1, self.T+1, (x.shape[0], ), dtype=torch.long, requires_grad=False).view(-1, 1, 1, 1).to(x.device)
        noise = torch.randn_like(x).to(x.device)

        alpha_bar_t = self.alpha_bar[t-1].view(-1, 1, 1, 1).to(x.device)
        noised_samples = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise
        v = torch.sqrt(alpha_bar_t) * noise - torch.sqrt(1 - alpha_bar_t) * x
        return noised_samples, v, (t / self.T).detach()

    @torch.no_grad()
    def sample(self, xt0, c):
        self.eval()
        with torch.no_grad():
            x = torch.randn_like(xt0).to(xt0.device)

            for t in range(self.T, 0, -1):

                t_val = torch.full((x.size(0),1), t/self.T, device=x.device)

                alpha_t      = self.alpha[t-1].view(-1,1,1,1)
                alpha_bar_t  = self.alpha_bar[t-1].view(-1,1,1,1)
                beta_t       = self.beta[t-1].view(-1,1,1,1)
                sigma2       = self.posterior_variance[t-1].view(-1,1,1,1)

                # ----- predict v -----
                v_pred = self.score_net(xt0, x, c, t_val)

                # ----- convert v -> eps -----
                eps_pred = torch.sqrt(alpha_bar_t) * v_pred + torch.sqrt(1 - alpha_bar_t) * x

                # ----- DDPM posterior mean -----
                mu = (1/torch.sqrt(alpha_t)) * (
                    x - beta_t / torch.sqrt(1 - alpha_bar_t) * eps_pred
                )

                if t > 1:
                    z = torch.randn_like(x).to(x.device)
                    x = mu + torch.sqrt(sigma2) * z 
                else:
                    x = mu

            return self.unnormalize_latent(x)



    def forward(self, xt0, c):
        return self.sample(xt0, c)
