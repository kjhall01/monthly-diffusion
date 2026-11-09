import math

import torch
from torch import nn


class DiTBlock(nn.Module):
    """Self-attention, separate component contexts, then channel mixing."""

    def __init__(self, width, n_heads, ffn_ratio=1.0):
        super().__init__()

        self.self_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(
            width, n_heads, batch_first=True,
        )

        self.query_norm = nn.LayerNorm(width)
        self.atmos_key_norm = nn.LayerNorm(width)
        self.ocean_key_norm = nn.LayerNorm(width)

        self.atmos_attention = nn.MultiheadAttention(
            width, n_heads, batch_first=True,
        )
        self.ocean_attention = nn.MultiheadAttention(
            width, n_heads, batch_first=True,
        )

        hidden_width = max(1, round(width * ffn_ratio))
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, width),
        )

        # Scale and shift for self-attention and MLP inputs.
        self.modulation = nn.Linear(width, 4 * width)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, h, atmos_context, ocean_context, condition):
        # h:             [B, N_target, width]
        # atmos_context: [B, N_atmos, width]
        # ocean_context: [B, N_ocean, width]
        # condition:     [B, width]

        modulation = self.modulation(condition).unsqueeze(1)
        attention_scale, attention_shift, mlp_scale, mlp_shift = (
            modulation.chunk(4, dim=-1)
        )

        # Candidate-state self-attention.
        x = self.self_norm(h)
        x = x * (1 + attention_scale) + attention_shift
        update, _ = self.self_attention(
            x, x, x, need_weights=False,
        )
        h = h + update

        # Both cross-attentions read the same candidate query.
        query = self.query_norm(h)

        atmos_update, _ = self.atmos_attention(
            query=query,
            key=self.atmos_key_norm(atmos_context),
            value=atmos_context,  # Unnormalized history values.
            need_weights=False,
        )
        ocean_update, _ = self.ocean_attention(
            query=query,
            key=self.ocean_key_norm(ocean_context),
            value=ocean_context,  # Unnormalized history values.
            need_weights=False,
        )
        h = h + atmos_update + ocean_update

        # Nonlinear channel mixing independently at each node.
        x = self.mlp_norm(h)
        x = x * (1 + mlp_scale) + mlp_shift
        return h + self.mlp(x)


class SphericalDiT(nn.Module):
    """One component's denoiser; no noise schedule or sampler lives here."""

    def __init__(
        self,
        component,
        atmos_latent_channels,
        ocean_latent_channels,
        n_atmos_nodes=162,
        n_ocean_nodes=162,
        width=128,
        depth=4,
        n_heads=4,
        ffn_ratio=1.0,
        time_embedding_channels=64,
    ):
        super().__init__()

        if component not in ("atmosphere", "ocean"):
            raise ValueError("component must be 'atmosphere' or 'ocean'")
        if width % n_heads:
            raise ValueError("width must be divisible by n_heads")
        if time_embedding_channels < 2 or time_embedding_channels % 2:
            raise ValueError("time_embedding_channels must be positive and even")

        self.component = component

        a = atmos_latent_channels
        o = ocean_latent_channels

        # Each history contains two months.
        # The OTHER component also supplies its current noisy candidate.
        atmos_context_channels = 2 * a + (a if component == "ocean" else 0)
        ocean_context_channels = 2 * o + (o if component == "atmosphere" else 0)

        own_channels = a if component == "atmosphere" else o

        self.input_projection = nn.Linear(own_channels, width)
        self.atmos_projection = nn.Linear(atmos_context_channels, width)
        self.ocean_projection = nn.Linear(ocean_context_channels, width)

        # Fixed node ordering is required across calls.
        self.atmos_position = nn.Parameter(
            0.02 * torch.randn(1, n_atmos_nodes, width)
        )
        self.ocean_position = nn.Parameter(
            0.02 * torch.randn(1, n_ocean_nodes, width)
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_channels, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.month_mlp = nn.Sequential(
            nn.Linear(2, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.co2_mlp = nn.Sequential(
            nn.Linear(1, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.null_co2 = nn.Parameter(torch.zeros(1, width))

        half = time_embedding_channels // 2
        frequencies = torch.exp(
            -math.log(10_000)
            * torch.arange(half, dtype=torch.float32)
            / max(half - 1, 1)
        )
        self.register_buffer("time_frequencies", frequencies)

        self.blocks = nn.ModuleList([
            DiTBlock(width, n_heads, ffn_ratio)
            for _ in range(depth)
        ])

        # Preserve an unnormalized route to the output.
        self.output_projection = nn.Linear(width, own_channels)

        # Indicates that the other component's target-month candidate is absent.
        self.missing_candidate = nn.Parameter(torch.zeros(1, 1, width))

    def forward(
        self,
        noisy_atmos,
        noisy_ocean,
        atmos_history,
        ocean_history,
        timestep,
        month,
        co2=None,
        drop_co2=None,
        drop_other_candidate=None,  # [B], boolean
    ):
        """
        noisy_atmos:   [B, N_atmos, C_atmos]
        noisy_ocean:   [B, N_ocean, C_ocean]
        atmos_history: [B, N_atmos, 2*C_atmos], current then previous
        ocean_history: [B, N_ocean, 2*C_ocean], current then previous
        timestep:      [B], raw diffusion indices 0..999
        month:         [B], target calendar month 1..12
        co2:           [B, 1], standardized CO2, or None
        drop_co2:      [B, 1], optional boolean withholding mask
        """
        batch_size = noisy_atmos.shape[0]

        if drop_other_candidate is None:
            drop_other_candidate = torch.zeros(
                batch_size, dtype=torch.bool, device=noisy_atmos.device,
            )

        missing = drop_other_candidate[:, None, None]  # [B, 1, 1]

        if self.component == "atmosphere":
            own_state = noisy_atmos
            own_position = self.atmos_position

            # Zero only the unavailable candidate, never the history.
            ocean_candidate = torch.where(missing, 0.0, noisy_ocean)

            atmos_context = self.atmos_projection(atmos_history)
            ocean_context = self.ocean_projection(
                torch.cat([ocean_history, ocean_candidate], dim=-1)
            )
            ocean_context = (
                ocean_context + missing.to(ocean_context.dtype) * self.missing_candidate
            )

        else:
            own_state = noisy_ocean
            own_position = self.ocean_position

            atmos_candidate = torch.where(missing, 0.0, noisy_atmos)

            atmos_context = self.atmos_projection(
                torch.cat([atmos_history, atmos_candidate], dim=-1)
            )
            ocean_context = self.ocean_projection(ocean_history)
            atmos_context = (
                atmos_context + missing.to(atmos_context.dtype) * self.missing_candidate
            )

        h = self.input_projection(own_state) + own_position
        atmos_context = atmos_context + self.atmos_position
        ocean_context = ocean_context + self.ocean_position

        # Raw integer diffusion index -> sinusoidal features.
        phase = timestep[:, None].float() * self.time_frequencies[None, :]
        time_features = torch.cat(
            [phase.sin(), phase.cos()], dim=-1,
        ).to(h.dtype)

        # Only target month is needed for consecutive monthly histories.
        angle = 2 * math.pi * (month.reshape(-1, 1).float() - 1) / 12
        month_features = torch.cat(
            [angle.cos(), angle.sin()], dim=-1,
        ).to(h.dtype)



        batch_size = h.shape[0]
        if co2 is None:
            co2_embedding = self.null_co2.expand(batch_size, -1)
        else:
            co2_embedding = self.co2_mlp(
                co2.reshape(batch_size, 1).to(h.dtype)
            )
            if drop_co2 is not None:
                co2_embedding = torch.where(
                    drop_co2, self.null_co2, co2_embedding,
                )

        condition = (
            self.time_mlp(time_features)
            + self.month_mlp(month_features)
            + co2_embedding
        )

        for block in self.blocks:
            h = block(h, atmos_context, ocean_context, condition)

        # Predict diffusion velocity, with the same shape as own_state.
        return self.output_projection(h)



class CoupledDiffusion(nn.Module):
    """
    Three coupling modes:
        0 / "fully_coupled"
        1 / "prescribed_ocean"
        2 / "prescribed_atmosphere"

    Histories are [current, previous] concatenated along channels.

    Training can mix modes within a batch.
    Sampling uses one mode for the whole batch.
    """

    MODES = {
        "fully_coupled": 0,
        "prescribed_ocean": 1,
        "prescribed_atmosphere": 2,
    }

    def __init__(
        self,
        atmos_latent_channels,
        ocean_latent_channels,
        n_atmos_nodes=162,
        n_ocean_nodes=162,
        width=128,
        depth=4,
        n_heads=4,
        ffn_ratio=1.0,
        time_embedding_channels=64,
        co2_dropout=0.1,
        atmosphere_tau=0.5,
        ocean_tau=6.0,
    ):
        super().__init__()

        if not 0 <= co2_dropout <= 1:
            raise ValueError("co2_dropout must be between 0 and 1")
        if atmosphere_tau < 0 or ocean_tau < 0:
            raise ValueError("Noise timescales cannot be negative")

        self.atmos_channels = atmos_latent_channels
        self.ocean_channels = ocean_latent_channels
        self.co2_dropout = co2_dropout
        self.atmosphere_tau = atmosphere_tau
        self.ocean_tau = ocean_tau

        common = dict(
            atmos_latent_channels=atmos_latent_channels,
            ocean_latent_channels=ocean_latent_channels,
            n_atmos_nodes=n_atmos_nodes,
            n_ocean_nodes=n_ocean_nodes,
            width=width,
            depth=depth,
            n_heads=n_heads,
            ffn_ratio=ffn_ratio,
            time_embedding_channels=time_embedding_channels,
        )
        self.atmosphere = SphericalDiT(component="atmosphere", **common)
        self.ocean = SphericalDiT(component="ocean", **common)

        self.diffusion_steps = 1000
        self.sampling_timesteps = (999, 674, 373, 120, 29, 6, 0)

        betas = torch.linspace(1e-4, 2e-2, self.diffusion_steps)
        alpha_bar = torch.cumprod(1 - betas, dim=0)

        self.register_buffer("alpha", alpha_bar.sqrt())
        self.register_buffer("sigma", (1 - alpha_bar).sqrt())
        self.register_buffer("atmos_mean", torch.tensor(0.0))
        self.register_buffer("atmos_std", torch.tensor(1.0))
        self.register_buffer("ocean_mean", torch.tensor(0.0))
        self.register_buffer("ocean_std", torch.tensor(1.0))
        self.register_buffer("co2_scale", torch.full((1, 1), 280.0))

    def prepare_mode(self, mode, batch_size, device):
        """Return a mode index for each sample: [B]."""
        if mode is None:
            if self.training:
                # Equal probabilities initially.
                return torch.randint(3, (batch_size,), device=device)

            # Default validation task is fully coupled.
            mode = "fully_coupled"

        if isinstance(mode, str):
            mode = self.MODES[mode]

        mode = torch.as_tensor(mode, dtype=torch.long, device=device)
        if mode.ndim == 0:
            mode = mode.expand(batch_size)

        if mode.shape != (batch_size,):
            raise ValueError("mode must be a name, an integer, or a [B] tensor")
        if not bool(((mode >= 0) & (mode <= 2)).all()):
            raise ValueError("Mode indices must be 0, 1, or 2")

        return mode

    def forward(
        self,
        noisy_atmos,
        noisy_ocean,
        atmos_history,
        ocean_history,
        timestep,
        month,
        co2=None,
        mode="fully_coupled",
        drop_co2=None,
        run_atmos=True,
        run_ocean=True,
    ):
        """
        Evaluate requested heads on the SAME candidate pair.

        Both heads run for mixed-mode training.
        Only the free component runs during prescribed-mode sampling.
        """
        mode = self.prepare_mode(
            mode, noisy_atmos.shape[0], noisy_atmos.device,
        )

        inputs = dict(
            noisy_atmos=noisy_atmos,
            noisy_ocean=noisy_ocean,
            atmos_history=atmos_history,
            ocean_history=ocean_history,
            timestep=timestep,
            month=month,
            co2=co2,
            drop_co2=drop_co2,
        )

        velocity_atmos = None
        velocity_ocean = None

        if run_atmos:
            velocity_atmos = self.atmosphere(
                **inputs,
                drop_other_candidate=(mode == 1),
            )

        if run_ocean:
            velocity_ocean = self.ocean(
                **inputs,
                drop_other_candidate=(mode == 2),
            )

        return velocity_atmos, velocity_ocean

    def loss(
        self,
        target_atmos,
        target_ocean,
        atmos_history,
        ocean_history,
        month,
        co2=None,
        mode=None,
        drop_co2=None,
    ):
        """
        One target-month training loss.

        mode=None:
            Training: randomly choose a mode per sample.
            Evaluation: fully coupled.

        Explicit mode and drop_co2 tensors can be reused across
        an entire multi-month training rollout.
        """
        batch_size = target_atmos.shape[0]
        device = target_atmos.device

        mode = self.prepare_mode(mode, batch_size, device)

        timestep = torch.randint(
            self.diffusion_steps, (batch_size,), device=device,
        )
        alpha = self.alpha[timestep][:, None, None]
        sigma = self.sigma[timestep][:, None, None]

        # Independent corruption noise; shared diffusion timestep.
        noise_atmos = torch.randn_like(target_atmos)
        noise_ocean = torch.randn_like(target_ocean)

        noisy_atmos = alpha * target_atmos + sigma * noise_atmos
        noisy_ocean = alpha * target_ocean + sigma * noise_ocean

        target_v_atmos = alpha * noise_atmos - sigma * target_atmos
        target_v_ocean = alpha * noise_ocean - sigma * target_ocean

        if drop_co2 is None and self.training and co2 is not None:
            # Independent of coupling mode; shared across both heads.
            drop_co2 = (
                torch.rand(batch_size, 1, device=device)
                < self.co2_dropout
            )

        predicted_atmos, predicted_ocean = self(
            noisy_atmos,
            noisy_ocean,
            atmos_history,
            ocean_history,
            timestep,
            month,
            co2=co2,
            mode=mode,
            drop_co2=drop_co2,
        )

        # Average over nodes and channels, keeping the batch dimension.
        atmos_loss = (
            predicted_atmos - target_v_atmos
        ).square().mean(dim=(1, 2))

        ocean_loss = (
            predicted_ocean - target_v_ocean
        ).square().mean(dim=(1, 2))

        # Only score the component being generated in prescribed modes.
        sample_loss = torch.where(
            mode == 0,
            (atmos_loss + ocean_loss) / 2,
            torch.where(mode == 1, atmos_loss, ocean_loss),
        )

        return sample_loss.mean()

    @staticmethod
    def ar1_noise(like, previous, tau):
        """Terminal-noise persistence across climate months."""
        innovation = torch.randn_like(like)

        if previous is None:
            return innovation

        persistence = math.exp(-1 / tau) if tau > 0 else 0.0
        return (
            persistence * previous
            + math.sqrt(1 - persistence**2) * innovation
        )

    def solver_step(
        self,
        noisy,
        clean,
        previous_clean,
        step,
        next_step,
        previous_step,
    ):
        """DPM-Solver++(2M), midpoint variant."""
        alpha = self.alpha[step]
        sigma = self.sigma[step]
        alpha_next = self.alpha[next_step]
        sigma_next = self.sigma[next_step]

        if previous_clean is None:
            predicted_noise = (noisy - alpha * clean) / sigma
            return alpha_next * clean + sigma_next * predicted_noise

        log_ratio = torch.log(alpha) - torch.log(sigma)
        next_log_ratio = torch.log(alpha_next) - torch.log(sigma_next)
        previous_log_ratio = (
            torch.log(self.alpha[previous_step])
            - torch.log(self.sigma[previous_step])
        )

        interval = next_log_ratio - log_ratio
        previous_interval = log_ratio - previous_log_ratio
        ratio = previous_interval / interval

        corrected_clean = (
            clean + 0.5 * (clean - previous_clean) / ratio
        )

        return (
            (sigma_next / sigma) * noisy
            - alpha_next * torch.expm1(-interval) * corrected_clean
        )

    def sample(
        self,
        atmos_history,
        ocean_history,
        month,
        co2=None,
        mode="fully_coupled",
        previous_noise=(None, None),
        guidance_scale=1.0,
        drop_co2=None,
    ):
        """
        Generate one climate month.

        Returns:
            atmos_prediction, ocean_prediction, terminal_noise

        The prescribed component's prediction and noise are None.
        Its observed history is advanced by the caller.

        Differentiable by default: use torch.no_grad() for inference.

        drop_co2 optionally supplies a fixed [B, 1] mask during rollout
        training. For ordinary inference leave it None.
        """
        mode_id = self.MODES[mode]
        run_atmos = mode_id != 2
        run_ocean = mode_id != 1

        if drop_co2 is not None and guidance_scale != 1.0:
            raise ValueError(
                "Use fixed CO2 dropout or guidance, not both in one call"
            )

        atmos_template = atmos_history[..., :self.atmos_channels]
        ocean_template = ocean_history[..., :self.ocean_channels]

        # Draw terminal noise ONLY for components we will generate.
        terminal_atmos = (
            self.ar1_noise(
                atmos_template, previous_noise[0], self.atmosphere_tau,
            )
            if run_atmos else None
        )
        terminal_ocean = (
            self.ar1_noise(
                ocean_template, previous_noise[1], self.ocean_tau,
            )
            if run_ocean else None
        )

        # Zero placeholders satisfy the common input interface.
        # The missing-candidate mask explicitly marks them unavailable.
        # No noise is added to the prescribed component.
        noisy_atmos = (
            terminal_atmos if run_atmos
            else torch.zeros_like(atmos_template)
        )
        noisy_ocean = (
            terminal_ocean if run_ocean
            else torch.zeros_like(ocean_template)
        )

        previous_atmos_clean = None
        previous_ocean_clean = None
        previous_step = None

        for index, step in enumerate(self.sampling_timesteps):
            timestep = torch.full(
                (atmos_history.shape[0],),
                step,
                dtype=torch.long,
                device=atmos_history.device,
            )

            inputs = dict(
                noisy_atmos=noisy_atmos,
                noisy_ocean=noisy_ocean,
                atmos_history=atmos_history,
                ocean_history=ocean_history,
                timestep=timestep,
                month=month,
                mode=mode,
                run_atmos=run_atmos,
                run_ocean=run_ocean,
            )

            v_atmos, v_ocean = self(
                **inputs, co2=co2, drop_co2=drop_co2,
            )

            # Keep histories, coupling mode, and candidates identical.
            if co2 is not None and guidance_scale != 1.0:
                null_atmos, null_ocean = self(**inputs, co2=None)

                if run_atmos:
                    v_atmos = null_atmos + guidance_scale * (
                        v_atmos - null_atmos
                    )
                if run_ocean:
                    v_ocean = null_ocean + guidance_scale * (
                        v_ocean - null_ocean
                    )

            alpha = self.alpha[step]
            sigma = self.sigma[step]

            clean_atmos = (
                alpha * noisy_atmos - sigma * v_atmos
                if run_atmos else None
            )
            clean_ocean = (
                alpha * noisy_ocean - sigma * v_ocean
                if run_ocean else None
            )

            if index == len(self.sampling_timesteps) - 1:
                return (
                    clean_atmos,
                    clean_ocean,
                    (terminal_atmos, terminal_ocean),
                )

            next_step = self.sampling_timesteps[index + 1]

            # Both predictions exist before either candidate advances.
            if run_atmos:
                noisy_atmos = self.solver_step(
                    noisy_atmos, clean_atmos, previous_atmos_clean,
                    step, next_step, previous_step,
                )
            if run_ocean:
                noisy_ocean = self.solver_step(
                    noisy_ocean, clean_ocean, previous_ocean_clean,
                    step, next_step, previous_step,
                )

            previous_atmos_clean = clean_atmos
            previous_ocean_clean = clean_ocean
            previous_step = step

    def fit_preprocessing(self, atmos, ocean, co2_scale=280):
        with torch.no_grad():
            self.atmos_mean.copy_(atmos.mean())
            self.atmos_std.copy_(atmos.std())
            self.ocean_mean.copy_(ocean.mean())
            self.ocean_std.copy_(ocean.std())
            self.co2_scale.fill_(co2_scale)

    def preprocess(self, atmos=None, ocean=None, co2=None):
        return(
            (atmos - self.atmos_mean) / self.atmos_std if atmos is not None else atmos,
            (ocean - self.ocean_mean) / self.ocean_std if ocean is not None else ocean,
            torch.log(co2 / self.co2_scale) if co2 is not None else co2
        )

    @torch.no_grad()
    def rollout(
        self,
        initial_atmos,
        initial_ocean,
        months,
        *,
        co2=None,
        prescribed_atmos=None,
        prescribed_ocean=None,
        guidance_scale=1.0,
        previous_noise=(None, None),
        return_noise=False,
    ):
        """
        Generate a latent trajectory.

        Initial conditions:
            initial_atmos: [B, 2, N_atmos, C_atmos]
            initial_ocean: [B, 2, N_ocean, C_ocean]
            Time order is [previous, current].

        Future inputs:
            months: [T], calendar month numbers 1..12
            co2: optional [T], [T, 1], or [B, T, 1]
                Already standardized using training statistics.
            prescribed_atmos: optional [B, T, N_atmos, C_atmos]
            prescribed_ocean: optional [B, T, N_ocean, C_ocean]

        Returns:
            atmosphere: [B, T, N_atmos, C_atmos]
            ocean:      [B, T, N_ocean, C_ocean]

            Initial conditions are not included.
            Prescribed components are included in the returned trajectories.
            Outputs are collected on CPU.

            If return_noise=True, also return the final terminal-noise tuple
            for continuing the trajectory without resetting AR(1) persistence.
        """
        if prescribed_atmos is not None and prescribed_ocean is not None:
            raise ValueError(
                "Prescribe at most one component; otherwise nothing is generated."
            )

        if initial_atmos.ndim != 4 or initial_atmos.shape[1] != 2:
            raise ValueError("initial_atmos must have shape [B, 2, N, C]")
        if initial_ocean.ndim != 4 or initial_ocean.shape[1] != 2:
            raise ValueError("initial_ocean must have shape [B, 2, N, C]")

        batch_size = initial_atmos.shape[0]
        if initial_ocean.shape[0] != batch_size:
            raise ValueError("Atmosphere and ocean batch sizes must match")

        parameter = next(self.parameters())
        device = parameter.device
        dtype = parameter.dtype

        months = torch.as_tensor(months, device=device)
        if months.ndim != 1 or months.numel() == 0:
            raise ValueError("months must be a nonempty [T] sequence")
        if not bool(((months >= 1) & (months <= 12) &
                    (months == months.round())).all()):
            raise ValueError("Calendar months must be integers from 1 through 12")
        if not bool((months[1:] == months[:-1] % 12 + 1).all()):
            raise ValueError("Rollout months must be consecutive")

        n_months = months.numel()

        # Validate prescribed trajectories without moving the full arrays
        # to the accelerator. Only the current month is transferred below.
        for name, trajectory, initial in (
            ("prescribed_atmos", prescribed_atmos, initial_atmos),
            ("prescribed_ocean", prescribed_ocean, initial_ocean),
        ):
            if trajectory is not None:
                expected = (batch_size, n_months, *initial.shape[2:])
                if tuple(trajectory.shape) != expected:
                    raise ValueError(f"{name} must have shape {expected}")

        if prescribed_ocean is not None:
            mode = "prescribed_ocean"
        elif prescribed_atmos is not None:
            mode = "prescribed_atmosphere"
        else:
            mode = "fully_coupled"

        # CO2 can be shared across the batch or member-specific.
        if co2 is not None:
            co2 = torch.as_tensor(co2, dtype=dtype, device=device)

            if co2.shape == (n_months,):
                co2 = co2[None, :, None].expand(batch_size, -1, -1)
            elif co2.shape == (n_months, 1):
                co2 = co2[None, :, :].expand(batch_size, -1, -1)
            elif co2.shape != (batch_size, n_months, 1):
                raise ValueError(
                    "co2 must have shape [T], [T, 1], or [B, T, 1]"
                )
        elif guidance_scale != 1.0:
            raise ValueError("CO2 guidance requires a supplied CO2 trajectory")

        initial_atmos = initial_atmos.to(device=device, dtype=dtype)
        initial_ocean = initial_ocean.to(device=device, dtype=dtype)

        atmos_previous = initial_atmos[:, 0]
        atmos_current = initial_atmos[:, 1]
        ocean_previous = initial_ocean[:, 0]
        ocean_current = initial_ocean[:, 1]

        noise = tuple(
            None if value is None else value.to(device=device, dtype=dtype)
            for value in previous_noise
        )

        atmos_output = []
        ocean_output = []

        was_training = self.training
        self.eval()

        try:
            for index in range(n_months):
                # The denoisers expect current first, previous second.
                atmos_history = torch.cat(
                    [atmos_current, atmos_previous], dim=-1
                )
                ocean_history = torch.cat(
                    [ocean_current, ocean_previous], dim=-1
                )

                target_month = months[index].expand(batch_size)
                target_co2 = None if co2 is None else co2[:, index]

                next_atmos, next_ocean, noise = self.sample(
                    atmos_history,
                    ocean_history,
                    target_month,
                    co2=target_co2,
                    mode=mode,
                    previous_noise=noise,
                    guidance_scale=guidance_scale,
                )

                # Insert the prescribed target-month state AFTER sampling.
                # It becomes history for the following month's prediction.
                if prescribed_atmos is not None:
                    next_atmos = torch.as_tensor(
                        prescribed_atmos[:, index],
                        device=device,
                        dtype=dtype,
                    )

                if prescribed_ocean is not None:
                    next_ocean = torch.as_tensor(
                        prescribed_ocean[:, index],
                        device=device,
                        dtype=dtype,
                    )

                atmos_output.append(next_atmos.cpu())
                ocean_output.append(next_ocean.cpu())

                atmos_previous, atmos_current = atmos_current, next_atmos
                ocean_previous, ocean_current = ocean_current, next_ocean

        finally:
            self.train(was_training)

        atmosphere = torch.stack(atmos_output, dim=1)
        ocean = torch.stack(ocean_output, dim=1)

        if return_noise:
            final_noise = tuple(
                None if value is None else value.cpu()
                for value in noise
            )
            return atmosphere, ocean, final_noise

        return atmosphere, ocean