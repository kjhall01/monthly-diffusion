"""Xarray-first orchestration of the public atomic PyTorch modules.

This is deliberately not an nn.Module: .train() fits two stages rather than
overriding PyTorch's train/eval-mode switch. The modules remain accessible as
.autoencoder and .transition for scientific experiments.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import logging
import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader

from .modules.mesh import make_equiangular_grid_from_xarray, build_mesh_hierarchy_context
from .modules.autoencoder import AutoencoderConfig, CoupledAutoencoder
from .components.coupled_latent_diffusion import CoupledLatentDiffusion, CoupledDiffusionConfig
from .data.configs import ModelConfig
from .data.preprocessing import Preprocessor, LatentNormalizer
from .data.dataloader import PhysicalStates, LatentSequences, co2_values, month_features, validate_fields, validate_units
from .training.autoencoder import train_autoencoder
from .training.coupled_training import train_coupled_diffusion
from .training.common import atomic_save

log = logging.getLogger(__name__)


class MonthlyDiffusionV1:
    def __init__(self, *, statics=None, config=None, device=None, **kwargs):
        if config is None:
            self.config = ModelConfig(**kwargs)
        elif isinstance(config, (str, Path)):
            self.config = ModelConfig.from_yaml(config, **kwargs)
        else:
            values = config.to_dict() if isinstance(config, ModelConfig) else dict(config)
            values.update(kwargs)
            self.config = ModelConfig(**values)
        self.device = torch.device(device or ("mps" if torch.backends.mps.is_available() else
                                              "cuda" if torch.cuda.is_available() else "cpu"))
        self.statics = statics
        self.preprocessor = Preprocessor(self.config.variables, self.config.centering, self.config.ocean_support)
        self.latent_normalizers = {}
        self.initialization_moments = {}
        self.stages = {}
        self.autoencoder = self.transition = None

    def _build(self):
        if self.statics is None:
            raise ValueError("Supply static phi_s and lsm fields once, at construction or prepare()")
        self.statics = self.statics[["phi_s", "lsm"]].transpose("lat", "lon").load()
        grid = make_equiangular_grid_from_xarray(self.statics.lat, self.statics.lon, name="grid")
        c = self.config
        self.geometry = build_mesh_hierarchy_context(nlat=len(self.statics.lat), nlon=len(self.statics.lon),
            mesh_levels=c.mesh_levels, rotation_degrees=c.rotation, grid=grid)
        self.names = {component: [v.name for v in c.variables if v.component == component]
                      for component in ("atmosphere", "ocean")}
        if not all(self.names.values()):
            raise ValueError("MonthlyDiffusionV1 requires both components; use atomic modules for a single component")
        ae_config = AutoencoderConfig(
            c.ae_width, c.latent_channels, tuple(c.mesh_levels), tuple(c.refiners),
            tuple(c.rotation), tuple(c.co2_film_levels))
        self.autoencoder = CoupledAutoencoder(len(self.names["atmosphere"]), len(self.names["ocean"]),
            self.geometry, (len(self.statics.lat), len(self.statics.lon)), ae_config,
            ocean_mask_aware=c.ocean_mask_aware).to(self.device)
        self.transition = CoupledLatentDiffusion(CoupledDiffusionConfig(
            atmosphere_latent_channels=c.latent_channels, ocean_latent_channels=c.latent_channels,
            n_latent_tokens=self.geometry.graphs[f"mesh_l{len(c.mesh_levels)-1}"].n_nodes,
            width=c.transition_width, depth=c.transition_depth, n_heads=c.transition_heads,
            ffn_ratio=c.transition_ffn_ratio, time_embedding_channels=c.time_embedding_channels,
            atmosphere_tau=c.atmosphere_tau, ocean_tau=c.ocean_tau,
            coupling=c.coupling, cross_gradients=c.cross_gradients)).to(self.device)

    def prepare(self, atmosphere, ocean, *, co2, statics=None):
        """Fit physical statistics on these TRAINING states only, then build modules."""
        if statics is not None:
            self.statics = statics
        if self.autoencoder is not None:
            raise RuntimeError("Already prepared; load a checkpoint or construct a fresh wrapper")
        torch.manual_seed(self.config.autoencoder_training.seed)
        self._build()
        self._validate_pair(atmosphere, ocean)
        forcing = co2_values(co2, atmosphere.time)
        self.preprocessor.fit(atmosphere, ocean, self.statics, forcing)
        return self

    def _validate_pair(self, atmosphere, ocean):
        for component, ds in (("atmosphere", atmosphere), ("ocean", ocean)):
            validate_fields(ds, self.names[component], self.statics)
            validate_units(ds, [v for v in self.config.variables if v.component == component])
        if not np.array_equal(atmosphere.time, ocean.time):
            raise ValueError("Atmosphere/ocean timestamps must match exactly, including calendar")

    def train(self, atmos, ocean, *, validation, co2, directory="runs/mdv1", resume=False):
        """AE-only -> fixed train-latent statistics -> cached coupled diffusion.

        Pass already-separated training and validation xarray datasets. No test
        data are used here. Stage-specific latest/best checkpoints and history
        tables are saved; completed stages are skipped on resume.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        progress = directory/"model.pt"
        if progress.exists():
            if not resume:
                raise FileExistsError("Run already exists; use resume=True or another directory")
            requested = self.config
            restored = self.load(progress, device=self.device)
            old, new = restored.config.to_dict(), requested.to_dict()
            for key in ("autoencoder_training", "diffusion_training"):
                old.pop(key); new.pop(key)
            if old != new:
                raise ValueError("Resume architecture/preprocessing differs from saved run")
            self.__dict__.update(restored.__dict__)
            self.config = requested
        if self.autoencoder is None:
            self.prepare(atmos, ocean, co2=co2)
            self.save(progress)
        self._validate_pair(atmos, ocean)
        self._validate_pair(*validation)
        if not np.array_equal(atmos.time, self.preprocessor.training_times):
            raise ValueError("Training times differ from fitted preprocessing")
        if not np.array_equal(co2_values(co2, atmos.time), self.preprocessor.training_co2):
            raise ValueError("Training CO2 differs from the saved preprocessing fit")
        for name, ds in (("atmosphere", atmos), ("ocean", ocean)):
            if dict(ds.attrs) != self.preprocessor.source_attrs[name]:
                raise ValueError("Training source metadata changed; use a new run directory")
        if np.intersect1d(atmos.time.values, validation[0].time.values).size:
            raise ValueError("Training and validation times overlap")
        if not self.initialization_moments:
            self.fit_stochastic_initialization(atmos, ocean)
            self.save(progress)
        self.config.save(directory/"config.yaml")
        datasets = [PhysicalStates(a, o, self.preprocessor, co2_values(co2, a.time))
                    for a, o in ((atmos, ocean), validation)]
        statics = self.preprocessor.static_tensor(self.statics).to(self.device).unsqueeze(0)
        if "autoencoder" not in self.stages:
            log.info("Stage 1: reconstruction-only autoencoder training")
            self.autoencoder.requires_grad_(True)
            weights = {comp: torch.tensor([v.weight for v in self.config.variables if v.component == comp])
                       for comp in self.names}
            result = train_autoencoder(self.autoencoder, *datasets, statics=statics,
                latitudes=torch.tensor(self.statics.lat.values), weights=weights,
                config=self.config.autoencoder_training, directory=directory/"autoencoder", resume=resume)
            self.stages["autoencoder"] = {k: v for k, v in result.items() if k != "history"}
            self.save(progress)
        self.autoencoder.eval().requires_grad_(False)
        cache_path = directory/"latents.pt"
        if cache_path.exists():
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            if cache["ae_digest"] != self.weight_digest(autoencoder_only=True):
                raise ValueError("Latent cache belongs to a different selected autoencoder")
            if not np.array_equal(cache["training_time"], atmos.time) or not np.array_equal(cache["validation_time"], validation[0].time):
                raise ValueError("Cached training/validation dates differ from supplied splits")
            self.latent_normalizers = cache["normalizers"]
        else:
            log.info("Stage 2: encoding training/validation states once; fitting train-only latent statistics")
            arrays = []
            with torch.no_grad():
                for dataset in datasets:
                    result = {component: [] for component in self.names}
                    loader = DataLoader(dataset, batch_size=self.config.autoencoder_training.batch_size)
                    for batch in loader:
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        for component in self.names:
                            node_mask = (torch.isfinite(batch[component][:, 0])
                                         if getattr(self.autoencoder, component, None).mask_aware else None)
                            z = getattr(self.autoencoder, component).encode(batch[component],
                                statics.expand(len(batch["month"]), -1, -1, -1), batch["month"], batch["co2"],
                                node_mask=node_mask)
                            result[component].append(z.cpu())
                    arrays.append({k: torch.cat(v) for k, v in result.items()})
            self.latent_normalizers = {k: LatentNormalizer().fit(arrays[0][k]) for k in self.names}
            for item in arrays:
                for key in item:
                    item[key] = self.latent_normalizers[key].normalize(item[key])
            cache = dict(training=arrays[0], validation=arrays[1], normalizers=self.latent_normalizers,
                         ae_digest=self.weight_digest(autoencoder_only=True),
                         training_time=atmos.time, validation_time=validation[0].time)
            atomic_save(cache, cache_path)
        self.save(progress)
        if "diffusion" not in self.stages:
            log.info("Stage 3: diffusion-only training on immutable cached latents")
            sequences = [LatentSequences(cache[split]["atmosphere"], cache[split]["ocean"],
                                         cache[split+"_time"], self.config.diffusion_training.rollout_steps)
                         for split in ("training", "validation")]
            result = train_coupled_diffusion(self.transition, *sequences, config=self.config.diffusion_training,
                                             directory=directory/"diffusion", resume=resume)
            self.stages["diffusion"] = {k: v for k, v in result.items() if k != "history"}
            self.save(progress)
        return self

    @torch.no_grad()
    def encode(self, atmos=None, ocean=None, *, co2):
        """Physical states -> transition-normalized latents; omitted component -> None."""
        if not self.latent_normalizers:
            raise RuntimeError("Train both stages or load a prepared model with fixed latent statistics")
        self.autoencoder.eval()
        outputs = []
        for component, ds in (("atmosphere", atmos), ("ocean", ocean)):
            if ds is None:
                outputs.append(None)
                continue
            validate_fields(ds, self.names[component], self.statics)
            validate_units(ds, [v for v in self.config.variables if v.component == component])
            months = month_features(ds.time).to(self.device)
            forcing = torch.tensor(((co2_values(co2, ds.time)-self.preprocessor.co2_mean)/self.preprocessor.co2_std).values,
                                   dtype=torch.float32, device=self.device)[:, None]
            static = self.preprocessor.static_tensor(self.statics).to(self.device)[None]
            latents = []
            batch = self.config.autoencoder_training.batch_size
            for start in range(0, len(ds.time), batch):
                physical = self.preprocessor.encode(ds.isel(time=slice(start, start+batch)), component).compute()
                values = np.stack([physical[v].transpose("time", "lat", "lon").values for v in self.names[component]], 1)
                x = torch.from_numpy(values).to(self.device)
                node_mask = (torch.isfinite(x[:, 0])
                             if getattr(self.autoencoder, component, None).mask_aware else None)
                z = getattr(self.autoencoder, component).encode(x, static.expand(len(x), -1, -1, -1),
                                                               months[start:start+batch], forcing[start:start+batch],
                                                               node_mask=node_mask)
                latents.append(self.latent_normalizers[component].normalize(z).cpu())
            graph = self.geometry.graphs[f"mesh_l{len(self.config.mesh_levels)-1}"]
            outputs.append(xr.DataArray(torch.cat(latents).numpy(), dims=("time", "node", "channel"),
                coords={"time": ds.time, "node": np.arange(graph.n_nodes),
                        "channel": np.arange(self.config.latent_channels),
                        "node_lat": ("node", graph.lat.numpy()), "node_lon": ("node", graph.lon.numpy())},
                name=component, attrs={"coordinate_system": "fixed_transition_normalized", "units": "1"}))
        return tuple(outputs)

    @torch.no_grad()
    def decode(self, atmos=None, ocean=None, *, co2):
        """Transition-normalized xarray latents -> masked physical-unit fields."""
        self.autoencoder.eval()
        outputs = []
        for component, z in (("atmosphere", atmos), ("ocean", ocean)):
            if z is None:
                outputs.append(None)
                continue
            with_member = "member" in z.dims
            dims = ("time", "member", "node", "channel") if with_member else ("time", "node", "channel")
            values = z.transpose(*dims).values
            members = z.sizes.get("member", 1)
            values = values.reshape(-1, *values.shape[-2:])
            month = month_features(z.time).repeat_interleave(members, 0).to(self.device)
            forcing = torch.tensor(((co2_values(co2, z.time)-self.preprocessor.co2_mean)/self.preprocessor.co2_std).values,
                                   dtype=torch.float32, device=self.device).repeat_interleave(members)[:, None]
            static = self.preprocessor.static_tensor(self.statics).to(self.device)[None]
            decoded = []
            batch = self.config.autoencoder_training.batch_size
            ocean_mask = None
            if component == "ocean" and self.config.ocean_mask_aware:
                # Prefer the train-fitted SST availability mask so training and
                # latent decoding use the same ocean support.  The LSM fallback
                # keeps the atomic wrapper usable with custom ocean variables.
                support = self.preprocessor.stats.get("sst", {}).get("mask")
                if support is None:
                    support = (1. - self.statics.lsm).clip(0., 1.)
                ocean_mask = torch.as_tensor(np.asarray(support), dtype=torch.float32,
                                              device=self.device)
            for start in range(0, len(values), batch):
                latent = torch.as_tensor(values[start:start+batch], dtype=torch.float32, device=self.device)
                latent = self.latent_normalizers[component].denormalize(latent)
                fields = getattr(self.autoencoder, component).decode(latent, static.expand(len(latent), -1, -1, -1),
                                    month[start:start+batch], forcing[start:start+batch],
                                    node_mask=None if ocean_mask is None else ocean_mask[None].expand(len(latent), -1, -1))
                decoded.append(fields.cpu().numpy())
            values = np.concatenate(decoded).reshape(len(z.time), members, len(self.names[component]),
                                                     len(self.statics.lat), len(self.statics.lon))
            coords = dict(time=z.time, lat=self.statics.lat, lon=self.statics.lon)
            if with_member:
                coords["member"] = z.member
                fields = {name: (("time", "member", "lat", "lon"), values[:, :, i])
                          for i, name in enumerate(self.names[component])}
            else:
                fields = {name: (("time", "lat", "lon"), values[:, 0, i])
                          for i, name in enumerate(self.names[component])}
            outputs.append(self.preprocessor.decode(xr.Dataset(fields, coords=coords), component))
        return tuple(outputs)

    def fit_stochastic_initialization(self, atmos, ocean):
        """Fit local monthly Gaussian input moments on EXACT checkpoint training dates.

        This does not train/change any network or normalization. For older
        checkpoints, call once with physical xarray datasets (extra dates are
        ignored), then save a new export. New .train() runs do this automatically.
        Moments live in already transformed/centered/scaled field coordinates.
        Zero SD stays zero; fewer than two finite samples means unavailable.
        Only one variable is materialized at a time, not the full training state.
        """
        if not self.preprocessor.stats:
            raise RuntimeError("Prepare or load a model before fitting initialization moments")
        dates = self.preprocessor.training_times
        atmos, ocean = atmos.sel(time=dates), ocean.sel(time=dates)
        self._validate_pair(atmos, ocean)
        for c, ds in (("atmosphere", atmos), ("ocean", ocean)):
            if dict(ds.attrs) != self.preprocessor.source_attrs[c]:
                raise ValueError("Initialization source metadata differs from checkpoint training data")
        moments = {}
        for c, ds in (("atmosphere", atmos), ("ocean", ocean)):
            normalized = self.preprocessor.encode(ds, c)
            for name in self.names[c]:
                x = normalized[name].transpose("time", "lat", "lon").compute().astype("float64")
                grouped = x.where(np.isfinite(x)).groupby("time.month")
                count = grouped.count("time")
                moments[name] = xr.Dataset(dict(
                    mean=grouped.mean("time", skipna=True).where(count >= 2).astype("float32"),
                    std=grouped.std("time", skipna=True, ddof=0).where(count >= 2).astype("float32"),
                    count=count.astype("int16"))).reindex(month=np.arange(1,13))
                moments[name].attrs.update(component=c, coordinate_system="checkpoint normalized field",
                    training_start=str(dates[0]), training_end=str(dates[-1]), training_months=len(dates), sd_ddof=0)
                log.info("Fitted stochastic-initialization moments: %s", name)
        self.initialization_moments = moments
        return self

    def initialize(self, atmos=None, ocean=None, *, co2, members=5, seed=20260716, gaussian_start=None,
                   encoded_noise_start=None, stochastic_start=None):
        """Return a stateful latent iterator, with observed or stochastic histories.

        Without physical ICs, ``stochastic_start='1850-01'`` draws independent
        normalized fields matched to training-month/gridpoint mean and SD,
        then encodes the preceding two months. The FIRST next() predicts
        January, decodes/re-encodes it once and feeds that projection back into
        the current history slot. It returns the January *prediction*, exactly
        as in the cycling-duration experiment. Later steps are latent-only.
        No months are skipped and terminal noise/RNG histories are not reset.
        CO2 must cover both history months and every generated month (or be a
        scalar). Training-fitted initialization_moments must be present.
        """
        from .forecast import Forecast
        return Forecast(self, atmos, ocean, co2=co2, members=members, seed=seed,
                        gaussian_start=gaussian_start, encoded_noise_start=encoded_noise_start,
                        stochastic_start=stochastic_start)

    def rollout(self, atmos=None, ocean=None, *, co2, months, members=5, seed=20260716, path=None, resume=False,
                open_result=True, gaussian_start=None, spinup_months=0, encoded_noise_start=None,
                save_spinup_latents=False, stochastic_start=None, extend=False):
        """Generate ``months`` saved months, optionally after discarded latent-only spin-up.

        ``gaussian_start='1845-01'`` uses independent N(0,1) history latents
        for November/December 1844, in fixed transition-normalized coordinates.
        No physical initial states are encoded in this mode.
        Alternatively, ``encoded_noise_start`` encodes independent N(0,1)
        preprocessed fields with train-fitted masks and the history months'
        calendar/CO2 conditioning. ``save_spinup_latents=True`` retains every
        spin-up latent in resumable annual blocks without decoding it.
        ``stochastic_start`` instead uses local training-month noise moments
        and one initial predict/decode/re-encode cycle, then latent-only steps.
        That first generated month counts toward spinup_months, if requested;
        otherwise it is the first saved prediction. See initialize().
        ``resume=True, extend=True`` continues a completed archive to a longer
        horizon, requiring identical checkpoint/initialization/forcing history.
        """
        from .forecast import rollout
        return rollout(self, atmos, ocean, co2=co2, months=months, members=members,
                       seed=seed, path=path, resume=resume, open_result=open_result,
                       gaussian_start=gaussian_start, spinup_months=spinup_months,
                       encoded_noise_start=encoded_noise_start, save_spinup_latents=save_spinup_latents,
                       stochastic_start=stochastic_start, extend=extend)

    def weight_digest(self, *, autoencoder_only=False):
        digest = hashlib.sha256()
        modules = (self.autoencoder,) if autoencoder_only else (self.autoencoder, self.transition)
        for module in modules:
            for name, tensor in module.state_dict().items():
                digest.update(name.encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def save(self, path):
        """Trusted local torch checkpoint: parameters plus complete scientific state."""
        payload = dict(format_version=1, config=self.config.to_dict(), statics=self.statics,
            preprocessing=self.preprocessor, normalizers=self.latent_normalizers, stages=self.stages,
            initialization_moments=self.initialization_moments,
            autoencoder=self.autoencoder.state_dict(), transition=self.transition.state_dict(),
            torch_version=str(torch.__version__))
        atomic_save(payload, path)

    @classmethod
    def load(cls, path, *, device=None):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("format_version") != 1:
            raise ValueError("Not a monthly_diffusion standalone checkpoint; legacy importing is not supported")
        model = cls(statics=saved["statics"], config=saved["config"], device=device)
        model._build()
        model.autoencoder.load_state_dict(saved["autoencoder"])
        model.transition.load_state_dict(saved["transition"])
        model.preprocessor, model.latent_normalizers = saved["preprocessing"], saved["normalizers"]
        model.stages = saved["stages"]
        model.initialization_moments = saved.get("initialization_moments", {})
        model.autoencoder.eval()
        model.transition.eval()
        return model
