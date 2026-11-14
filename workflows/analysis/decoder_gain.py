"""Same-latent forcing and ENSO gain, isolated from transition dynamics."""
from monthly_diffusion.checkpoints import load_checkpoint
from pathlib import Path
import json
import logging
import numpy as np
import pandas as pd
import torch
import xarray as xr
from scipy.signal import detrend
from workflows.legacy_model import MonthlyDiffusionV1
from workflows.data.sources import open_co2
from workflows.analysis.archives import ClimateArchive
from workflows.analysis.indices import enso_index, regional_mean
from workflows.analysis.climatology import remove_monthly_climatology
from workflows.analysis.spatial import weighted_mean, weighted_uncentered_pattern_correlation
from workflows.analysis.long_rollout import save_dataset
from workflows.analysis.long_rollout_plots import finish, display
from workflows.analysis.plotting import plot_map_comparison, plot_response_map

log = logging.getLogger(__name__)


def decoder_diagnostics(config, output, *, device=None):
    """Paired ±0.5 SD training ENSO latent perturbations at fixed CO2.

    Fit the direction on the model's own training states only. Hold 36 base latents,
    their calendar months, and statics at every CO2. Gains include all downstream
    nonlinearities and inverse transforms, but no transition evolution.
    """
    output = Path(output)/"decoder_gains"
    output.mkdir(parents=True, exist_ok=True)
    path = output/"paired_decodes.nc"
    if not path.exists():
        run = Path(config["checkpoint"]).parent
        model = MonthlyDiffusionV1.load(config["checkpoint"], device=device)
        archive = json.loads((Path(config["forecast"])/"forecast.json").read_text())
        if model.weight_digest() != archive["checkpoint_sha256"]:
            raise ValueError("Decoder checkpoint differs from the forecast being diagnosed")
        torch.set_num_threads(4)
        log.info("Decoder gain calculation uses %s", model.device)
        cache = load_checkpoint(run/"latents.pt", map_location="cpu", weights_only=False)
        if cache["ae_digest"] != model.weight_digest(autoencoder_only=True):
            raise ValueError("Cached training latents do not match selected autoencoder")
        dates = np.asarray(cache["training_time"])
        training = ClimateArchive([config["training_source"]],
            ocean_support=getattr(model.preprocessor, "ocean_support_mask", None),
            ocean_variables=model.names["ocean"])
        years = model.preprocessor.training_times
        if not np.array_equal(dates, years):
            raise ValueError("Training latent dates disagree with preprocessing")
        date_axis = xr.DataArray(dates, dims="time", coords=dict(time=dates))
        keys = date_axis.dt.year.values*12+date_axis.dt.month.values
        selection = np.searchsorted(training.keys, keys)
        if not np.array_equal(training.keys[selection], keys):
            raise ValueError("Training source dates unavailable")
        index_parts = []
        for block in training.blocks(selection):
            field = training.read("sst", block).isel(member=0, drop=True)
            index_parts.append(enso_index(field.to_dataset(), remove_monthly=False))
        physical_index = remove_monthly_climatology(xr.concat(index_parts, "time"))
        values = detrend(physical_index.values, type="linear")
        values /= values.std()
        index = xr.DataArray(values, dims="time", coords=dict(time=dates))
        components = {}
        directions = {}
        for name in ("atmosphere", "ocean"):
            z = xr.DataArray(cache["training"][name].numpy(), dims=("time", "node", "channel"),
                coords=dict(time=dates, node=np.arange(cache["training"][name].shape[1]),
                            channel=np.arange(cache["training"][name].shape[2])))
            components[name] = z
            directions[name] = (remove_monthly_climatology(z)*index).mean("time")
        save_dataset(xr.Dataset({name: d for name, d in directions.items()}), output/"enso_latent_directions.nc")
        lo, hi = config["training"]
        sample_years = [lo, (lo+hi)//2, hi]
        keep = np.flatnonzero(np.isin(date_axis.dt.year, sample_years))
        co2 = open_co2(config["co2_netcdf"], expected_sha256=config.get("co2_sha256"))
        early = float(co2.sel(time=slice(str(config["early"][0]), str(config["early"][1]))).mean())
        late = float(co2.sel(time=slice(str(config["late"][0]), str(config["late"][1]))).mean())
        levels = np.unique([early, 350., 450., 600., late])
        names = ("t2m", "sst", "mtpr")
        expressions, base = [], []
        for ppm in levels:
            sums = {n: None for n in names}
            base_sums = {n: None for n in names}
            for start in range(0, len(keep), 2):
                pick = keep[start:start+2]
                paired = []
                for sign in (-.5, .5):
                    za = components["atmosphere"].isel(time=pick)+sign*directions["atmosphere"]
                    zo = components["ocean"].isel(time=pick)+sign*directions["ocean"]
                    a, o = model.decode(za, zo, co2=float(ppm))
                    paired.append(xr.merge((a[["t2m", "mtpr"]], o[["sst"]])))
                for n in names:
                    diff = (paired[1][n]-paired[0][n]).sum("time", skipna=False)
                    sums[n] = diff if sums[n] is None else sums[n]+diff
                if ppm in (levels[0], levels[-1]):
                    a, o = model.decode(components["atmosphere"].isel(time=pick), components["ocean"].isel(time=pick), co2=float(ppm))
                    for n, ds in (("t2m", a), ("mtpr", a), ("sst", o)):
                        field = ds[n].sum("time", skipna=False)
                        base_sums[n] = field if base_sums[n] is None else base_sums[n]+field
            expressions.append(xr.Dataset({n: v/len(keep) for n, v in sums.items()}).expand_dims(co2_ppm=[ppm]))
            if ppm in (levels[0], levels[-1]):
                base.append(xr.Dataset({n: v/len(keep) for n, v in base_sums.items()}).expand_dims(co2_ppm=[ppm]))
            log.info("Finished paired decoder gains at %.2f ppm", ppm)
        gains = xr.concat(expressions, "co2_ppm")
        base = xr.concat(base, "co2_ppm")
        result = xr.merge([gains.rename({n: f"gain_{n}" for n in names}),
            (base.isel(co2_ppm=-1, drop=True)-base.isel(co2_ppm=0, drop=True)).rename({n: f"response_{n}" for n in names})])
        for name in names:
            for prefix in ("gain", "response"):
                result[f"{prefix}_{name}"].attrs["units"] = training.units[name]
        result.attrs.update(definition="mean D(z+0.5 beta_ENSO,C)-D(z-0.5 beta_ENSO,C)",
            direction="normalized training latents regressed onto detrended, monthly-anomalized standardized training Niño3.4",
            sample_years=sample_years, samples=len(keep), low_co2=early, high_co2=late,
            checkpoint=str(Path(config["checkpoint"]).resolve()), device=str(model.device),
            caveat="local finite-difference decoder expression, not a forecast-variance estimate")
        save_dataset(result, path)
    result = xr.load_dataset(path)
    rows = []
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for name in ("t2m", "sst", "mtpr"):
        field = result[f"gain_{name}"]
        rms = np.sqrt(weighted_mean(field**2))
        ratio = rms/rms.isel(co2_ppm=0)
        axes[0].plot(field.co2_ppm, ratio, marker="o", label=name)
        for ppm in result.co2_ppm.values:
            current = field.sel(co2_ppm=ppm)
            rows.append(dict(variable=name, co2_ppm=ppm, spatial_rms=float(rms.sel(co2_ppm=ppm)),
                rms_ratio=float(ratio.sel(co2_ppm=ppm)),
                pattern_cosine=float(weighted_uncentered_pattern_correlation(current, field.isel(co2_ppm=0))),
                nino34_gain=float(regional_mean(current, lat_bounds=(-5,5), lon_bounds=(190,240))) if name == "sst" else np.nan))
        # Two-map comparison shows how the SAME latent ENSO direction is decoded.
        first, unit = display(name, field.isel(co2_ppm=0, drop=True))
        last, _ = display(name, field.isel(co2_ppm=-1, drop=True))
        f = plot_map_comparison(first, last, path=output/f"{name}_enso_expression", units=unit,
            title=f"{name}: fixed ENSO perturbation · {result.co2_ppm.values[0]:.0f} vs {result.co2_ppm.values[-1]:.0f} ppm",
            labels=("Low CO₂ decoder", "High CO₂ decoder", "High − low"))
        plt.close(f)
        response, unit = display(name, result[f"response_{name}"])
        if config.get("reference_kind") == "reanalysis":
            f = plot_response_map(response, path=output/f"{name}_same_latent_response", units=unit,
                title=f"{name}: same-latent CO₂ response · {result.attrs['low_co2']:.0f} to {result.attrs['high_co2']:.0f} ppm\n"
                      "Decoder-only counterfactual; no future reanalysis reference")
        else:
            reference = xr.load_dataset(Path(output).parent/"reference"/f"{name}.nc").forced_response
            reference, unit = display(name, reference)
            f = plot_map_comparison(reference, response, path=output/f"{name}_same_latent_response", units=unit,
                title=f"{name}: same-latent CO₂ response vs held-out E3SM forced response")
        plt.close(f)
    nino = regional_mean(result.gain_sst, lat_bounds=(-5,5), lon_bounds=(190,240))
    axes[1].plot(result.co2_ppm, nino, marker="o")
    axes[0].set(xlabel="CO₂ (ppm)", ylabel="Expression RMS / low-CO₂ RMS", title="Fixed latent ENSO direction")
    axes[1].set(xlabel="CO₂ (ppm)", ylabel="Niño3.4 response (°C)", title="Decoder ENSO gain")
    axes[0].legend(frameon=False)
    for ax in axes: ax.grid(alpha=.2)
    finish(fig, output/"enso_decoder_gains")
    pd.DataFrame(rows).to_csv(output/"gains.csv", index=False)
