"""Streaming physical-space long-rollout diagnostics, using package definitions.

Indices use exact ensemble-mean subtraction, sqrt(N/(N-1)), and no subsequent
monthly-climatology subtraction. Regressions are native units per independently
standardized +1 SD event, fitted over pooled member/time samples with intercept.
"""
from pathlib import Path
import json
import logging
import resource
import sys
import numpy as np
import xarray as xr
from workflows.analysis.archives import ClimateArchive
from workflows.analysis.climate import internal_variability
from workflows.analysis.indices import climate_indices, pdo_loading
from workflows.analysis.spatial import weighted_mean

log = logging.getLogger(__name__)


def save_dataset(ds, path):
    """Atomic NetCDF product; a killed process never leaves a valid cache file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = ds.copy()
    for obj in [ds, *ds.variables.values()]:
        obj.attrs = {k: int(v) if isinstance(v, (bool, np.bool_)) else v for k, v in obj.attrs.items()}
    partial = path.with_suffix(".partial.nc")
    ds.to_netcdf(partial, encoding={n: dict(zlib=True, complevel=2) for n in ds.data_vars})
    partial.replace(path)


class RegressionMoments:
    """Sufficient statistics for masked OLS against already-standardized indices."""
    def __init__(self, modes, shape):
        self.values = np.zeros((6, modes, *shape), dtype=np.float64)

    def update(self, fields, index):
        # field [member,time,lat,lon], index [mode,member,time]. No spatial
        # broadcasting across all modes: accumulate one small map at a time.
        y = fields.reshape(-1, *fields.shape[-2:])
        for j, predictor in enumerate(index.reshape(index.shape[0], -1)):
            x = predictor[:, None, None]
            valid = np.isfinite(y) & np.isfinite(x)
            yy, xx = np.where(valid, y, 0.), np.where(valid, x, 0.)
            self.values[:, j] += np.stack((valid.sum(0), xx.sum(0), yy.sum(0),
                (xx*xx).sum(0), (xx*yy).sum(0), (yy*yy).sum(0)))

    def result(self):
        n, sx, sy, sxx, sxy, _ = self.values
        with np.errstate(invalid="ignore", divide="ignore"):
            denominator = sxx-sx*sx/n
            slope = (sxy-sx*sy/n)/denominator
        return np.where((n > 1) & (denominator > 0), slope, np.nan)


def fit_pdo(reference, training, block_months, output, anomalies=None):
    path = output/"reference_pdo_loading.nc"
    if path.exists():
        return xr.load_dataset(path)["PDO_loading"]
    select = np.flatnonzero((reference.time.dt.year >= training[0]) & (reference.time.dt.year <= training[1]))
    ilat = np.flatnonzero((reference.lat >= 20) & (reference.lat <= 70))
    ilon = np.flatnonzero((reference.lon >= 110) & (reference.lon <= 260))
    pieces = []
    for block in reference.blocks(select, block_months):
        sst = reference.read("sst", block, lat=slice(ilat[0], ilat[-1]+1), lon=slice(ilon[0], ilon[-1]+1))
        pieces.append((internal_variability(sst) if anomalies is None else anomalies.transform(sst)).astype("float32"))
    log.info("Fitting fixed PDO EOF on reference training-period North Pacific SST only")
    basis = pdo_loading(xr.concat(pieces, "time").to_dataset(), remove_monthly=False)
    save_dataset(basis.to_dataset(), path)
    return basis


def extract_indices(archive, selection, basis, output, block_months, anomalies=None):
    path = output/"indices.nc"
    if path.exists():
        return xr.load_dataset(path)
    pieces = []
    for k, block in enumerate(archive.blocks(selection, block_months)):
        values = []
        for variable in ("sst", "mslp", "u50"):
            field = archive.read(variable, block)
            residual = internal_variability(field) if anomalies is None else anomalies.transform(field)
            values.append(climate_indices(residual.to_dataset(), remove_monthly=False, pdo_loading=basis))
        pieces.append(xr.merge(values, compat="no_conflicts"))
        if k % 25 == 0:
            log.info("%s indices: %d/%d months", output.name, min((k+1)*block_months, len(selection)), len(selection))
    result = xr.concat(pieces, "time")
    result.attrs["anomaly_definition"] = (
        "sqrt(N/(N-1))*(field - exact same-month ensemble mean); no monthly re-centering" if anomalies is None else
        "per-trajectory training-only monthly-intercept and linear-trend residual; no ensemble subtraction or finite-N rescaling")
    save_dataset(result, path)
    return result


def variable_products(archive, selection, indices, variable, periods, output, block_months, anomalies=None):
    path = output/f"{variable}.nc"
    if path.exists():
        return path
    modes = list(indices.data_vars)
    z = ((indices-indices.mean(("member", "time")))/indices.std(("member", "time"))).to_array("mode").transpose("mode", "member", "time")
    shape = (len(archive.lat), len(archive.lon))
    sums = np.zeros((len(periods), *shape))
    counts = np.zeros_like(sums)
    regression = RegressionMoments(len(modes), shape)
    globals_ = []
    offset = 0
    for k, block in enumerate(archive.blocks(selection, block_months)):
        field = archive.read(variable, block)
        globals_.append(weighted_mean(field))
        # Require identical member support per date/cell for the reference
        # mean and IV, never replace missing ocean cells by zeros.
        mean = field.mean("member", skipna=False).values
        years = field.time.dt.year.values
        for i, (lo, hi) in enumerate(periods.values()):
            selected = mean[(years >= lo) & (years <= hi)]
            counts[i] += np.isfinite(selected).sum(0)
            sums[i] += np.nansum(selected, axis=0)
        # Global trajectories span the full archive; index regressions may be
        # restricted to the training period, independently of those trajectories.
        keep = np.isin(field.time.values, indices.time.values)
        if keep.any():
            matched = field.isel(time=np.flatnonzero(keep))
            residual = internal_variability(matched) if anomalies is None else anomalies.transform(matched)
            regression.update(residual.values, z.sel(time=matched.time).values)
        offset += len(block)
        if k % 50 == 0:
            log.info("%s %s: %d/%d months", output.name, variable, offset, len(selection))
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums/counts
    coords = dict(lat=archive.lat, lon=archive.lon)
    maps = xr.DataArray(means, dims=("period", "lat", "lon"), coords=dict(period=list(periods), **coords))
    result = xr.Dataset(dict(
        climatology=maps,
        regression=(("mode", "lat", "lon"), regression.result()),
        regression_count=(("mode", "lat", "lon"), regression.values[0]),
        global_mean=xr.concat(globals_, "time")), coords=dict(mode=modes))
    if "early" in periods and "late" in periods:
        result["forced_response"] = maps.sel(period="late", drop=True)-maps.sel(period="early", drop=True)
    for name in ("climatology", "forced_response", "regression", "global_mean"):
        if name in result:
            result[name].attrs["units"] = field.attrs["units"]
    result.attrs.update(variable=variable, periods=json.dumps(periods),
        regression="pooled OLS with intercept; separate ensemble index standardization; native units per +1 SD",
        climatology="equal-month means over complete calendar years")
    result.attrs.update(anomaly_definition=indices.attrs.get("anomaly_definition", "ensemble residual"),
                        regression_start=str(indices.time.values[0]), regression_end=str(indices.time.values[-1]))
    save_dataset(result, path)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform == "darwin" else 1024)
    log.info("Saved %s (process peak RSS %.0f MiB)", path, rss)
    return path


def run_physical_diagnostics(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    forecast = ClimateArchive([config["forecast"]], forecast=True)
    # The model's fixed TRAINING SST mask governs both sides of comparison.
    # Reference-specific missing cells inside that domain remain missing.
    reference = ClimateArchive(config["references"], ocean_support=forecast.ocean_support,
                               ocean_variables=forecast.ocean_variables)
    reanalysis = config.get("reference_kind", "held_out_ensemble") == "reanalysis"
    if reanalysis and len(reference.members) != 1:
        raise ValueError("Reanalysis comparisons require one observed trajectory")
    if not reanalysis and (config.get("training_member", "r1i1p1f1") in reference.members or len(reference.members) < 2):
        raise ValueError("Use at least two HELD-OUT reference members")
    if not np.array_equal(forecast.lat, reference.lat) or not np.array_equal(forecast.lon, reference.lon):
        raise ValueError("Reference and model must share the same grid")
    from workflows.data.dataloader import validate_units
    from monthly_diffusion.preprocessing import Variable
    validate_units(xr.Dataset({v: xr.DataArray(attrs={"units": reference.units.get(v, "")}) for v in config["variables"]}),
                   [Variable(v, "", forecast.units[v]) for v in config["variables"]])
    periods = {k: config[k] for k in ("early", "training", "late")}
    selections = []
    for name, archive in (("reference", reference), ("model", forecast)):
        start, end = config.get("reference_evaluation" if name == "reference" else "evaluation", config.get("evaluation", [1851, 2100]))
        keep = np.flatnonzero((archive.time.dt.year >= start) & (archive.time.dt.year <= end))
        expected = np.arange(start*12+1, (end+1)*12+1)
        if not np.array_equal(archive.keys[keep], expected):
            raise ValueError("Evaluation requires all twelve months of each requested year")
        required = [periods["training"]] if reanalysis and name == "reference" else periods.values()
        if any(lo < start or hi > end for lo, hi in required):
            raise ValueError("Climatology periods must lie inside evaluation coverage")
        selections.append(keep)
    # Run identity binds caches to the scientific choices, all references and
    # the archived checkpoint, without loading the model or resuming inference.
    identity = dict(config=config, forecast=forecast.record, reference_members=list(reference.members), version=2 if reanalysis else 1)
    marker = output/"configuration.json"
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise ValueError("Diagnostic configuration changed; choose a new output directory")
    marker.write_text(json.dumps(identity, indent=2))
    block_months = config.get("block_months", 12)
    if block_months < 1 or block_months > 24:
        raise ValueError("block_months must be 1..24 to bound memory")
    anomaly_fits = dict(reference=None, model=None)
    if reanalysis:
        if config.get("anomaly_method") != "training_monthly_linear_detrend" or config.get("index_period") != config["training"]:
            raise ValueError("Reanalysis diagnostics require explicit matched training-period anomaly comparisons")
        from workflows.analysis.trajectory_anomalies import TrainingTrajectoryAnomalies
        for name, archive in (("reference", reference), ("model", forecast)):
            anomaly_fits[name] = TrainingTrajectoryAnomalies(archive, periods["training"], output/name/"anomalies", block_months)
    basis = fit_pdo(reference, periods["training"], block_months, output, anomaly_fits["reference"])
    for name, archive, selection in zip(("reference", "model"), (reference, forecast), selections):
        root = output/name
        root.mkdir(exist_ok=True)
        index_selection = selection
        if "index_period" in config:
            lo, hi = config["index_period"]
            index_selection = selection[(archive.time.dt.year.values[selection] >= lo) & (archive.time.dt.year.values[selection] <= hi)]
        indices = extract_indices(archive, index_selection, basis, root, block_months, anomaly_fits[name])
        archive_periods = {"training": periods["training"]} if reanalysis and name == "reference" else periods
        for variable in config["variables"]:
            variable_products(archive, selection, indices, variable, archive_periods, root, block_months, anomaly_fits[name])
    return output
