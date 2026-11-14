"""ENSO evolution and Niño-box E/C diagnostics, without model inference.

Primary three-way comparison: per-trajectory monthly intercepts and a linear
trend fitted jointly over the SAME analysis interval. This is a retrospective
climate diagnostic, not a hindcast calibration. E3SM-only exact ensemble-mean
removal is retained as a sensitivity. All regressions use raw degree-C indices.
"""
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import xarray as xr

from workflows.analysis.archives import ClimateArchive
from workflows.analysis.climate import internal_variability
from workflows.analysis.indices import regional_mean
from workflows.analysis.spatial import weighted_mean
from workflows.analysis.trajectory_anomalies import TrainingTrajectoryAnomalies
from workflows.analysis.lagged_enso import lagged_seasonal_means, SEASONS
from workflows.analysis.long_rollout import RegressionMoments, save_dataset

log = logging.getLogger(__name__)
OCEAN_VARIABLES = ("sst", "ohc300", "d20", "ssh")
BOXES = {"nino34": ((-5, 5), (190, 240)), "nino4": ((-5, 5), (160, 210)),
         "nino12": ((-10, 0), (270, 280))}


def box_ec(nino4, nino12):
    """Takahashi (2011) eqs. 3--4 SST-box approximations, °C; NOT PCs.

    E = Niño1+2 - 0.5 Niño4; C = 1.7 Niño4 - 0.1 Niño1+2.
    Neither boxes nor outputs are standardized or forced to be orthogonal.
    """
    return nino12-.5*nino4, 1.7*nino4-.1*nino12


def detect_events(indices, *, threshold=.5, peak_years=(1959, 2024), max_lag=12):
    """All contiguous ±threshold episodes of centered 3-month Niño3.4.

    Equal-month means, complete 3-month windows, inclusive threshold ties,
    no minimum duration. One peak per contiguous same-sign episode (earliest
    maximum if tied). Peak E/C use that SAME centered three-month window.
    Boundary-censored episodes are saved but excluded from plotted summaries.
    Never smooth or group across member boundaries. No RONI tropical adjustment.
    """
    if threshold <= 0:
        raise ValueError("Event threshold must be positive")
    rolling = indices[["nino34", "nino4", "nino12"]].rolling(time=3, center=True, min_periods=3).mean()
    e, c = box_ec(rolling.nino4, rolling.nino12)
    dates = pd.DatetimeIndex(indices.time.values)
    rows = []
    for member in indices.member.values:
        value = rolling.nino34.sel(member=member).values
        finite = np.flatnonzero(np.isfinite(value))
        if not finite.size:
            continue
        for sign, phase in ((1, "El Niño"), (-1, "La Niña")):
            active = np.isfinite(value) & (sign*value >= threshold)
            edges = np.diff(np.r_[False, active, False].astype(int))
            for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                peak = start+int(np.argmax(sign*value[start:stop]))
                censored = start == finite[0] or stop-1 == finite[-1]
                # An interior missing window also prevents knowing onset/end.
                censored |= (start > 0 and not np.isfinite(value[start-1])) or (
                    stop < len(value) and not np.isfinite(value[stop]))
                inside = peak_years[0] <= dates[peak].year <= peak_years[1]
                complete = peak >= max_lag and peak+max_lag < len(value)
                rows.append(dict(member=str(member), phase=phase, start=str(dates[start].date()),
                    end=str(dates[stop-1].date()), peak=str(dates[peak].date()), peak_index=peak,
                    duration_months=int(stop-start), peak_nino34=float(value[peak]),
                    E=float(e.sel(member=member).isel(time=peak)),
                    C=float(c.sel(member=member).isel(time=peak)), boundary_censored=bool(censored),
                    complete_lag_window=bool(complete), used=bool(inside and complete and not censored)))
    columns = ["member", "phase", "start", "end", "peak", "peak_index", "duration_months",
               "peak_nino34", "E", "C", "boundary_censored", "complete_lag_window", "used"]
    return pd.DataFrame(rows, columns=columns), rolling.assign(E=e, C=c)


def djf_indices(indices, years, calendar):
    """Day-weight DJF Niño3.4 for the named January years, never standardize."""
    pieces = []
    for year in years:
        months = indices.nino34.sel(time=slice(f"{year-1}-09", f"{year}-08"))
        part = lagged_seasonal_means(months, year, calendar=calendar).sel(season="DJF(0)", drop=True)
        pieces.append(part.expand_dims(winter_year=[year]))
    return xr.concat(pieces, "winter_year").transpose("member", "winter_year")


def fixed_sst_support(archive, years=(1966, 2006)):
    """Ocean cells with finite SST for EVERY training month and source member."""
    support = np.ones((len(archive.lat), len(archive.lon)), dtype=bool)
    use = np.flatnonzero((archive.time.dt.year >= years[0]) & (archive.time.dt.year <= years[1]))
    if len(use) != 12*(years[1]-years[0]+1):
        raise ValueError("Incomplete SST-support training interval")
    for block in archive.blocks(use, 12):
        support &= np.isfinite(archive.read("sst", block)).all(("member", "time")).values
    return xr.DataArray(support, dims=("lat", "lon"), coords=dict(lat=archive.lat, lon=archive.lon))


def extract_source(archive, config, spec, root):
    """Persist small monthly indices/latitude means and streamed seasonal maps."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    years = np.arange(config["winter_years"][0], config["winter_years"][1]+1)
    anomaly_fit = TrainingTrajectoryAnomalies(archive, config["analysis_years"], root/"anomaly_fits", 12)
    methods = ["trajectory"] + (["ensemble_residual"] if len(archive.members) > 1 else [])
    shape = (len(archive.lat), len(archive.lon))
    selection = np.flatnonzero((archive.time.dt.year >= config["analysis_years"][0]) &
                               (archive.time.dt.year <= config["analysis_years"][1]))
    if len(selection) != 12*(config["analysis_years"][1]-config["analysis_years"][0]+1):
        raise ValueError("Incomplete common analysis period")
    indices_path = root/"indices.nc"
    index_datasets = xr.load_dataset(indices_path) if indices_path.exists() else None
    for variable in ["sst", *config["map_variables"]]:
        series_path = root/f"monthly_{variable}.nc"
        map_path = root/f"seasonal_{variable}.nc"
        if series_path.exists() and (variable == "sst" or map_path.exists()):
            if variable != "sst" or indices_path.exists():
                continue
        # Materialize only one variable's small coefficient maps at a time.
        anomaly_fit.coefficients(variable)
        moments = {m: [RegressionMoments(1, shape) for _ in SEASONS] for m in methods}
        boxes, monthly, previous = [], [], None
        winters = None if variable == "sst" else {
            m: djf_indices(index_datasets.sel(anomaly_method=m), years, spec["calendar"]) for m in methods}
        for k, block in enumerate(archive.blocks(selection, 12)):
            raw = archive.read(variable, block)
            transformed = {"trajectory": anomaly_fit.transform(raw).reset_coords("month", drop=True)}
            if "ensemble_residual" in methods:
                transformed["ensemble_residual"] = internal_variability(raw)
            fields = xr.concat([transformed[m] for m in methods], dim=xr.IndexVariable("anomaly_method", methods))
            fields = fields.transpose("anomaly_method", "member", "time", "lat", "lon")
            eq = fields.sel(lat=slice(-5, 5))
            # Fixed support prevents land/island zeros diluting zonal averages.
            weights = np.cos(np.deg2rad(eq.lat))
            strip = eq.weighted(weights).mean("lat", skipna=True)
            strip.attrs.update(units=archive.units[variable], latitude_band="5S--5N")
            local = xr.Dataset(dict(equatorial=strip))
            if variable == "sst":
                box_ds = xr.Dataset({name: regional_mean(fields, lat_bounds=lat, lon_bounds=lon)
                                     for name, (lat, lon) in BOXES.items()})
                boxes.append(box_ds)
            elif variable == "tau_x":
                local["regional"] = regional_mean(fields, lat_bounds=(-5, 5), lon_bounds=(160, 210))
                local.regional.attrs["region"] = "western/central Pacific: 5S--5N, 160E--150W"
            else:
                local["regional"] = regional_mean(fields, lat_bounds=(-5, 5), lon_bounds=(120, 280))
                local.regional.attrs["region"] = "equatorial Pacific: 5S--5N, 120E--80W"
            monthly.append(local)
            year = int(raw.time.dt.year.values[0])
            if variable != "sst" and previous is not None and year in years:
                season_input = xr.concat([previous, fields.isel(time=slice(0, 8))], "time")
                seasonal = lagged_seasonal_means(season_input, year, calendar=spec["calendar"])
                for method in methods:
                    x = winters[method].sel(winter_year=year).values[None, :, None]
                    for j, reducer in enumerate(moments[method]):
                        y = seasonal.sel(anomaly_method=method).isel(season=j).transpose("member", "lat", "lon").values
                        reducer.update(y[:, None], x)
            previous = fields.isel(time=slice(8, 12)).copy()
        series = xr.concat(monthly, "time")
        series.attrs.update(variable=variable, source_units=archive.units[variable],
                            description="monthly physical-unit anomalies; no spatial or index standardization")
        save_dataset(series, series_path)
        if variable == "sst":
            index_datasets = xr.concat(boxes, "time")
            e, c = box_ec(index_datasets.nino4, index_datasets.nino12)
            index_datasets = index_datasets.assign(E=e, C=c)
            for name in index_datasets: index_datasets[name].attrs["units"] = "degC"
            index_datasets.attrs.update(description="raw SST-box anomalies and Takahashi Niño-box E/C approximations",
                                       analysis_years=str(config["analysis_years"]))
            save_dataset(index_datasets, indices_path)
        else:
            products = xr.Dataset(dict(
                slope=(("anomaly_method", "season", "lat", "lon"),
                    np.stack([[r.result()[0] for r in moments[m]] for m in methods])),
                count=(("anomaly_method", "season", "lat", "lon"),
                    np.stack([[r.values[0, 0] for r in moments[m]] for m in methods]).astype("int32"))),
                coords=dict(anomaly_method=methods, season=list(SEASONS), lat=archive.lat, lon=archive.lon))
            products.slope.attrs["units"] = f"{archive.units[variable]} per degC"
            products.attrs.update(calendar=spec["calendar"], variable=variable,
                                  winter_years=str(config["winter_years"]), predictor="raw day-weighted DJF Niño3.4; OLS with intercept")
            save_dataset(products, map_path)
        log.info("%s: finished %s", root.name, variable)
        anomaly_fit.coefficients.cache_clear()
    return index_datasets


def lagged_products(root, config, spec):
    """Latitude-mean regressions and event composites; no large field reads."""
    root = Path(root)
    indices = xr.load_dataset(root/"indices.nc")
    years = np.arange(config["winter_years"][0], config["winter_years"][1]+1)
    lags = np.arange(-config["lag_months"], config["lag_months"]+1)
    events, smoothed = detect_events(indices.sel(anomaly_method="trajectory", drop=True),
        threshold=config["threshold_degC"], peak_years=tuple(config["winter_years"]), max_lag=config["lag_months"])
    events.to_csv(root/"events.csv", index=False)
    save_dataset(smoothed, root/"smoothed_indices.nc")
    used = events.loc[events.used.astype(bool)]
    dates = pd.DatetimeIndex(indices.time.values)
    anchors = np.array([int(np.flatnonzero((dates.year == y) & (dates.month == 1))[0]) for y in years])
    summaries, curves = [], []
    for variable in ["sst", *config["map_variables"]]:
        monthly = xr.load_dataset(root/f"monthly_{variable}.nc")
        slope, counts = [], []
        for method in indices.anomaly_method.values:
            x = djf_indices(indices.sel(anomaly_method=method), years, spec["calendar"]).values[None]
            y = monthly.equatorial.sel(anomaly_method=method).transpose("member", "time", "lon").values
            lag_maps, lag_counts = [], []
            for lag in lags:
                reducer = RegressionMoments(1, (1, y.shape[-1]))
                reducer.update(y[:, anchors+lag, None, :], x)
                lag_maps.append(reducer.result()[0, 0])
                lag_counts.append(reducer.values[0, 0, 0])
            slope.append(lag_maps)
            counts.append(lag_counts)
        y = monthly.equatorial.sel(anomaly_method="trajectory").transpose("member", "time", "lon").values
        members = list(map(str, monthly.member.values))
        composites, composite_counts = [], []
        for phase in ("El Niño", "La Niña"):
            samples = [y[members.index(row.member), row.peak_index+lags]
                       for row in used[used.phase == phase].itertuples()]
            if samples:
                samples = np.stack(samples)
                n = np.isfinite(samples).sum(0)
                mean = np.divide(np.nansum(samples, axis=0), n,
                                 out=np.full(n.shape, np.nan), where=n > 0)
            else:
                n = np.zeros((len(lags), y.shape[-1]), dtype=int)
                mean = np.full(n.shape, np.nan)
            composites.append(mean)
            composite_counts.append(n)
        ds = xr.Dataset(dict(
            slope=(("anomaly_method", "lag", "lon"), np.asarray(slope)),
            count=(("anomaly_method", "lag", "lon"), np.asarray(counts).astype("int32")),
            composite=(("phase", "lag", "lon"), np.stack(composites)),
            composite_count=(("phase", "lag", "lon"), np.stack(composite_counts))),
            coords=dict(anomaly_method=indices.anomaly_method, lag=lags, lon=monthly.lon, phase=["El Niño", "La Niña"]),
            attrs=dict(variable=variable, source_units=monthly.attrs["source_units"],
                lag_definition="positive = field follows January of DJF for regression; follows event-peak month for composites",
                latitude_band="5S--5N", composite_definition="equal weight per complete, uncensored threshold-crossing episode"))
        ds.slope.attrs["units"] = monthly.attrs["source_units"]+" per degC"
        ds.composite.attrs["units"] = monthly.attrs["source_units"]
        save_dataset(ds, root/f"hovmoller_{variable}.nc")
        if variable != "sst":
            field = monthly.regional.sel(anomaly_method="trajectory").transpose("member", "time").values
            predictor = indices.nino34.sel(anomaly_method="trajectory").transpose("member", "time").values
            for lag in range(-18, 19):
                # No flattening across member boundaries before slicing lags.
                a = predictor[:, max(0, -lag):min(len(dates), len(dates)-lag)].ravel()
                b = field[:, max(0, lag):min(len(dates), len(dates)+lag)].ravel()
                good = np.isfinite(a) & np.isfinite(b)
                curves.append(dict(variable=variable, lag=lag, count=int(good.sum()),
                    correlation=float(np.corrcoef(a[good], b[good])[0, 1])))
    for phase in ("El Niño", "La Niña"):
        group = used[used.phase == phase]
        summaries.append(dict(phase=phase, events=len(group),
            events_per_decade_per_member=10*len(group)/(len(years)*indices.sizes["member"]),
            median_duration_months=float(group.duration_months.median()),
            mean_abs_peak_nino34=float(group.peak_nino34.abs().mean()),
            mean_E=float(group.E.mean()), mean_C=float(group.C.mean()),
            ec_correlation=float(group[["E", "C"]].corr().iloc[0, 1]) if len(group) > 2 else np.nan))
    pd.DataFrame(summaries).to_csv(root/"event_summary.csv", index=False)
    pd.DataFrame(curves).to_csv(root/"regional_lag_correlations.csv", index=False)
    return events
