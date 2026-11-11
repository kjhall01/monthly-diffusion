"""Seasonal fields regressed on the SAME winter's raw DJF Niño3.4 (°C).

SON(-1) is Sep--Nov of Y-1; DJF(0) is Dec(Y-1)--Feb(Y); MAM(0) and
JJA(0) follow in Y. Exact monthly ensemble-mean removal and sqrt(N/(N-1))
are applied independently to each ensemble, BEFORE seasonal averaging.
There is no additional climatology removal, detrending, or standardization.
The slope has native field units per degree Celsius of winter Niño3.4.
"""
from pathlib import Path
import logging
import numpy as np
import xarray as xr

from .climate import internal_variability
from .indices import enso_index
from .long_rollout import RegressionMoments, save_dataset

log = logging.getLogger(__name__)
SEASONS = ("SON(-1)", "DJF(0)", "MAM(0)", "JJA(0)")


def lagged_seasonal_means(monthly, anchor_year, *, calendar="noleap"):
    """Day-weight complete Sep(Y-1)--Aug(Y) seasons, preserving missing data.

    E3SM's canonical Gregorian timestamps are month LABELS, not its native
    calendar. Explicit 365-day weights prevent leap-year February reweighting.
    Use calendar='standard' for reanalysis; there the timestamp calendar gives
    actual Gregorian month lengths. The default remains native E3SM no-leap.
    """
    if calendar not in ("noleap", "365_day", "standard", "proleptic_gregorian"):
        raise ValueError("Choose the documented noleap or Gregorian calendar")
    keys = monthly.time.dt.year.values*12 + monthly.time.dt.month.values
    expected = np.arange((anchor_year-1)*12+9, anchor_year*12+9)
    if not np.array_equal(keys, expected):
        raise ValueError("Require exactly Sep(Y-1)--Aug(Y), ordered without gaps")
    days = np.array([30, 31, 30, 31, 31, 28, 31, 30, 31, 30, 31, 31])
    if calendar in ("standard", "proleptic_gregorian"):
        days = monthly.time.dt.days_in_month.values
    seasons = []
    for j in range(4):
        part = monthly.isel(time=slice(3*j, 3*j+3))
        weights = xr.DataArray(days[3*j:3*j+3], dims="time", coords={"time": part.time})
        seasons.append((part*weights).sum("time", skipna=False)/weights.sum())
    result = xr.concat(seasons, dim=xr.IndexVariable("season", list(SEASONS)))
    result.attrs.update(monthly.attrs)
    result.attrs.update(calendar=calendar, season_year=anchor_year)
    return result


def lagged_enso_products(archive, years, variables, output, *, calendar="noleap"):
    """Stream one Sep--Aug year at a time; return small native-unit map products.

    Cases pool member × winter with equal weights. Every season uses the same
    DJF predictor and the same years. Grid cells must have all three months
    and all members available; counts are retained. No whole-rollout field is
    held in memory. Already completed products can be reused by the caller.
    """
    output = Path(output)
    years = np.asarray(years, dtype=int)
    if len(years) < 2 or np.any(np.diff(years) != 1):
        raise ValueError("Require at least two consecutive anchor winters")
    if len(archive.members) < 2:
        raise ValueError("Ensemble-mean removal needs at least two members")
    shape = (len(archive.lat), len(archive.lon))
    regression = {v: [RegressionMoments(1, shape) for _ in SEASONS] for v in variables}
    ilat = np.flatnonzero((archive.lat >= -5) & (archive.lat <= 5))
    ilon = np.flatnonzero((archive.lon >= 190) & (archive.lon <= 240))
    if not len(ilat) or not len(ilon):
        raise ValueError("Archive does not cover Niño3.4 in 0--360 longitude")
    indices = []
    for n, year in enumerate(years):
        keys = np.arange((year-1)*12+9, year*12+9)
        locations = np.searchsorted(archive.keys, keys)
        if np.any(locations >= len(archive.keys)) or not np.array_equal(archive.keys[locations], keys):
            raise ValueError(f"Incomplete Sep--Aug season sequence for DJF {year}")
        sst = archive.read("sst", locations, lat=slice(ilat[0], ilat[-1]+1),
                           lon=slice(ilon[0], ilon[-1]+1))
        monthly_index = enso_index(internal_variability(sst).to_dataset(), remove_monthly=False)
        winter_index = lagged_seasonal_means(monthly_index, year, calendar=calendar).sel(
            season="DJF(0)", drop=True)
        if not np.isfinite(winter_index).all():
            raise ValueError(f"Nonfinite DJF {year} Niño3.4 index")
        indices.append(winter_index.expand_dims(winter_year=[year]))
        # RegressionMoments takes [mode,member,time]. The index is deliberately
        # NOT standardized: slopes represent a +1 degree-C Niño3.4 event.
        predictor = winter_index.values[None, :, None]
        for variable in variables:
            fields = archive.read(variable, locations)
            seasons = lagged_seasonal_means(internal_variability(fields), year, calendar=calendar)
            for j, reducer in enumerate(regression[variable]):
                value = seasons.isel(season=j).transpose("member", "lat", "lon").values
                reducer.update(value[:, None], predictor)
        if n == 0 or (n+1) % 25 == 0 or n+1 == len(years):
            log.info("%s: %d/%d winters (DJF %d)", output.name, n+1, len(years), year)
    index = xr.concat(indices, dim="winter_year").transpose("member", "winter_year")
    if float(index.std()) <= 0:
        raise ValueError("Cannot regress on a constant winter ENSO index")
    result = xr.Dataset(coords=dict(season=list(SEASONS), lat=archive.lat, lon=archive.lon))
    for variable in variables:
        result[variable] = (("season", "lat", "lon"),
                            np.stack([x.result()[0] for x in regression[variable]]))
        result[variable].attrs.update(units=f"{archive.units[variable]} per degC",
            source_units=archive.units[variable], long_name=f"{variable} response per +1°C DJF Niño3.4")
        result[f"count_{variable}"] = (("season", "lat", "lon"),
            np.stack([x.values[0, 0] for x in regression[variable]]).astype("int32"))
    result.attrs.update(calendar=calendar, winter_start=int(years[0]), winter_end=int(years[-1]),
        members=len(archive.members), member_ids=",".join(map(str, archive.members)),
        regression="OLS with intercept; pooled member/winter; raw °C index, NOT standardized",
        anomalies="sqrt(N/(N-1))*(field - exact monthly ensemble mean); no further detrend or climatology removal",
        seasons="SON(Y-1), Dec(Y-1)-Feb(Y), MAM(Y), JJA(Y); day-weighted native noleap calendar")
    index.attrs.update(units="degC", standardization="none", **result.attrs)
    save_dataset(index.rename("nino34_djf").to_dataset(), output/"winter_indices.nc")
    save_dataset(result, output/"regressions.nc")
    return result


def plot_lagged_enso(reference, model, variable, path):
    """3×4 Pacific-centered Robinson maps; one shared signed scale per figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MaxNLocator
    import cartopy.crs as ccrs
    from cartopy.util import add_cyclic_point

    reference, model = xr.align(reference, model, join="exact")
    factor, title, units, cmap = {
        "z500": (1/9.80665, "500-hPa Geopotential Height", "m per °C Niño3.4", "RdBu_r"),
        "mtpr": (86400., "Precipitation", "mm day⁻¹ per °C Niño3.4", "BrBG"),
    }[variable]
    arrays = [model*factor, reference*factor, (model-reference)*factor]
    values = np.concatenate([x.values.ravel() for x in arrays])
    # A rounded 99.5th-percentile bound keeps isolated extremes from washing
    # out the maps. Saturation is visible at both colorbar ends and recorded.
    bound = max(float(np.nanpercentile(np.abs(values), 99.5)), 1.e-12)
    step = 10**np.floor(np.log10(bound))/2
    limit = float(np.ceil(bound/step)*step)
    norm = Normalize(-limit, limit)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "axes.titleweight": "normal"}):
        fig, axes = plt.subplots(3, 4, figsize=(16, 8.1),
            subplot_kw=dict(projection=ccrs.Robinson(central_longitude=180)))
        fig.subplots_adjust(left=.065, right=.99, bottom=.12, top=.89, hspace=.13, wspace=.055)
        for row, data in enumerate(arrays):
            for col, season in enumerate(SEASONS):
                ax = axes[row, col]
                field = data.sel(season=season)
                z, lon = add_cyclic_point(field.values, coord=field.lon)
                image = ax.pcolormesh(lon, field.lat, z, transform=ccrs.PlateCarree(),
                    cmap=cmap, norm=norm, shading="auto", rasterized=True)
                ax.set_global()
                ax.coastlines(resolution="110m", linewidth=.45, color=".22")
                ax.spines["geo"].set_linewidth(.55)
                if row == 0:
                    ax.set_title(season.replace("-1", "−1"), fontsize=13, pad=10)
        fig.canvas.draw()
        for row, label in enumerate(("Monthly Diffusion v1.0", "E3SMv2", "MD − E3SMv2")):
            box = axes[row, 0].get_position()
            fig.text(.035, (box.y0+box.y1)/2, label, rotation=90,
                     ha="center", va="center", fontsize=12)
        cax = fig.add_axes([.32, .065, .43, .019])
        extend = "both" if np.any(np.abs(values[np.isfinite(values)]) > limit) else "neither"
        bar = fig.colorbar(image, cax=cax, orientation="horizontal", extend=extend)
        bar.set_label(units, labelpad=5)
        bar.locator = MaxNLocator(nbins=7)
        bar.update_ticks()
        bar.outline.set_linewidth(.5)
        fig.suptitle(f"{title} · Lagged ENSO Teleconnections", fontsize=16, y=.98)
        fig.text(.53, .933, f"DJF Niño3.4 anchor · {model.attrs['winter_start']}–{model.attrs['winter_end']} · regression per +1°C",
                 ha="center", fontsize=10, color=".3")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path.with_suffix(".png"), dpi=300, facecolor="white", bbox_inches="tight")
        fig.savefig(path.with_suffix(".pdf"), dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig)
    return dict(display_factor=factor, display_units=units, color_limit=limit,
                limit_percentile=99.5, colorbar_extend=extend, cmap=cmap,
                saturated_fraction=float(np.mean(np.abs(values[np.isfinite(values)]) > limit)))
