"""Thin, explicit adapter for monthly-diffusion-data final products.

No regridding, meteorological unit conversion or split selection occurs here.
Source release attributes/checksums are checked when supplied. The explicit
constant-CO2 spin-up helper never changes the prescribed source values.
"""

from pathlib import Path
import hashlib
import numpy as np
import xarray as xr
from .configs import mdv1_variables


def open_canonical_state(path, *, expected_attrs=None):
    path = Path(path).expanduser()
    raw = xr.open_zarr(path, consolidated=True)
    for name, expected in (expected_attrs or {}).items():
        if raw.attrs.get(name) != expected:
            raise ValueError(
                f"Source {name} does not match the requested release: {raw.attrs.get(name)!r}"
            )
    if "member" in raw.dims:
        raise ValueError("Select one source member before training")
    fields = raw.rename({"ohc300_700": "ohc700_minus_ohc300"})
    # Wet, ice-free cells are genuine zeros; missing land stays missing.
    fields = fields.assign(
        sivol=xr.where(raw.sic > 1.0e-6, raw.sic * raw.sit, 0.0)
        .where(raw.sic.notnull())
        .astype("float32")
    )
    fields.sivol.attrs = dict(
        units="m",
        long_name="Sea-ice equivalent thickness",
        derivation="sic * sit; zero on wet ice-free cells; missing on land",
    )
    names = {
        component: [v.name for v in mdv1_variables() if v.component == component]
        for component in ("atmosphere", "ocean")
    }
    for component in names:
        missing = set(names[component]) - set(fields.data_vars)
        if missing:
            raise ValueError(
                f"Missing canonical {component} variables: {sorted(missing)}"
            )
    statics = fields[["phi_s", "lsm"]]
    if "time" in statics.dims:
        # Static fields must really be static, not silently the first dynamic map.
        for name in statics:
            if "time" in statics[name].dims and not bool(
                (statics[name] == statics[name].isel(time=0)).all().compute()
            ):
                raise ValueError(f"{name} varies with time")
        statics = statics.isel(time=0, drop=True)
    for ds in (fields, statics):
        ds.attrs.update(dict(raw.attrs), source_path=str(path.resolve()))
    return fields[names["atmosphere"]], fields[names["ocean"]], statics


def open_co2(path, *, expected_sha256=None):
    path = Path(path).expanduser()
    if expected_sha256:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected_sha256:
            raise ValueError("CO2 file checksum does not match the requested product")
    ds = xr.open_dataset(path)
    if "co2" not in ds or ds.co2.dims != ("time",):
        raise ValueError("Expected an already-prepared monthly co2(time) product")
    result = ds.co2.load()
    result.attrs.update(dict(ds.attrs), source_path=str(path.resolve()))
    return result


def hold_first_co2_back_to(co2, start):
    """Explicit constant prefix for spin-up; original monthly values stay exact.

    This is a caller-requested spin-up assumption, not earlier observed CO2.
    The existing prescribed product is never modified on disk.
    """
    first = np.asarray(co2.time.values)[0].astype("datetime64[M]")
    start = np.datetime64(start, "M")
    if start >= first:
        raise ValueError("Constant prefix must begin before the first prescribed month")
    dates = np.arange(start, first, dtype="datetime64[M]").astype("datetime64[ns]")
    prefix = xr.DataArray(
        np.full(len(dates), float(co2.values[0]), dtype=co2.dtype),
        dims="time",
        coords=dict(time=dates),
        name="co2",
    )
    # Derived products may carry auxiliary year/month coordinates. Keep only
    # the time coordinate here, never feed inconsistent coordinates to concat.
    result = xr.concat([prefix, co2.reset_coords(drop=True)], dim="time")
    result.attrs.update(
        co2.attrs,
        spinup_prefix_start=str(start),
        spinup_prefix_end=str(dates[-1])[:7],
        spinup_prefix_ppm=float(co2.values[0]),
        spinup_prefix="constant first available CO2 value",
    )
    return result
