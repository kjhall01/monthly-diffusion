"""Compute forced-response maps one variable at a time."""

import xarray as xr
from . import annual_means
from .plotting import plot_map_comparison


def _forced_map(fields, variable, early, late):
    """Compute one variable's late-minus-early map without loading its history."""
    early_mean = annual_means(
        fields[variable].sel(time=slice(str(early[0]), str(early[1])))
    ).mean("year")
    late_mean = annual_means(
        fields[variable].sel(time=slice(str(late[0]), str(late[1])))
    ).mean("year")
    return (late_mean - early_mean).compute()


def run_forced_only(generated, reference_paths, output, early, late):
    """Low-memory forced-response maps for the standard poster variables."""
    output.mkdir(parents=True, exist_ok=True)
    variables = ("t2m", "sst", "mtpr")
    reference_maps = {name: [] for name in variables}
    for path in reference_paths:
        reference = xr.open_dataset(path) if path.is_file() else xr.open_zarr(path)
        for name in variables:
            reference_maps[name].append(_forced_map(reference, name, early, late))
    for name in variables:
        reference_response = xr.concat(reference_maps[name], dim="member").mean(
            "member"
        )
        model_response = _forced_map(generated, name, early, late).mean("member")
        reference_response.to_netcdf(output / f"reference_forced_{name}.nc")
        model_response.to_netcdf(output / f"model_forced_{name}.nc")
        plot_map_comparison(
            reference_response,
            model_response,
            path=output / f"forced_{name}",
            title=name,
        )
    (output / "forced_response_metadata.txt").write_text(
        f"reference_members={len(reference_paths)}\n"
        f"early={early[0]}-{early[1]}\nlate={late[0]}-{late[1]}\n"
        "method=member-wise annual climatology maps; ensemble mean taken after map computation\n"
    )
