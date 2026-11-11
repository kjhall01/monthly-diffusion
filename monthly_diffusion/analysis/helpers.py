from __future__ import annotations

from collections.abc import Callable
from typing import Any

import xarray as xr


def apply_unary(x: xr.DataArray | xr.Dataset, func: Callable[..., xr.DataArray], **kwargs) -> xr.DataArray | xr.Dataset:
    if isinstance(x, xr.Dataset):
        return xr.Dataset(
            {name: func(da, **kwargs) for name, da in x.data_vars.items()},
            attrs=dict(x.attrs),
        )
    return func(x, **kwargs)


def apply_binary(
    x: xr.DataArray | xr.Dataset,
    y: xr.DataArray | xr.Dataset,
    func: Callable[..., xr.DataArray],
    **kwargs,
) -> xr.DataArray | xr.Dataset:
    if isinstance(x, xr.Dataset) and isinstance(y, xr.Dataset):
        names = tuple(name for name in x.data_vars if name in y.data_vars)
        if not names:
            raise ValueError("Datasets do not share any data variables.")
        return xr.Dataset(
            {name: func(x[name], y[name], **kwargs) for name in names},
            attrs=dict(x.attrs),
        )
    if isinstance(x, xr.Dataset):
        return xr.Dataset(
            {name: func(da, y, **kwargs) for name, da in x.data_vars.items()},
            attrs=dict(x.attrs),
        )
    if isinstance(y, xr.Dataset):
        return xr.Dataset(
            {name: func(x, da, **kwargs) for name, da in y.data_vars.items()},
            attrs=dict(y.attrs),
        )
    return func(x, y, **kwargs)


def existing_dims(x: xr.DataArray, dims: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dim for dim in dims if dim in x.dims)


def require_dims(x: xr.DataArray | xr.Dataset, dims: tuple[str, ...]) -> None:
    missing = [dim for dim in dims if dim not in x.dims]
    if missing:
        raise ValueError(f"Object is missing required dimensions: {missing}")


def maybe_name(x: Any, name: str) -> Any:
    if hasattr(x, "rename"):
        return x.rename(name)
    return x
