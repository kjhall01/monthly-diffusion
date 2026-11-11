"""Niño-box E/C proxies, not principal components or fitted EOF rotations.

Takahashi et al. (2011), doi:10.1029/2011GL047364, equations (3)--(4):
E = Niño1+2 - 0.5 Niño4; C = 1.7 Niño4 - 0.1 Niño1+2.
Inputs are SST anomalies in degrees Celsius, NOT standardized box indices.
We retain the units of these fixed linear SST combinations (degC); these are
approximations to the paper's standardized-PC E/C indices, not identical to
them, and are not guaranteed uncorrelated in another climate model.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from .indices import regional_mean


def nino_boxes(sst):
    """Cos-latitude weighted means, skipping land; no anomaly removal here."""
    return xr.concat(
        [
            regional_mean(sst, lat_bounds=(-5, 5), lon_bounds=(160, 210)),
            regional_mean(sst, lat_bounds=(-10, 0), lon_bounds=(270, 280)),
        ],
        dim=xr.IndexVariable("box", ["Nino4", "Nino1+2"]),
    ).transpose("time", ..., "box")


def box_ec(anomalies):
    """Last input axis: [Niño4, Niño1+2]; last output axis: [E, C]."""
    a = np.asarray(anomalies, dtype=float)
    if a.shape[-1] != 2:
        raise ValueError("Expected last axis [Nino4, Nino1+2]")
    return np.stack(
        (a[..., 1] - 0.5 * a[..., 0], 1.7 * a[..., 0] - 0.1 * a[..., 1]), axis=-1
    )
