"""Small checks for the new read-only comparison's coastline definition."""
import importlib.util
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location("comparability_audit",
    Path(__file__).parents[1]/"scripts/audit_reanalysis_comparability.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_coast_wraps_longitude_not_latitude():
    wet = np.ones((4, 5), dtype=bool)
    wet[0, 0] = False
    expected = np.zeros_like(wet)
    expected[0, 1] = expected[0, -1] = expected[1, 0] = True
    np.testing.assert_array_equal(audit.coast_mask(wet), expected)


def test_coast_excludes_land_and_diagonal_neighbors():
    wet = np.ones((5, 5), dtype=bool)
    wet[2, 2] = False
    expected = np.zeros_like(wet)
    expected[1, 2] = expected[3, 2] = expected[2, 1] = expected[2, 3] = True
    np.testing.assert_array_equal(audit.coast_mask(wet), expected)


def test_no_coast_on_all_ocean_or_all_land():
    for value in (False, True):
        assert not audit.coast_mask(np.full((4, 5), value)).any()
