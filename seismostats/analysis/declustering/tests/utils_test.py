from seismostats.analysis.declustering.utils import haversine
from numpy.testing import assert_array_almost_equal
import numpy as np
import pytest


@pytest.mark.parametrize(["lon1", "lat1", "lon2", "lat2", "expected"], [
    (np.array([40.0]), np.array([10.0]), 40.0, 10.0, np.array([[0.0]])),
    (np.array([5, 10]), np.array([15, 20]), 0.0, 0.0,
        np.array([[1756.18897213],
                  [2476.2596371]])),
    (np.array([0, 10, 20, 30, 40]), np.array([0, 1, 2, 3, 4]), 42.2, 3.5,
     np.array([[4705.68408325],
               [3588.26704913],
               [2471.30068294],
               [1355.57263248],
               [250.36456431]]))
])
def test_haversine(lon1, lat1, lon2, lat2, expected):
    result = haversine(lon1, lat1, lon2, lat2)
    assert_array_almost_equal(result, expected)