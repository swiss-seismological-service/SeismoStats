import numpy as np
from numpy.testing import assert_array_almost_equal

from seismostats.analysis.declustering import (
    GardnerKnopoffWindow,
    GruenthalWindow,
    UhrhammerWindow,
)


def test_gardner_knopoff_window():
    window = GardnerKnopoffWindow()
    mag = np.array([5.0, 6.6])
    sw_space, sw_time = window(mag)
    assert_array_almost_equal(sw_space, np.array([39.994475, 63.107358]))
    assert_array_almost_equal(sw_time, np.array(
        [143.71430 / 364.75, 891.45618 / 364.75]))


def test_gruenthal_window():
    window = GruenthalWindow()
    mag = np.array([5.0, 6.6])
    sw_space, sw_time = window(mag)
    assert_array_almost_equal(sw_space, np.array([56.62752, 79.180511]))
    assert_array_almost_equal(sw_time, np.array([0.600467, 2.491178]))


def test_uhrhammer_window():
    window = UhrhammerWindow()
    mag = np.array([5.0, 6.6])
    sw_space, sw_time = window(mag)
    assert_array_almost_equal(sw_space, np.array([20.005355, 72.414025]))
    assert_array_almost_equal(sw_time, np.array([0.074705, 0.538907]))
