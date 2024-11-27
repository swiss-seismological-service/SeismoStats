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
    total_s = np.array([delta.total_seconds() for delta in sw_time])
    expected = np.array([12416915.980999, 77021813.941164])
    assert_array_almost_equal(total_s, expected)


def test_gruenthal_window():
    window = GruenthalWindow()
    mag = np.array([5.0, 6.6])
    sw_space, sw_time = window(mag)
    assert_array_almost_equal(sw_space, np.array([56.62752, 79.180511]))
    total_s = np.array([delta.total_seconds() for delta in sw_time])
    expected = np.array([18923361.968176, 78507969.029983])
    assert_array_almost_equal(total_s, expected)


def test_uhrhammer_window():
    window = UhrhammerWindow()
    mag = np.array([5.0, 6.6])
    sw_space, sw_time = window(mag)
    assert_array_almost_equal(sw_space, np.array([20.005355, 72.414025]))
    total_s = np.array([delta.total_seconds() for delta in sw_time])
    expected = np.array([2354273.993272, 16983332.073632])
    assert_array_almost_equal(total_s, expected)
