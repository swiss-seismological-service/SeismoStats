import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from datetime import datetime, timedelta

from seismostats.analysis.estimate_a import (
    estimate_a_classic, estimate_a_positive, estimate_a_more_positive)


def test_estimate_a_classic():
    mags = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    a = estimate_a_classic(mags, delta_m=1)
    assert a == 1.0

    # reference magnitude is given and b-value given
    a = estimate_a_classic(mags, mc=1, m_ref=0, b_value=1, delta_m=1)
    assert a == 2.0

    # reference magnitude but no b-value
    try:
        a = estimate_a_classic(mags, mc=1, m_ref=0, delta_m=1)
    except ValueError as e:
        assert str(e) == "b_value must be provided if m_ref is given"

    # reference time is given
    a = estimate_a_classic(mags, scaling_factor=10, delta_m=1)
    assert a == 0.0

    # magnitudes not cut at mc
    with pytest.warns(UserWarning):
        estimate_a_classic(mags, mc=2, delta_m=1)


def test_estimate_a_positive():
    mags = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 10])
    # array of dates from 1.1.2000 to 10.1.2000
    times = np.arange(datetime(2000, 1, 1), datetime(
        2000, 1, 12), timedelta(days=1)).astype(datetime)

    a = estimate_a_positive(mags, times, delta_m=1)
    assert_almost_equal(10**a, 10.0)

    # reference magnitude is given and b-value given
    a = estimate_a_positive(mags, times, delta_m=1, mc=1, m_ref=0, b_value=1)
    assert_almost_equal(10**a, 100.0)

    # reference magnitude but no b-value
    try:
        a = estimate_a_positive(mags, times, delta_m=1, mc=1, m_ref=0)
    except ValueError as e:
        assert str(e) == "b_value must be provided if m_ref is given"

    # reference time is given
    a = estimate_a_positive(mags, times, delta_m=1, scaling_factor=10)
    assert_almost_equal(10**a, 1.0)

    # magnitudes not cut at mc
    with pytest.warns(UserWarning):
        estimate_a_positive(mags, times, delta_m=1, mc=2)


def test_estimate_a_more_positive():
    mags = np.array([0.9, 0.9, 0.9, 0.9, 10.9])
    times = np.arange(datetime(2000, 1, 1), datetime(
        2000, 1, 6), timedelta(days=1)).astype(datetime)

    a = estimate_a_more_positive(
        mags, times, delta_m=0.1, dmc=0.1, mc=0, b_value=1)
    assert_almost_equal(10**a, 16.0)

    a = estimate_a_more_positive(
        mags, times, delta_m=0.1, dmc=0.1, m_ref=-1, b_value=1)
    assert_almost_equal(10**a, 160.0)
