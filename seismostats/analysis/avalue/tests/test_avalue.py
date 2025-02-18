from datetime import datetime, timedelta

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from seismostats.analysis.avalue import (APositiveAValueEstimator,
                                         ClassicAValueEstimator, estimate_a)
from seismostats.analysis.avalue.more_positive import \
    AMorePositiveAValueEstimator


@pytest.mark.filterwarnings("ignore")
def test_estimate_a():
    mags = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # arguments are passed correctly
    a = estimate_a(mags, mc=1, delta_m=1)
    assert a == 1.0

    mags = np.array([0.9, 0.9, 0.9, 0.9, 10.9])
    times = np.arange(datetime(2000, 1, 1), datetime(
        2000, 1, 6), timedelta(days=1)).astype(datetime)

    # *args and **kwargs are passed correctly
    a = estimate_a(mags,
                   mc=0,
                   delta_m=0.1,
                   times=times,
                   b_value=1,
                   method=AMorePositiveAValueEstimator,
                   m_ref=-1,
                   dmc=0.2)
    assert_almost_equal(10**a, 201.42806588)


def test_estimate_a_classic():
    mags = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    estimator = ClassicAValueEstimator()
    a_estimate = estimator.calculate(mags, mc=1, delta_m=1)
    assert a_estimate == 1.0

    # reference magnitude is given and b-value given
    a_estimate = estimator.calculate(mags, mc=1, delta_m=0.0,
                                     m_ref=0, b_value=1)
    assert a_estimate == 2.0

    # reference magnitude but no b-value
    with pytest.raises(ValueError):
        estimator.calculate(mags, mc=1, delta_m=0.0001, m_ref=0)

    # reference time is given
    a_estimate = estimator.calculate(mags, mc=1, delta_m=0.0001,
                                     scaling_factor=10)
    assert a_estimate == 0.0

    # test that warning is raised if smallest magnitude is much larger than mc
    with pytest.warns(UserWarning):
        estimator.calculate(mags, mc=-1, delta_m=0.0)


def test_estimate_a_positive():
    mags = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 10])
    # array of dates from 1.1.2000 to 10.1.2000
    times = np.arange(datetime(2000, 1, 1), datetime(
        2000, 1, 12), timedelta(days=1)).astype(datetime)

    estimator = APositiveAValueEstimator()
    a = estimator.calculate(mags, mc=1, delta_m=1, times=times)
    assert_almost_equal(10**a, 10.0)

    # reference magnitude is given and b-value given
    a = estimator.calculate(mags, mc=1, delta_m=1,
                            m_ref=0, b_value=1, times=times)
    assert_almost_equal(10**a, 100.0)

    # reference magnitude but no b-value
    try:
        a = estimator.calculate(mags, mc=1, delta_m=1,
                                m_ref=0, times=times)
    except ValueError as e:
        assert str(e) == "b_value must be provided if m_ref is given"

    # reference time is given
    a = estimator.calculate(mags, mc=1, delta_m=1,
                            scaling_factor=10, times=times)
    assert_almost_equal(10**a, 1.0)

    # test that warning is raised if smallest magnitude is much larger than mc
    with pytest.warns(UserWarning):
        estimator.calculate(mags, delta_m=1, mc=-1, times=times)

    # test that index works correctly
    mags = np.array([0, 0.9, -1, 0.2, 0.5])
    times = np.array([datetime(2000, 1, 1),
                      datetime(2000, 1, 2),
                      datetime(2000, 1, 3),
                      datetime(2000, 1, 4),
                      datetime(2000, 1, 5)])
    estimator.calculate(mags, mc=-1, delta_m=0.1, times=times)
    assert (mags[estimator.idx] == np.array([0.9, 0.2, 0.5])).all()
    assert (mags[estimator.idx] == estimator.magnitudes).all()

    # test that time array is correctly used
    mags = np.array([0, 0.9, 0.5, 0.2, -1])
    times = np.array([datetime(2000, 1, 1),
                      datetime(2000, 1, 2),
                      datetime(2000, 1, 5),
                      datetime(2000, 1, 4),
                      datetime(2000, 1, 3)])
    estimator.calculate(mags, mc=-1, delta_m=0.1, times=times)
    assert (mags[estimator.idx] == np.array([0.9, 0.2, 0.5])).all()
    assert (mags[estimator.idx] == estimator.magnitudes).all()


@pytest.mark.filterwarnings("ignore")
def test_estimate_a_more_positive():

    mags = np.array([0.9, 0.9, 0.9, 0.9, 10.9])
    times = np.arange(datetime(2000, 1, 1), datetime(
        2000, 1, 6), timedelta(days=1)).astype(datetime)

    estimator = AMorePositiveAValueEstimator()

    a = estimator.calculate(
        mags, 0, 0.1, times, b_value=1, dmc=0.1)
    assert_almost_equal(10**a, 16.0)

    a = estimator.calculate(
        mags, 0, 0.1, times, b_value=1, m_ref=-1, dmc=0.1)
    assert_almost_equal(10**a, 160.0)

    with pytest.raises(ValueError):
        estimate_a(mags, mc=0, delta_m=0.1, times=times,
                   m_ref=-1, method=AMorePositiveAValueEstimator)

    # check that cutting at mc is handled correctly
    mags = np.array([-0.5, 0.9, 0.9, 0.9, 0.9, 10.9])
    times = np.arange(datetime(2000, 1, 1), datetime(
        2000, 1, 7), timedelta(days=1)).astype(datetime)

    a = estimator.calculate(
        mags, 0, 0.1, times, b_value=1, dmc=0.1)
    assert_almost_equal(10**a, 16.0)

    # test that index works correctly
    mags = np.array([0, 0.9, -1, 0.2, 0.5, 1])
    times = np.array([datetime(2000, 1, 1),
                      datetime(2000, 1, 2),
                      datetime(2000, 1, 3),
                      datetime(2000, 1, 4),
                      datetime(2000, 1, 5),
                      datetime(2000, 1, 6)])
    estimator.calculate(mags, mc=-1, delta_m=0.1, times=times, b_value=1)
    assert (mags[estimator.idx] == np.array([0.9, 1, 0.2, 0.5, 1])).all()
    assert (estimator.idx == np.array([1, 5, 3, 4, 5])).all()
    assert (mags[estimator.idx] == estimator.magnitudes).all()
    assert (estimator.times == times[estimator.idx]).all()

    # test that time array is correctly used
    mags = np.array([0, 0.9, 0.5, 0.2, -1])
    times = np.array([datetime(2000, 1, 1),
                      datetime(2000, 1, 2),
                      datetime(2000, 1, 5),
                      datetime(2000, 1, 4),
                      datetime(2000, 1, 3)])
    estimator.calculate(mags, mc=-1, delta_m=0.1, times=times, b_value=1)
    assert (mags[estimator.idx] == np.array([0.9, 0.2, 0.5])).all()
    assert (mags[estimator.idx] == estimator.magnitudes).all()
    assert (estimator.times == times[estimator.idx]).all()
