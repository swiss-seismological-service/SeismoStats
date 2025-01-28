import numpy as np
import warnings
from numpy.testing import assert_almost_equal
from datetime import datetime, timedelta

from seismostats.analysis.avalue import (
    APositiveAValueEstimator, ClassicAValueEstimator
)


def test_estimate_a_classic():
    mags = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    estimator = ClassicAValueEstimator()
    a_estimate = estimator.calculate(mags, mc=1, delta_m=10.0)
    assert a_estimate == 1.0

    # reference magnitude is given and b-value given
    a_estimate = estimator.calculate(mags, mc=1, delta_m=0.0,
                                     m_ref=0, b_value=1)
    assert a_estimate == 2.0

    # reference magnitude but no b-value
    try:
        estimator.calculate(mags, mc=1, delta_m=0.0,
                            m_ref=0)
    except ValueError as e:
        assert str(e) == "b_value must be provided if m_ref is given"

    # reference time is given
    a_estimate = estimator.calculate(mags, mc=1, delta_m=0.0,
                                     scaling_factor=10)
    assert a_estimate == 0.0

    # magnitudes not cut at mc
    with warnings.catch_warnings(record=True) as w:
        estimator.calculate(mags, mc=2, delta_m=0.0)
        assert w[-1].category == UserWarning

    # test that warning is raised if smallest magnitude is much larger than mc
    with warnings.catch_warnings(record=True) as w:
        estimator.calculate(mags, mc=-1, delta_m=0.0)
        assert w[-1].category == UserWarning


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

    # magnitudes not cut at mc
    with warnings.catch_warnings(record=True) as w:
        estimator.calculate(mags, delta_m=1, mc=2, times=times)
        assert w[-1].category == UserWarning

    # test that warning is raised if smallest magnitude is much larger than mc
    with warnings.catch_warnings(record=True) as w:
        estimator.calculate(mags, delta_m=1, mc=-1, times=times)
        assert w[-1].category == UserWarning
