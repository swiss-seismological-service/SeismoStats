import warnings

import numpy as np
import pytest

from seismostats.analysis.bvalue import ClassicBValueEstimator
from seismostats.analysis.bvalue.positive import BPositiveBValueEstimator
from seismostats.analysis.bvalue.utils import shi_bolt_confidence
from seismostats.utils.simulate_distributions import simulate_magnitudes_binned


def test_estimate_b_warnings():
    mags = simulate_magnitudes_binned(n=100, b=1, mc=0, delta_m=0.1)

    # test that uncorrect binninng leads to error
    with pytest.raises(AssertionError):
        estimator = ClassicBValueEstimator(mags, mc=0, delta_m=0.2)
        estimator.b_value()

    # test that warning is raised if smallest magnitude is much larger than mc
    with warnings.catch_warnings(record=True) as w:
        estimator = ClassicBValueEstimator(mags, mc=-1, delta_m=0.1)
        estimator.b_value()
        assert w[-1].category == UserWarning


def test_by_reference():
    mags = simulate_magnitudes_binned(n=100, b=1, mc=0, delta_m=0.1)
    estimator = ClassicBValueEstimator(mags, mc=1, delta_m=0.1)
    estimator.b_value()
    estimator.magnitudes.sort()
    assert not np.array_equal(mags, estimator.magnitudes)


def test_beta():
    mags = np.array([0.1, 0.3, -0., 0.5, 0.4, 0.1, 0.3, -0., 0.2, 1.])
    estimator = ClassicBValueEstimator(mags, mc=0, delta_m=0.1)
    np.testing.assert_almost_equal(estimator.beta(), 2.9626581614317242)


def test_getters():
    mags = simulate_magnitudes_binned(n=100, b=1, mc=0, delta_m=0.1)
    weights = np.ones_like(mags)
    estimator = BPositiveBValueEstimator(
        mags, mc=0, delta_m=0.1, weights=weights)

    assert np.array_equal(mags, estimator.magnitudes)
    assert np.array_equal(weights, estimator.weights)

    estimator.b_value()

    assert not np.array_equal(mags, estimator.magnitudes)
    assert not np.array_equal(weights, estimator.weights)
    assert estimator.n == len(estimator.magnitudes)

    with pytest.raises(Exception):
        estimator.n = 10

    estimator.dmc = 0.1  # resets the estimates

    with pytest.raises(AttributeError):
        estimator.std()

    np.testing.assert_array_equal(weights, estimator.weights)
    np.testing.assert_array_equal(mags, estimator.magnitudes)


def test_std():
    mags = simulate_magnitudes_binned(n=100, b=1, mc=0, delta_m=0.1)
    estimator = ClassicBValueEstimator(mags, mc=1, delta_m=0.1)

    std_shi = shi_bolt_confidence(mags, b=estimator.b_value())
    np.testing.assert_almost_equal(estimator.std(), std_shi)

    std_shi_beta = shi_bolt_confidence(mags, b=estimator.beta(),
                                       b_parameter='beta')
    np.testing.assert_almost_equal(estimator.std_beta(), std_shi_beta)
