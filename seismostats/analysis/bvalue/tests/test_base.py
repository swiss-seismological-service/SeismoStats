import numpy as np
import pytest

from seismostats.analysis.bvalue import ClassicBValueEstimator
from seismostats.analysis.bvalue.positive import BPositiveBValueEstimator
from seismostats.analysis.bvalue.utils import shi_bolt_confidence
from seismostats.utils.simulate_distributions import simulate_magnitudes_binned


def test_estimate_b_warnings():
    mags = simulate_magnitudes_binned(n=100, b=1, mc=0, delta_m=0.1)

    # test that uncorrect binninng leads to error
    with pytest.raises(ValueError):
        estimator = ClassicBValueEstimator()
        estimator.calculate(mags, mc=0, delta_m=0.2)

    # test that warning is raised if smallest magnitude is much larger than mc
    with pytest.warns(UserWarning):
        estimator = ClassicBValueEstimator()
        estimator.calculate(mags, mc=-1, delta_m=0.1)

    # Magnitudes contain NaN values
    mags1 = np.array([np.nan, 1, 2, 3, 4])
    b1 = estimator.calculate(mags1, mc=1, delta_m=1)
    mags2 = np.array([1, 2, 3, 4])
    b2 = estimator.calculate(mags2, mc=1, delta_m=1)
    assert (b1 == b2)

    # No magnitudes above completeness magnitude
    mags = np.array([0, 0.9, 0.1, 0.2, 0.5])
    # make sure that warning is raised
    with pytest.warns(UserWarning):
        estimator.calculate(mags, mc=1, delta_m=0.1)
    assert (np.isnan(estimator.b_value))


def test_by_reference():
    estimator = ClassicBValueEstimator()

    # test that values below mc are filtered out
    mags = np.array([0, 1, 3.1, 1.1, 2, 1.2, 1.3, 1.4,
                    4.7, 1.5, 1.6, 1.7, 1.8, 3.4])
    estimator.calculate(mags, mc=1, delta_m=0.1)
    estimator.magnitudes.sort()
    assert not np.array_equal(mags, estimator.magnitudes)

    # test index is working
    mags = np.array([0, 0.9, -1, 0.2, 0.5])
    estimator.calculate(mags, mc=0.1, delta_m=0.1)
    assert (estimator.magnitudes == mags[estimator.idx]).all()


def test_beta():
    mags = np.array([0.1, 0.3, -0., 0.5, 0.4, 0.1, 0.3, -0., 0.2, 1.])
    estimator = ClassicBValueEstimator()
    estimator.calculate(mags, mc=0, delta_m=0.1)
    np.testing.assert_almost_equal(estimator.beta, 2.9626581614317242)


def test_getters():
    mags = simulate_magnitudes_binned(n=100, b=1, mc=0, delta_m=0.1)
    weights = np.ones_like(mags)

    estimator = ClassicBValueEstimator()
    estimator.calculate(mags, mc=0, delta_m=0.1, weights=weights)

    np.testing.assert_array_equal(weights, estimator.weights)
    np.testing.assert_array_equal(mags, estimator.magnitudes)

    with pytest.raises(Exception):
        estimator.n = 10

    estimator = BPositiveBValueEstimator()

    with pytest.raises(AttributeError):
        estimator.std

    estimator.calculate(mags, mc=0, delta_m=0.1, weights=weights)

    assert not np.array_equal(mags, estimator.magnitudes)
    assert not np.array_equal(weights, estimator.weights)
    assert estimator.n == len(estimator.magnitudes)


def test_std():
    mags = simulate_magnitudes_binned(n=100, b=1, mc=0, delta_m=0.1)
    estimator = ClassicBValueEstimator()
    estimator.calculate(mags, mc=1, delta_m=0.1)

    std_shi = shi_bolt_confidence(
        estimator.magnitudes, estimator.b_value)
    np.testing.assert_almost_equal(estimator.std, std_shi)

    std_shi_beta = shi_bolt_confidence(estimator.magnitudes, b=estimator.beta,
                                       b_parameter='beta')
    np.testing.assert_almost_equal(estimator.std_beta, std_shi_beta)
