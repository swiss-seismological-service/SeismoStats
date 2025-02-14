import numpy as np
import pytest

from seismostats.analysis.avalue.classic import ClassicAValueEstimator
from seismostats.utils.simulate_distributions import simulate_magnitudes_binned


def test_estimate_a_warnings():
    mags = simulate_magnitudes_binned(n=100, b=1, mc=0, delta_m=0.1)

    # test that uncorrect binninng leads to error
    with pytest.raises(AssertionError):
        estimator = ClassicAValueEstimator()
        estimator.calculate(mags, mc=0, delta_m=0.2)

    # test that warning is raised if smallest magnitude is much larger than mc
    with pytest.warns(UserWarning):
        estimator = ClassicAValueEstimator()
        estimator.calculate(mags, mc=-1, delta_m=0.1)

    # Magnitudes contain NaN values
    mags = np.array([np.nan])
    with pytest.raises(ValueError):
        estimator.calculate(mags, mc=1, delta_m=0.1)

    # No magnitudes above completeness magnitude
    mags = np.array([0, 0.9, 0.1, 0.2, 0.5])
    with pytest.raises(AssertionError):
        estimator.calculate(mags, mc=1, delta_m=0.1)


def test_by_reference():
    estimator = ClassicAValueEstimator()

    mags = simulate_magnitudes_binned(n=100, b=1, mc=0, delta_m=0.1)
    estimator.calculate(mags, mc=0, delta_m=0.1)
    estimator.magnitudes.sort()
    assert not np.array_equal(mags, estimator.magnitudes)

    # test index is working
    mags = np.array([0, 0.9, -1, 0.2, 0.5])
    estimator.calculate(mags, mc=0.1, delta_m=0.1)
    assert (estimator.magnitudes == mags[estimator.idx]).all()


def test_reference_scaling():
    mags = simulate_magnitudes_binned(n=100, b=1, mc=0, delta_m=0.1)

    with pytest.raises(ValueError):
        estimator = ClassicAValueEstimator()
        estimator.calculate(mags, mc=0, delta_m=0.1, m_ref=0)

    # TODO: test that the scaling is correct
