import warnings

import numpy as np
import pytest

from seismostats.analysis.bvalue import ClassicBValueEstimator
from seismostats.utils.simulate_distributions import simulate_magnitudes_binned


def test_estimate_b_warnings():
    mags = simulate_magnitudes_binned(n=100, b=1, mc=0, delta_m=0.1)

    # test that uncorrect binninng leads to error
    with pytest.raises(AssertionError):
        estimator = ClassicBValueEstimator(mc=0, delta_m=0.2)
        estimator(mags)

    # test that warning is raised if smallest magnitude is much larger than mc
    with warnings.catch_warnings(record=True) as w:
        estimator = ClassicBValueEstimator(mc=-1, delta_m=0.1)
        estimator(mags)
        assert w[-1].category == UserWarning


def test_by_reference():
    mags = simulate_magnitudes_binned(n=100, b=1, mc=0, delta_m=0.1)
    estimator = ClassicBValueEstimator(mc=1, delta_m=0.1)
    estimator(mags)
    estimator.magnitudes.sort()
    assert not np.array_equal(mags, estimator.magnitudes)
