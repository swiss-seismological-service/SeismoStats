import numpy as np
import warnings
from seismostats.analysis.estimate_a import estimate_a


def test_estimate_a():
    mags = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    a = estimate_a(mags)
    assert a == 1.0

    # reference magnitude is given and b-value given
    a = estimate_a(mags, mc=1, m_ref=0, b_value=1)
    assert a == 2.0

    # reference magnitude but no b-value
    try:
        a = estimate_a(mags, mc=1, m_ref=0)
    except ValueError as e:
        assert str(e) == "b_value must be provided if m_ref is given"

    # reference time is given
    a = estimate_a(mags, scaling_factor=10)
    assert a == 0.0

    # magnitudes not cut at mc
    with warnings.catch_warnings(record=True) as w:
        estimate_a(mags, mc=2)
        assert w[-1].category == UserWarning
