from numpy.testing import assert_almost_equal
from datetime import datetime

import numpy as np
import pytest

from seismostats.analysis.bvalue.tests.test_bvalues import magnitudes
from seismostats.analysis.bvalue.utils import (b_value_to_beta,
                                               make_more_incomplete,
                                               shi_bolt_confidence)


def test_make_more_incomplete():
    magnitudes = np.array([1, 2, 20, 3, 4, 9, 3])
    times = np.array([
        datetime(2020, 1, 1),
        datetime(2020, 1, 2),
        datetime(2020, 1, 3),
        datetime(2020, 1, 4),
        datetime(2020, 1, 5),
        datetime(2020, 1, 6),
        datetime(2020, 1, 7),
    ])

    mags_inc, times_inc = make_more_incomplete(
        magnitudes, times, delta_t=np.timedelta64(49, "h")
    )

    assert (mags_inc == [1, 2, 20, 9]).all()
    assert (
        times_inc
        == [
            datetime(2020, 1, 1),
            datetime(2020, 1, 2),
            datetime(2020, 1, 3),
            datetime(2020, 1, 6),
        ]
    ).all()

    mags_inc, times_inc, idx = make_more_incomplete(
        magnitudes, times, delta_t=np.timedelta64(49, "h"), return_idx=True
    )

    assert (mags_inc == magnitudes[idx]).all()


@pytest.mark.parametrize(
    "std, mags, b, b_parameter",
    [
        (0.09776728958456313, magnitudes(1)[:100], 1, "b_value"),
        (0.1062329763800726, magnitudes(1.5)[:200], 1.5, "b_value"),
        (0.100184931569467, magnitudes(0.5)[
         :100], b_value_to_beta(0.5), "beta"),
    ],
)
def test_shi_bolt_confidence(
        std: float, mags: np.ndarray, b: float, b_parameter: str):
    assert_almost_equal(
        shi_bolt_confidence(mags, b=b, b_parameter=b_parameter), std)
