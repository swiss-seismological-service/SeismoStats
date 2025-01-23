import numpy as np
import warnings
from numpy.testing import assert_almost_equal
from datetime import datetime, timedelta

from seismostats.analysis.b_significant import (
    est_morans_i,
    b_series,
    cut_constant_idx,
    transform_n,
)


def test_est_morans_i():
    # 1 dimensional case
    values = np.array([1, 2, 3, 4, 5])
    w = np.array([[0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0]])
    ac, n, n_p = est_morans_i(values, w)
    assert_almost_equal(ac, 0.4)
    assert n == 5
    assert n_p == 4
