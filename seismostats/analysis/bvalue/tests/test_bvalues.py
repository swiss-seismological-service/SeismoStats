import os

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from seismostats.analysis.bvalue import (BMorePositiveEstimator,
                                         BPositiveEstimator,
                                         ClassicBValueEstimator,
                                         UtsuBValueEstimator)
from seismostats.utils.simulate_distributions import bin_to_precision

PATH_RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'data')


def magnitudes(b: float):
    df_mags = pd.read_csv(os.path.join(
        PATH_RESOURCES, 'simulated_magnitudes.csv'))
    if b == 0.5:
        mags = df_mags["b_value = 0.5"].values
    elif b == 1:
        mags = df_mags["b_value = 1"].values
    elif b == 1.5:
        mags = df_mags["b_value = 1.5"].values
    return mags


@pytest.mark.parametrize(
    "b_est_correct, mags, mc, delta_m",
    [
        (0.9985052730956719, magnitudes(1), 0, 0.1),
        (1.486114976626299, magnitudes(1.5), 0.5, 0.01),
        (0.5012460753985151, magnitudes(0.5), 2, 0.2),
    ],
)
def test_estimate_b_classic(
    b_est_correct: float,
    mags: np.ndarray,
    mc: float,
    delta_m: float,
):
    mags = bin_to_precision(mags, delta_m)
    mags = mags[mags >= mc - delta_m / 2]
    estimator = ClassicBValueEstimator(mc=mc, delta_m=delta_m)
    b_estimate = estimator(mags)

    assert_almost_equal(b_estimate, b_est_correct)


@pytest.mark.parametrize(
    "b_est_correct, mags, mc, delta_m",
    [
        (0.9941299341459253, magnitudes(1), 0, 0.1),
        (1.485969980462011, magnitudes(1.5), 0.5, 0.01),
        (0.49903218920704306, magnitudes(0.5), 2, 0.2),
    ],
)
def test_estimate_b_utsu(
    b_est_correct: float,
    mags: np.ndarray,
    mc: float,
    delta_m: float,
):
    mags = bin_to_precision(mags, delta_m)
    mags = mags[mags >= mc - delta_m / 2]
    estimator = UtsuBValueEstimator(mc=mc, delta_m=delta_m)
    b_estimate = estimator(mags)
    assert_almost_equal(b_estimate, b_est_correct)


@pytest.mark.parametrize(
    "b_est_correct, mags, mc, delta_m, dmc",
    [
        (1.00768483769521, magnitudes(1), 0, 0.1, 0.3),
        (1.4946439854664, magnitudes(1.5), 0.5, 0.01, None),
        (0.4903952163745402, magnitudes(0.5), 2, 0.2, None),
    ],
)
def test_estimate_b_positive(
    b_est_correct: float,
    mags: np.ndarray,
    mc: float,
    delta_m: float,
    dmc: float,
):
    mags = bin_to_precision(mags, delta_m)
    mags = mags[mags >= mc - delta_m / 2]
    estimator = BPositiveEstimator(mc=mc, delta_m=delta_m, dmc=dmc)
    b_estimate = estimator(mags)
    assert_almost_equal(b_estimate, b_est_correct)


@pytest.mark.parametrize(
    "b_est_correct, mags, mc, delta_m, dmc",
    [
        (1.03259579513585, magnitudes(1), 0, 0.1, 0.3),
        (1.476841984167775, magnitudes(1.5), 0.5, 0.01, None),
        (0.4869048157602977, magnitudes(0.5), 2, 0.2, None),
    ],
)
def test_estimate_b_more_positive(
    b_est_correct: float,
    mags: np.ndarray,
    mc: float,
    delta_m: float,
    dmc: float
):
    mags = bin_to_precision(mags, delta_m)
    mags = mags[mags >= mc - delta_m / 2]
    estimator = BMorePositiveEstimator(mc=mc, delta_m=delta_m, dmc=dmc)
    b_estimate = estimator(mags)
    assert_almost_equal(b_estimate, b_est_correct)
