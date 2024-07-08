import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal

from seismostats.analysis.estimate_mc import bin_to_precision
from seismostats.analysis.estimate_mc import (empirical_cdf, mc_ks,
                                              mc_max_curvature)


@pytest.fixture
def setup_magnitudes():
    mags = np.array(
        [
            2.3,
            1.2,
            1.5,
            1.2,
            1.7,
            1.1,
            1.2,
            1.5,
            1.8,
            1.6,
            1.2,
            1.5,
            1.2,
            1.7,
            1.6,
            1.1,
            1.1,
            1.2,
            2.0,
            1.1,
            1.2,
            1.1,
            1.2,
            1.6,
            1.9,
            1.3,
            1.7,
            1.3,
            1.0,
            1.2,
            1.7,
            1.3,
            1.3,
            1.1,
            1.5,
            1.4,
            1.1,
            2.1,
            1.2,
            2.2,
            1.7,
            1.6,
            1.1,
            2.0,
            2.1,
            1.2,
            1.0,
            1.5,
            1.2,
            1.7,
            1.8,
            1.1,
            1.3,
            1.1,
            1.3,
            1.4,
            2.1,
            2.0,
            1.1,
            2.2,
            1.8,
            1.4,
            1.1,
            1.0,
            2.0,
            2.0,
            1.1,
            1.0,
            1.0,
            1.5,
            1.6,
            3.7,
            2.8,
            1.5,
            1.1,
            1.2,
            1.4,
            2.3,
            1.5,
            1.2,
            1.7,
            1.1,
            1.6,
            1.2,
            1.5,
            1.1,
            1.2,
            1.7,
            1.2,
            1.6,
            1.2,
            1.1,
            1.8,
            1.2,
            1.1,
            1.0,
            1.3,
            1.1,
            1.6,
            1.6,
        ]
    )
    return mags


def test_empirical_cdf(setup_magnitudes, delta_m=0.1):
    x, y = empirical_cdf(setup_magnitudes, delta_m=delta_m)

    x_expected = bin_to_precision(np.arange(min(setup_magnitudes), max(
        setup_magnitudes) + delta_m, delta_m))

    assert_allclose(x, x_expected, rtol=1e-7)
    assert_equal(y[-1], 1)
    assert_equal(len(x), len(y))
    assert_equal(y[0], 0.06)

    # test that weights function the way that they should
    # 1. with equal weights
    magnitudes = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    weights = np.array([1, 1, 1, 1, 1])
    x_weight, y_weight = empirical_cdf(
        magnitudes, mc=0, delta_m=0.1, weights=weights)
    x, y = empirical_cdf(magnitudes, mc=0, delta_m=0.1)
    assert_almost_equal(x_weight, np.arange(0, 1, 0.1))
    assert_almost_equal(y_weight, [0, 0, 0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1])
    assert_almost_equal(x_weight, x)
    assert_almost_equal(y_weight, y)

    # 2. with different weights
    magnitudes = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    weights = np.array([0.5, 0.5, 0.5, 0.5, 3])
    x_weight, y_weight = empirical_cdf(
        magnitudes, mc=0, delta_m=0.1, weights=weights)
    assert_almost_equal(x_weight, np.arange(0, 1, 0.1))
    assert_almost_equal(y_weight, [0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 1])


@pytest.fixture
def setup_ks_dists():
    ks_ds_df = pd.read_csv(
        'seismostats/analysis/tests/data/ks_ds.csv', index_col=0)
    ks_ds_list = ks_ds_df.values.T
    return ks_ds_list


def test_estimate_mc_ks(
        setup_magnitudes,
        setup_ks_dists,
        mcs=[0.8, 0.9, 1.0, 1.1]
):
    # test when beta is given
    best_mc, best_beta, mcs_tested, betas, ks_ds, ps = mc_ks(
        setup_magnitudes,
        delta_m=0.1,
        mcs_test=mcs,
        p_pass=0.1,
        beta=2.24,
        ks_ds_list=setup_ks_dists,
    )
    assert_equal(1.1, best_mc)
    assert_equal(2.24, best_beta)
    assert_allclose(
        [2.24, 2.24, 2.24, 2.24],
        betas,
        rtol=1e-7,
    )
    assert_allclose([0.0, 0.0, 0.0128, 0.4405], ps, atol=0.03)
    assert_equal(mcs_tested, mcs)

    # test when beta is not given
    best_mc, best_beta, mcs_tested, betas, ks_ds, ps = mc_ks(
        setup_magnitudes,
        delta_m=0.1,
        mcs_test=[1.1],
        p_pass=0.1,
    )
    assert_almost_equal(2.242124985031149, best_beta)
    assert_allclose(
        [
            0.07087102843116255,
        ],
        ks_ds,
        rtol=1e-7,
    )
    assert_allclose(
        [
            2.242124985031149,
        ],
        betas,
        rtol=1e-7,
    )
    assert_allclose([4.362e-01], ps, atol=0.03)

    # test when mcs are not given
    best_mc, best_beta, mcs_tested, betas, ks_ds, ps = mc_ks(
        setup_magnitudes,
        delta_m=0.1,
        p_pass=0.1,
        beta=2.24,
        ks_ds_list=setup_ks_dists[2:],
    )
    assert_equal(1.1, best_mc)
    assert_equal([1.0, 1.1], mcs_tested)

    # test when b-positive is used
    best_mc, best_beta, mcs_tested, betas, ks_ds, ps = mc_ks(
        setup_magnitudes,
        delta_m=0.1,
        mcs_test=[1.5],
        b_method="positive"
    )
    assert_equal(1.5, best_mc)
    assert_almost_equal(3.2542240043462796, best_beta)
    assert_equal(len(mcs_tested), 1)
    assert_equal(len(betas), 1)
    assert_equal(len(ks_ds), 1)
    assert_equal(len(ps), 1)


def test_estimate_mc_maxc(setup_magnitudes):
    mc = mc_max_curvature(setup_magnitudes, delta_m=0.1, correction_factor=0.2)

    assert_equal(1.3, mc)
