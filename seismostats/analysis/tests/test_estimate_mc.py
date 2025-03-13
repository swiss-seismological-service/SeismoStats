import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal

from seismostats.analysis.bvalue.positive import BPositiveBValueEstimator
from seismostats.analysis.estimate_mc import (mc_by_bvalue_stability, mc_ks,
                                              mc_max_curvature)
from seismostats.analysis.bvalue.utils import beta_to_b_value


@pytest.fixture
def magnitudes():
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


@pytest.fixture
def ks_dists():
    ks_ds_df = pd.read_csv(
        'seismostats/analysis/tests/data/ks_ds.csv', index_col=0)
    ks_ds_list = ks_ds_df.values.T
    return ks_ds_list


def test_estimate_mc_ks(
        magnitudes,
        ks_dists,
        mcs=[0.8, 0.9, 1.0, 1.1]
):
    # test when beta is given
    best_mc, best_b_value, mcs_tested, b_values, ks_ds, ps = mc_ks(
        magnitudes,
        delta_m=0.1,
        mcs_test=mcs,
        p_pass=0.1,
        b_value=beta_to_b_value(2.24),
        ks_ds_list=ks_dists,
    )
    assert_equal(1.1, best_mc)
    assert_equal(beta_to_b_value(2.24), best_b_value)
    print(ps)
    assert_allclose(
        [beta_to_b_value(2.24), beta_to_b_value(
            2.24), beta_to_b_value(2.24), beta_to_b_value(2.24)],
        b_values,
        rtol=1e-7,
    )
    assert_allclose([0.0, 0.0, 0.0128, 0.4405], ps, atol=0.03)
    assert_equal(mcs_tested, mcs)

    # test when beta is not given
    best_mc, best_b_value, mcs_tested, b_values, ks_ds, ps = mc_ks(
        magnitudes,
        delta_m=0.1,
        mcs_test=[1.1],
        p_pass=0.1,
    )
    assert_almost_equal(beta_to_b_value(2.242124985031149), best_b_value)
    assert_allclose(
        [
            0.07087102843116255,
        ],
        ks_ds,
        rtol=1e-7,
    )
    assert_allclose(
        [
            beta_to_b_value(2.242124985031149),
        ],
        b_values,
        rtol=1e-7,
    )
    assert_allclose([4.362e-01], ps, atol=0.03)

    # test when mcs are not given
    best_mc, best_beta, mcs_tested, b_values, ks_ds, ps = mc_ks(
        magnitudes,
        delta_m=0.1,
        p_pass=0.1,
        b_value=beta_to_b_value(2.24),
        ks_ds_list=ks_dists[2:],
    )
    assert_equal(1.1, best_mc)
    assert_equal([1.0, 1.1], mcs_tested)

    # test when b-positive is used
    best_mc, best_b_value, mcs_tested, b_values, ks_ds, ps = mc_ks(
        magnitudes,
        delta_m=0.1,
        mcs_test=[1.5],
        b_method=BPositiveBValueEstimator
    )
    assert_equal(1.5, best_mc)
    assert_almost_equal(beta_to_b_value(3.2542240043462796), best_b_value)
    assert_equal(len(mcs_tested), 1)
    assert_equal(len(b_values), 1)
    assert_equal(len(ks_ds), 1)
    assert_equal(len(ps), 1)


def test_estimate_mc_maxc(magnitudes):
    mc = mc_max_curvature(magnitudes, delta_m=0.1, correction_factor=0.2)

    assert_equal(1.3, mc)


@pytest.fixture
def setup_catalog():
    swiss_catalog = pd.read_csv(
        'seismostats/analysis/tests/data/catalog_sed.csv', index_col=0)
    return swiss_catalog, 0.01


def test_estimate_mc_bvalue_stability(setup_catalog):
    swiss_catalog = setup_catalog[0]
    mags = swiss_catalog['magnitude'].values
    delta_m = setup_catalog[1]
    # make sure that the warning of no mags in lowest bin is not raised
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mc, _, _, _, _ = mc_by_bvalue_stability(
            mags, delta_m=delta_m,
            stability_range=0.5,
            mcs_test=np.arange(0.12, 2.0, delta_m))

    assert_almost_equal(1.44, mc)


def test_estimate_mc_bvalue_stability_larger_bins(magnitudes):
    mc, _, _, _, _ = mc_by_bvalue_stability(
        magnitudes, delta_m=0.1, stability_range=0.5)

    assert_almost_equal(1.1, mc)
