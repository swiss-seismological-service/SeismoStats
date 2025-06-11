import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal

from seismostats.analysis.bvalue.positive import BPositiveBValueEstimator
from seismostats.analysis.bvalue.utils import beta_to_b_value
from seismostats.analysis.estimate_mc import (estimate_mc_b_stability,
                                              estimate_mc_ks, estimate_mc_maxc)
from seismostats.utils.binning import bin_to_precision

MAGNITUDES = np.array(
    [
        2.3, 1.2, 1.5, 1.2, 1.7, 1.1, 1.2, 1.5, 1.8, 1.6, 1.2, 1.5,
        1.2, 1.7, 1.6, 1.1, 1.1, 1.2, 2.0, 1.1, 1.2, 1.1, 1.2, 1.6,
        1.9, 1.3, 1.7, 1.3, 1.0, 1.2, 1.7, 1.3, 1.3, 1.1, 1.5, 1.4,
        1.1, 2.1, 1.2, 2.2, 1.7, 1.6, 1.1, 2.0, 2.1, 1.2, 1.0, 1.5,
        1.2, 1.7, 1.8, 1.1, 1.3, 1.1, 1.3, 1.4, 2.1, 2.0, 1.1, 2.2,
        1.8, 1.4, 1.1, 1.0, 2.0, 2.0, 1.1, 1.0, 1.0, 1.5, 1.6, 3.7,
        2.8, 1.5, 1.1, 1.2, 1.4, 2.3, 1.5, 1.2, 1.7, 1.1, 1.6, 1.2,
        1.5, 1.1, 1.2, 1.7, 1.2, 1.6, 1.2, 1.1, 1.8, 1.2, 1.1, 1.0,
        1.3, 1.1, 1.6, 1.6,
    ]
)

KS_DISTS = pd.read_csv(
    'seismostats/analysis/tests/data/ks_ds.csv', index_col=0).values.T


def test_estimate_mc_ks_out(capfd):
    mcs = [0.8, 0.9, 1.0, 1.1]

    # test when beta is given
    _ = estimate_mc_ks(
        MAGNITUDES,
        delta_m=0.1,
        mcs_test=mcs,
        p_value_pass=0.1,
        b_value=beta_to_b_value(2.24),
        ks_ds_list=KS_DISTS,
        verbose=True)

    out, err = capfd.readouterr()
    assert f"with a b-value of: {beta_to_b_value(2.24):.3f}" in out
    assert "..p-value: " in out

    with pytest.raises(ValueError):
        _ = estimate_mc_ks(
            MAGNITUDES * 1.01234,
            delta_m=0.1,
            mcs_test=mcs,
            p_value_pass=0.1,
            ks_ds_list=KS_DISTS)

    with pytest.warns(UserWarning):
        _ = estimate_mc_ks(
            MAGNITUDES * 1.01234,
            delta_m=0.1,
            mcs_test=mcs,
            p_value_pass=0.1,
            b_value=beta_to_b_value(2.24),
            ks_ds_list=KS_DISTS)

    with pytest.warns():
        _ = estimate_mc_ks(
            MAGNITUDES,
            delta_m=0.1,
            mcs_test=np.array(mcs) * 1.11234,
            p_value_pass=0.1,
            b_value=beta_to_b_value(2.24),
            ks_ds_list=KS_DISTS)


def test_estimate_mc_ks_fail(capfd):
    mcs = [2]

    # test when beta is given
    best_mc, mc_info = estimate_mc_ks(
        MAGNITUDES,
        delta_m=0.1,
        mcs_test=mcs,
        p_value_pass=0.1,
        b_value=beta_to_b_value(2.24),
        ks_ds_list=KS_DISTS,
        verbose=True
    )
    out, _ = capfd.readouterr()
    assert best_mc is None
    assert mc_info['best_b_value'] is None
    assert "None of the mcs passed the test." in out


def test_estimate_mc_ks():
    mcs = [0.8, 0.9, 1.0, 1.1]

    # test when beta is given
    best_mc, mc_info = estimate_mc_ks(
        MAGNITUDES,
        delta_m=0.1,
        mcs_test=mcs,
        p_value_pass=0.1,
        b_value=beta_to_b_value(2.24),
        ks_ds_list=KS_DISTS,
    )
    assert_equal(1.1, best_mc)
    assert_equal(beta_to_b_value(2.24), mc_info['best_b_value'])

    assert_allclose(
        [beta_to_b_value(2.24), beta_to_b_value(
            2.24), beta_to_b_value(2.24), beta_to_b_value(2.24)],
        mc_info['b_values_tested'],
        rtol=1e-7,
    )
    assert_allclose([0.0, 0.0, 0.0128, 0.4405], mc_info['p_values'], atol=0.03)
    assert_equal(mc_info['mcs_tested'], mcs)

    # test when beta is not given
    best_mc, mc_info = estimate_mc_ks(
        MAGNITUDES,
        delta_m=0.1,
        mcs_test=[1.1],
        p_value_pass=0.1,
    )
    assert_almost_equal(beta_to_b_value(2.242124985031149),
                        mc_info['best_b_value'])
    assert_allclose(
        [
            0.07087102843116255,
        ],
        mc_info['ks_ds'],
        rtol=1e-7,
    )
    assert_allclose(
        [
            beta_to_b_value(2.242124985031149),
        ],
        mc_info['b_values_tested'],
        rtol=1e-7,
    )
    assert_allclose([4.362e-01], mc_info['p_values'], atol=0.03)

    # test when mcs are not given
    best_mc, mc_info = estimate_mc_ks(
        MAGNITUDES,
        delta_m=0.1,
        p_value_pass=0.1,
        b_value=beta_to_b_value(2.24),
        ks_ds_list=KS_DISTS[2:],
    )
    assert_equal(1.1, best_mc)
    assert_equal([1.0, 1.1], mc_info['mcs_tested'])

    # test when b-positive is used
    best_mc, mc_info = estimate_mc_ks(
        MAGNITUDES,
        delta_m=0.1,
        mcs_test=[1.5],
        b_method=BPositiveBValueEstimator
    )
    assert_equal(1.5, best_mc)
    assert_almost_equal(beta_to_b_value(3.2542240043462796),
                        mc_info['best_b_value'])
    assert_equal(len(mc_info['mcs_tested']), 1)
    assert_equal(len(mc_info['b_values_tested']), 1)
    assert_equal(len(mc_info['ks_ds']), 1)
    assert_equal(len(mc_info['p_values']), 1)


def test_estimate_mc_maxc():
    mc, _ = estimate_mc_maxc(MAGNITUDES, fmd_bin=0.1, correction_factor=0.2)
    assert_equal(1.3, mc)


@pytest.fixture
def setup_catalog():
    swiss_catalog = pd.read_csv(
        'seismostats/analysis/tests/data/catalog_sed.csv', index_col=0)
    return swiss_catalog, 0.01


def test_estimate_mc_b_stability(setup_catalog):
    swiss_catalog = setup_catalog[0]
    mags = swiss_catalog['magnitude'].values
    delta_m = setup_catalog[1]
    mags = bin_to_precision(mags, delta_m)
    # make sure that the warning of no mags in lowest bin is not raised
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mc, _ = estimate_mc_b_stability(
            mags, delta_m=delta_m,
            stability_range=0.5,
            mcs_test=np.arange(0.12, 2.0, delta_m))

    assert_almost_equal(1.44, mc)


def test_estimate_mc_b_stability_larger_bins(capfd):
    mc, _ = estimate_mc_b_stability(
        MAGNITUDES, delta_m=0.1, stability_range=0.5, verbose=True)
    assert_almost_equal(1.1, mc)
    out, _ = capfd.readouterr()
    assert f"Best mc to pass the test: {mc:.3f}" in out


def test_estimate_mc_b_stability_fail(capfd):
    with pytest.warns():
        mc, mc_info = estimate_mc_b_stability(
            MAGNITUDES, delta_m=0.1, mcs_test=[0.5], stability_range=0.5,
            verbose=True)
    out, err = capfd.readouterr()
    assert mc is None
    assert mc_info['best_b_value'] is None
    assert "None of the mcs passed the stability test." in out

    with pytest.warns():
        estimate_mc_b_stability(
            MAGNITUDES, delta_m=0.1, mcs_test=[1.123, 1.231],
            stability_range=0.5)

    with pytest.raises(ValueError):
        estimate_mc_b_stability(
            MAGNITUDES * 1.01234, delta_m=0.1, mcs_test=[1.1],
            stability_range=0.5, b_value=1.2)
