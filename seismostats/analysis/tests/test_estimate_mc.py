import pickle
import numpy as np
import pytest
import pandas as pd
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal

from seismostats.io.client import FDSNWSEventClient
from seismostats.analysis.estimate_mc import (empirical_cdf,
                                              mc_ks,
                                              mc_max_curvature,
                                              mc_by_bvalue_stability)


@pytest.fixture
def swiss_2023_magnitudes():
    start_time = pd.to_datetime('2023/01/01')
    end_time = pd.to_datetime('2024/01/01')
    min_longitude = 5
    max_longitude = 11
    min_latitude = 45
    max_latitude = 48
    min_magnitude = 0
    url = 'http://eida.ethz.ch/fdsnws/event/1/query'

    client = FDSNWSEventClient(url)
    df_sed = client.get_events(
        start_time=start_time,
        end_time=end_time,
        min_magnitude=min_magnitude,
        min_longitude=min_longitude,
        max_longitude=max_longitude,
        min_latitude=min_latitude,
        max_latitude=max_latitude)

    magnitudes = df_sed['magnitude'].to_numpy()
    binsize = 0.01
    return magnitudes, binsize


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


# load data for test_empirical_cdf
with open("seismostats/analysis/tests/data/test_empirical_cdf.p", "rb") as f:
    data = pickle.load(f)


@pytest.mark.parametrize("sample,xs,ys", [data["values_test"]])
def test_empirical_cdf(sample, xs, ys):
    x, y = empirical_cdf(sample)

    assert_allclose(x, xs, rtol=1e-7)
    assert_allclose(y, ys, rtol=1e-7)


# load data for test_estimate_mc
with open("seismostats/analysis/tests/data/test_estimate_mc.p", "rb") as f:
    data = pickle.load(f)


@pytest.mark.parametrize(
    "mags,mcs",
    [data["values_test"]],
)
def test_estimate_mc_ks(mags, mcs):

    # test when beta is not given
    best_mc, best_beta, mcs_tested, betas, ks_ds, ps = mc_ks(
        mags, delta_m=0.1, mcs_test=mcs, p_pass=0.1
    )
    assert_equal(1.1, best_mc)
    assert_almost_equal(2.242124985031149, best_beta)
    assert_equal([0.8, 0.9, 1.0, 1.1], mcs_tested)
    assert_allclose(
        [
            0.2819699492277921,
            0.21699092832299466,
            0.11605633802816911,
            0.07087102843116255,
        ],
        ks_ds,
        rtol=1e-7,
    )
    assert_allclose(
        [
            1.395015596110264,
            1.6216675481549436,
            1.9365312280350473,
            2.242124985031149,
        ],
        betas,
        rtol=1e-7,
    )
    assert_allclose([0.000e00, 1.000e-04, 5.100e-02, 4.362e-01], ps, atol=0.03)

    # test when beta is given
    best_mc, best_beta, mcs_tested, betas, ks_ds, ps = mc_ks(
        mags, delta_m=0.1, mcs_test=mcs, p_pass=0.1, beta=2.24
    )
    assert_equal(1.1, best_mc)
    assert_equal(2.24, best_beta)
    assert_allclose(
        [2.24, 2.24, 2.24, 2.24],
        betas,
        rtol=1e-7,
    )
    assert_allclose([0.0, 0.0, 0.0128, 0.4405], ps, atol=0.03)

    # test when mcs are not given
    best_mc, best_beta, mcs_tested, betas, ks_ds, ps = mc_ks(
        mags, delta_m=0.1, p_pass=0.1
    )
    assert_equal(1.1, best_mc)
    assert_almost_equal(2.242124985031149, best_beta)
    assert_equal([1.0, 1.1], mcs_tested)

    # test when b-positive is used
    best_mc, best_beta, mcs_tested, betas, ks_ds, ps = mc_ks(
        mags, delta_m=0.1, b_method="positive"
    )
    assert_equal(1.5, best_mc)
    assert_almost_equal(3.2542240043462796, best_beta)
    assert_equal(len(mcs_tested), 6)
    assert_equal(len(betas), 6)
    assert_equal(len(ks_ds), 6)
    assert_equal(len(ps), 6)


def test_estimate_mc_maxc(setup_magnitudes):
    mc = mc_max_curvature(setup_magnitudes, delta_m=0.1, correction_factor=0.2)

    assert_equal(1.3, mc)


def test_estimate_mc_bvalue_stability(swiss_2023_magnitudes):
    _, mc, _, _, _, _, _, _ = mc_by_bvalue_stability(
        swiss_2023_magnitudes[0], delta_m=swiss_2023_magnitudes[1],
        stability_factor=0.1)

    assert_equal(1.44, mc)
