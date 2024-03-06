from numpy.testing import assert_allclose, assert_equal
import pickle
import pytest

from seismostats.analysis.estimate_mc import (
    empirical_cdf,
    mc_ks,
    mc_max_curvature,
)


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
    mcs, ks_ds, ps, best_mc, beta = mc_ks(
        mags, mcs, delta_m=0.1, p_pass=0.1
    )

    assert_equal([0.8, 0.9, 1.0, 1.1], mcs)
    assert_equal(2.242124985031149, beta)
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
    assert_allclose([0.000e00, 1.000e-04, 5.100e-02, 4.362e-01], ps, atol=0.03)
    assert_equal(1.1, best_mc)


def test_estimate_mc_maxc(setup_magnitudes):
    mc = mc_max_curvature(setup_magnitudes, delta_m=0.1, correction_factor=0.2)

    assert_equal(1.3, mc)
