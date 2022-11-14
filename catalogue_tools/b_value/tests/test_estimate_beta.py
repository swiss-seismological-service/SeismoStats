import numpy as np

# import functions from other modules
from catalogue_tools.utils.simulate_distributions import simulate_magnitudes

# import functions to be tested
from catalogue_tools.b_value.estimate_beta import estimate_beta_tinti
from catalogue_tools.b_value.estimate_beta import estimate_beta_utsu
from catalogue_tools.b_value.estimate_beta import estimate_beta_elst


def test_estimate_beta_tinti():
    n = 1000000
    beta = 1 / np.log10(np.e)
    mc = 2.0
    precision = 0.005

    # generate synthetic magnitudes
    mags = simulate_magnitudes(n, beta, mc)

    # without binning
    delta_m = 0.0
    beta_estimate = estimate_beta_tinti(mags, mc, delta_m)
    assert np.log10(np.e) * abs(beta - beta_estimate) <= precision


def test_estimate_beta_utsu():
    n = 1000000
    beta = 1 / np.log10(np.e)
    mc = 2.0
    precision = 0.005

    # generate synthetic magnitudes
    mags = simulate_magnitudes(n, beta, mc)

    # without binning
    delta_m = 0.0
    beta_estimate = estimate_beta_utsu(mags, mc, delta_m)
    assert np.log10(np.e) * abs(beta - beta_estimate) <= precision


def test_estimate_beta_elst():
    n = 1000000
    beta = 1 / np.log10(np.e)
    mc = 2.0
    precision = 0.005

    # generate synthetic magnitudes
    mags = simulate_magnitudes(n, beta, mc)

    # without binning
    beta_estimate = estimate_beta_elst(mags)
    assert np.log10(np.e) * abs(beta - beta_estimate) <= precision


if __name__ == '__main__':
    test_estimate_beta_tinti()
    test_estimate_beta_utsu()
    test_estimate_beta_elst()
