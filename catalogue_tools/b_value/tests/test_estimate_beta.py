import numpy as np

# import functions from other modules
from catalogue_tools.utils.simulate_distributions import simulate_magnitudes

# import functions to be tested
from catalogue_tools.b_value.estimate_beta import estimate_beta_tinti
from catalogue_tools.b_value.estimate_beta import estimate_beta_utsu
from catalogue_tools.b_value.estimate_beta import estimate_beta_elst

# parameters for testing
n = 1000000
beta = np.log(10)
mc = 2.0
precision = 0.005


def test_estimate_beta_tinti():
    # generate synthetic magnitudes
    mags = simulate_magnitudes(n, beta, mc)
    # without binning
    delta_m = 0.0
    beta_estimate = estimate_beta_tinti(mags, mc, delta_m)
    assert abs(beta - beta_estimate) / beta <= precision


def test_estimate_beta_utsu():
    # generate synthetic magnitudes
    mags = simulate_magnitudes(n, beta, mc)
    # without binning
    delta_m = 0.0
    beta_estimate = estimate_beta_utsu(mags, mc, delta_m)
    assert abs(beta - beta_estimate) / beta <= precision


def test_estimate_beta_elst():
    # generate synthetic magnitudes
    mags = simulate_magnitudes(n, beta, mc)
    # without binning
    beta_estimate = estimate_beta_elst(mags)
    assert abs(beta - beta_estimate) / beta <= precision
