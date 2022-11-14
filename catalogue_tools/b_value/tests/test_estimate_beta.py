import numpy as np
from catalogue_tools.utils.simulate_distributions import simulate_magnitudes
from catalogue_tools.b_value.estimate_beta import estimate_beta_tinti
from catalogue_tools.b_value.estimate_beta import estimate_beta_utsu

n = 1000000
beta = 1 / np.log10(np.e)
mc = 2.0
precision = 0.005
# generate synthetic magnitudes
mags = simulate_magnitudes(n, beta, mc)

def test_estimate_beta_tinti(mags, mc, delta_m, beta):
    beta_estimate = estimate_beta_tinti(mags, mc, delta_m)
    assert np.log10(np.e) * abs(beta - beta_estimate) <= precision


def test_estimate_beta_utsu(mags, mc, delta_m, beta):
    beta_estimate = estimate_beta_utsu(mags, mc, delta_m)
    assert np.log10(np.e) * abs(beta - beta_estimate) <= precision


if __name__ == '__main__':
    n = 1000000
    beta = 1 / np.log10(np.e)
    mc = 2.0
    precision = 0.005
    mags = simulate_magnitudes(n, beta, mc)    

    # no binning
    delta_m = 0.0
    test_estimate_beta_tinti(mags, mc, delta_m, beta)
    test_estimate_beta_utsu(mags, mc, delta_m, beta)

    # with binning
