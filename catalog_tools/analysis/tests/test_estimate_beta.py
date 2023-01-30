import pytest
import numpy as np

# import functions from other modules
from catalog_tools.utils.simulate_distributions import simulate_magnitudes
from catalog_tools.utils.binning import bin_to_precision
# import functions to be tested
from catalog_tools.analysis.estimate_beta import\
    estimate_beta_tinti,\
    estimate_beta_utsu,\
    estimate_beta_elst,\
    estimate_beta_laplace,\
    differences


def simulate_magnitudes_w_offset(n: int, beta: float, mc: float,
                                 delta_m: float) -> np.ndarray:
    """ This function simulates the magnitudes with the correct offset"""
    mags = simulate_magnitudes(n, beta, mc - delta_m / 2)
    if delta_m > 0:
        mags = bin_to_precision(mags, delta_m)
    return mags


@pytest.mark.parametrize(
    "n,beta,mc,delta_m,precision",
    [(1000000, np.log(10), 3, 0, 0.005),
     (1000000, np.log(10), 3, 0.1, 0.01)]
)
def test_estimate_beta_tinti(n: int, beta: float, mc: float, delta_m: float,
                             precision: float):
    mags = simulate_magnitudes_w_offset(n, beta, mc, delta_m)
    beta_estimate = estimate_beta_tinti(mags, mc, delta_m)
    print((beta_estimate - beta) / beta)
    assert abs(beta - beta_estimate) / beta <= precision


@pytest.mark.parametrize(
    "n,beta,mc,delta_m,precision",
    [(1000000, np.log(10), 3, 0, 0.005),
     (1000000, np.log(10), 3, 0.1, 0.01)]
)
def test_estimate_beta_utsu(n: int, beta: float, mc: float, delta_m: float,
                            precision: float):
    mags = simulate_magnitudes_w_offset(n, beta, mc, delta_m)
    beta_estimate = estimate_beta_utsu(mags, mc, delta_m)
    assert abs(beta - beta_estimate) / beta <= precision


@pytest.mark.parametrize(
    "magnitudes,mag_diffs",
    [(np.array([1, -2, 3]),
      np.array([-3, 2, 3, 5, -2, -5]))]
)
def test_differences(magnitudes: np.ndarray, mag_diffs: np.ndarray):
    y = differences(magnitudes)
    assert (y == mag_diffs).all()


@pytest.mark.parametrize(
    "n,beta,mc,delta_m,precision",
    [(1000000, np.log(10), 3, 0, 0.005),
     (1000000, np.log(10), 3, 0.1, 0.01)]
)
def test_estimate_beta_elst(n: int, beta: float, mc: float, delta_m: float,
                            precision: float):
    mags = simulate_magnitudes_w_offset(n, beta, mc, delta_m)
    beta_estimate = estimate_beta_elst(mags, delta_m=delta_m)
    assert abs(beta - beta_estimate) / beta <= precision


@pytest.mark.parametrize(
    "n,beta,mc,delta_m,precision",
    [(1000, np.log(10), 3, 0, 0.15),
     (1000, np.log(10), 3, 0.1, 0.2)]
)
def test_estimate_beta_laplace(n: int, beta: float, mc: float, delta_m: float,
                               precision: float):
    mags = simulate_magnitudes_w_offset(n, beta, mc, delta_m)
    beta_estimate = estimate_beta_laplace(mags, delta_m=delta_m)
    assert abs(beta - beta_estimate) / beta <= precision
