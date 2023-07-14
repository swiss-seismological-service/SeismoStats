import pytest
import numpy as np
from numpy.testing import assert_allclose

from catalog_tools.plots.basics import dot_size, get_cum_fmd


def test_dot_size():
    # Example input data
    magnitudes = np.array([1, 2, 3, 4, 5])
    smallest = 10
    largest = 200
    interpolation_power = 2

    # Expected output based on input data
    expected_sizes = np.array([10., 21.875, 57.5, 116.875, 200.])

    # Compute dot sizes using the function
    sizes = dot_size(magnitudes, smallest=smallest, largest=largest,
                     interpolation_power=interpolation_power)

    # Check that the computed sizes are close to the expected ones
    tolerance = 1e-8
    assert_allclose(sizes, expected_sizes, rtol=tolerance, atol=tolerance)

    # Test with another set of input data
    magnitudes = np.array([5, 4, 3, 2, 1])
    smallest = 5
    largest = 50
    interpolation_power = 1

    # Expected output based on input data
    expected_sizes = np.array([50., 38.75, 27.5, 16.25, 5.])

    # Compute dot sizes using the function
    sizes = dot_size(magnitudes, smallest=smallest, largest=largest,
                     interpolation_power=interpolation_power)

    # Check that the computed sizes are close to the expected ones
    assert_allclose(sizes, expected_sizes, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize(
    "magnitudes, delta_m, bins, c_counts, left",
    [(np.array([0.20990507, 0.04077336, 0.27906596, 0.57406287, 0.64256544,
                0.07293118, 0.58589873, 0.02678655, 0.27631233, 0.17682814]),
      0.1, np.array([0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]),
      np.array([3, 3, 3, 5, 7, 8, 10]), False),
     (np.array([0.02637757, 0.06353823, 0.10257919, 0.54494906, 0.03928375,
                0.08825028, 0.77713586, 0.54553981, 0.69315583, 0.06656642,
                0.29035447, 0.2051877, 0.30858087, 0.68896342, 0.03328782,
                0.45016109, 0.40779409, 0.06788892, 0.02684032, 0.56140282,
                0.29443359, 0.36328762, 0.17124489, 0.02154936, 0.36461541,
                0.03613088, 0.15798366, 0.09111875, 0.16169287, 0.11986668,
                0.10232035, 0.72695761, 0.19484174, 0.0459675, 0.40514163,
                0.08979514, 0.0442659, 0.18672424, 0.21239088, 0.02287468,
                0.1244267, 0.04939361, 0.11232758, 0.02706083, 0.04275401,
                0.02732529, 0.83884229, 0.4147758, 0.07416183, 0.05636252]),
      0.2, np.array([0.7, 0.5, 0.3, 0.1, -0.1]), np.array([3, 8,
                                                           15, 29, 50]), True)]
)
def test_get_cum_fmd(magnitudes: np.ndarray, delta_m: float,
                     bins: np.ndarray, c_counts: np.ndarray, left: bool):
    errors = []
    nbins, nc_counts, nmags = get_cum_fmd(magnitudes, delta_m, left=left)
    if not np.allclose(bins, nbins, atol=1e-10):
        errors.append("Incorrect bin values.")
    if not np.allclose(c_counts, nc_counts, atol=1e-10):
        errors.append("Incorrect cumulative counts.")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))
