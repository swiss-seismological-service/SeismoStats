import numpy as np
from numpy.testing import assert_allclose

from catalog_tools.plots.basics import dot_size


def test_dot_size():
    # Example input data
    magnitudes = np.array([1, 2, 3, 4, 5])
    smallest = 10
    largest = 200
    interpolation_power = 2

    # Expected output based on input data
    expected_sizes = np.array([ 10.   ,  21.875,  57.5  , 116.875, 200.   ])

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
    expected_sizes = np.array([50.  , 38.75, 27.5 , 16.25,  5.  ])

    # Compute dot sizes using the function
    sizes = dot_size(magnitudes, smallest=smallest, largest=largest,
                     interpolation_power=interpolation_power)

    # Check that the computed sizes are close to the expected ones
    assert_allclose(sizes, expected_sizes, rtol=tolerance, atol=tolerance)
