import numpy as np
from numpy.testing import assert_allclose
import pytest

from seismostats.plots.basics import dot_size, reverse_dot_size
from seismostats.plots.basics import plot_fmd


def test_dot_size():
    # Example input data
    magnitudes = np.array([1, 2, 3, 4, 5])
    smallest = 10
    largest = 200
    interpolation_power = 2

    # Expected output based on input data
    expected_sizes = np.array([10.0, 21.875, 57.5, 116.875, 200.0])

    # Compute dot sizes using the function
    sizes = dot_size(
        magnitudes,
        smallest=smallest,
        largest=largest,
        interpolation_power=interpolation_power,
    )

    # Check that the computed sizes are close to the expected ones
    tolerance = 1e-8
    assert_allclose(sizes, expected_sizes, rtol=tolerance, atol=tolerance)

    # Test with another set of input data
    magnitudes = np.array([5, 4, 3, 2, 1])
    smallest = 5
    largest = 50
    interpolation_power = 1

    # Expected output based on input data
    expected_sizes = np.array([50.0, 38.75, 27.5, 16.25, 5.0])

    # Compute dot sizes using the function
    sizes = dot_size(
        magnitudes,
        smallest=smallest,
        largest=largest,
        interpolation_power=interpolation_power,
    )

    # Check that the computed sizes are close to the expected ones
    assert_allclose(sizes, expected_sizes, rtol=tolerance, atol=tolerance)


def test_reverse_dot_size():
    # Example input data
    sizes = np.array([10.0, 21.875, 57.5, 116.875, 200.0])
    interpolation_power = 2

    # Expected output based on input data
    expected_magnitudes = np.array([1, 2, 3, 4, 5])
    magnitudes = reverse_dot_size(
        sizes,
        min_mag=expected_magnitudes[0],
        max_mag=expected_magnitudes[-1],
        interpolation_power=interpolation_power,
    )

    tolerance = 1e-8
    assert_allclose(
        magnitudes, expected_magnitudes, rtol=tolerance, atol=tolerance
    )

    # Test in combination with the default values of dot_size()

    # Example input data
    magnitudes_expected = np.array([1, 2, 3, 4, 5])
    sizes = dot_size(magnitudes_expected)

    # Expected output based on input data
    magnitudes = reverse_dot_size(
        sizes, min_mag=magnitudes_expected[0], max_mag=magnitudes_expected[-1]
    )

    # Check that the computed magnitudes are close to the expected ones
    assert_allclose(
        magnitudes, magnitudes_expected, rtol=tolerance, atol=tolerance
    )


def test_plot_fmd_none():
    magnitudes = np.array([0.235, -0.235, 4.499, 4.5, 6, 0.1, 1.6])

    with pytest.raises(TypeError):
        plot_fmd(magnitudes)
