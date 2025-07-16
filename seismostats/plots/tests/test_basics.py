import numpy as np
from numpy.testing import assert_allclose
import matplotlib.pyplot as plt
import pytest

from seismostats.plots.basics import dot_size, reverse_dot_size
from seismostats.plots.basics import plot_fmd, plot_cum_count, plot_cum_fmd


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


def test_plot_fmd():
    # Example data
    magnitudes = np.array([2.0, 2.1, 2.3, 2.5, 2.7, 3.0])
    fmd_bin = 0.2

    # test with no fmd_bin
    with pytest.raises(TypeError):
        plot_fmd(magnitudes)

    # Call plot function
    ax = plot_fmd(magnitudes, fmd_bin, color='blue',
                  size=20, grid=True, label="Test")

    # Check labels
    assert ax.get_xlabel() == "Magnitude"
    assert ax.get_ylabel() == "N"
    assert ax.get_yscale() == "log"

    # Check if scatter points are plotted
    assert len(ax.collections) == 1

    # Check color
    scatter = ax.collections[0]
    facecolors = scatter.get_facecolor()
    # Convert RGBA to hex or check shape
    assert facecolors.shape[0] > 0

    # Check legend
    legend = ax.get_legend()
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    assert "Test" in labels
    plt.close(ax.figure)  # Avoid leaving open figures


def test_plot_cum_fmd():
    magnitudes = np.array([2.5, 3.0, 3.2, 2.8, 3.5])
    ax = plot_cum_fmd(magnitudes, b_value=1.0)

    # default labels
    legend = ax.get_legend()
    assert legend is not None, "Expected a legend but none was found."
    legend_labels = [text.get_text() for text in legend.get_texts()]
    expected_labels = ['cumulative', 'GR fit, b=1.00']
    assert legend_labels == expected_labels

    # no labels
    ax = plot_cum_fmd(magnitudes, b_value=1.0, label=False, label_line=False)
    legend = ax.get_legend()
    assert legend is None, "Expected no legend but one was found."

    # given labels
    ax = plot_cum_fmd(magnitudes, b_value=1.0,
                      label='Test1', label_line='Test2')
    legend = ax.get_legend()
    assert legend is not None, "Expected a legend but none was found."
    legend_labels = [text.get_text() for text in legend.get_texts()]
    expected_labels = ['Test1', 'Test2']
    assert legend_labels == expected_labels


def test_plot_cum_count():
    # Example data
    times = np.array([1, 2, 3, 4, 5])
    magnitudes = np.array([2.5, 3.0, 3.2, 2.8, 3.5])
    mcs = np.array([2.5])

    # Call function
    ax = plot_cum_count(times, magnitudes, mcs=mcs, normalize=True)

    # Check axis labels
    assert ax.get_xlabel() == "Time"
    assert ax.get_ylabel() == "Cumulative number of events"

    # Check at least one line was plotted
    lines = ax.get_lines()
    assert len(lines) == 1

    # Check line data
    line = lines[0]
    x_data = line.get_xdata()
    y_data = line.get_ydata()

    # There should be some data points
    assert len(x_data) == len(y_data)
    assert len(x_data) > 2

    # Check legend text
    legend = ax.get_legend()
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    assert any("Mc=" in label for label in labels)

    plt.close(ax.figure)  # Cleanup

    # check that fails when color is array of wrong length
    mcs = np.array([2, 3, 4, 5])
    colors = np.array(['red', 'green', 'blue', 'k'])
    plot_cum_count(times, magnitudes, mcs=mcs, color=colors)
    with pytest.raises(ValueError):
        plot_cum_count(times, magnitudes, mcs=mcs, color=colors[1:])
