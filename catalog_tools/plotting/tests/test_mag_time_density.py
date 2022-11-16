import os

import numpy as np
from numpy.testing import assert_equal

from catalog_tools.plotting.mag_time_density import \
    plot_magnitude_time_density


def test_mag_time_density():
    PATH_RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'data')
    catalog = np.genfromtxt(
        f'{PATH_RESOURCES}/test_completeness_catalog.csv',
        delimiter=',',
        skip_header=1)

    # Sort into it's magnitudes and years
    magnitudes = catalog[:, 0]
    years = catalog[:, 1]

    # completeness table
    completeness_table = np.array([[2005, 3.0],
                                   [1975, 4.0],
                                   [1960, 5.0],
                                   [1900, 6.0]])

    # As we did before, let's take a look at the density of events
    magnitude_bins = np.arange(3.0, 7.1, 0.1)

    year_bins = np.arange(years.min(), years.max() + 1, 1)

    plt = plot_magnitude_time_density(magnitudes, years,
                                      mbins=magnitude_bins,
                                      time_bins=year_bins,
                                      completeness_table=completeness_table,
                                      filename=f'{PATH_RESOURCES}/myfile.png')
    # make sure that the plot is created, then delete
    assert os.path.isfile(f'{PATH_RESOURCES}/myfile.png')
    os.remove(f'{PATH_RESOURCES}/myfile.png')

    # check that y data is correct
    assert_equal(plt.gca().lines[0].get_data()[1], np.array([3., 4., 5., 6.]))


if __name__ == '__main__':
    test_mag_time_density()
