import os

import numpy as np
from numpy.testing import assert_array_almost_equal

from catalogue_tools.recurrence.wls import weighted_least_squares


def test_wls():
    # Load in the catalogue from the completeness notebook
    PATH_RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'data')

    catalogue = np.genfromtxt(
        f'{PATH_RESOURCES}/test_completeness_catalogue.csv',
        delimiter=',',
        skip_header=1)

    # Sort into it's magnitudes and years
    magnitudes = catalogue[:, 0]
    years = catalogue[:, 1]

    # completeness by year
    completeness_table = np.array([[2005, 3.0],
                                  [1975, 4.0],
                                   [1960, 5.0],
                                   [1900, 6.0]])

    als, als_u, bls, bls_u = weighted_least_squares(
        years, magnitudes, completeness_table)

    assert_array_almost_equal([als, als_u, bls, bls_u],
                              [5.1076, 0.1304, 0.9796, 0.0254], 4)
