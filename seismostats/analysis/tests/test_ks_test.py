import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal

from seismostats.analysis.ks_test import ks_test_gr, cdf_discrete_exp

MAGNITUDES = np.array(
    [
        2.3, 1.2, 1.5, 1.2, 1.7, 1.1, 1.2, 1.5, 1.8, 1.6, 1.2, 1.5,
        1.2, 1.7, 1.6, 1.1, 1.1, 1.2, 2.0, 1.1, 1.2, 1.1, 1.2, 1.6,
        1.9, 1.3, 1.7, 1.3, 1.0, 1.2, 1.7, 1.3, 1.3, 1.1, 1.5, 1.4,
        1.1, 2.1, 1.2, 2.2, 1.7, 1.6, 1.1, 2.0, 2.1, 1.2, 1.0, 1.5,
        1.2, 1.7, 1.8, 1.1, 1.3, 1.1, 1.3, 1.4, 2.1, 2.0, 1.1, 2.2,
        1.8, 1.4, 1.1, 1.0, 2.0, 2.0, 1.1, 1.0, 1.0, 1.5, 1.6, 3.7,
        2.8, 1.5, 1.1, 1.2, 1.4, 2.3, 1.5, 1.2, 1.7, 1.1, 1.6, 1.2,
        1.5, 1.1, 1.2, 1.7, 1.2, 1.6, 1.2, 1.1, 1.8, 1.2, 1.1, 1.0,
        1.3, 1.1, 1.6, 1.6,
    ]
)

KS_DISTS = pd.read_csv(
    'seismostats/analysis/tests/data/ks_ds.csv', index_col=0).values.T


def test_ks_test_gr():
    mc = 1.0
    delta_m = 0.1
    out = ks_test_gr(MAGNITUDES, mc, delta_m, b_value=1, ks_ds=KS_DISTS[0])
    assert_almost_equal(out[0], 0.0104)


def test_cdf_discrete_exp():
    mc = 1.0
    delta_m = 0.1
    _, y = cdf_discrete_exp(MAGNITUDES, mc, delta_m, beta=2)
    assert_almost_equal(y,
                        [0.18126925, 0.32967995, 0.45118836, 0.55067104,
                         0.63212056, 0.69880579, 0.75340304, 0.79810348,
                         0.83470111, 0.86466472, 0.88919684, 0.90928205,
                         0.92572642, 0.93918994, 0.97762923, 0.99630214])
