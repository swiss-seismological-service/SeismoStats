import numpy as np
from seismostats.utils.simulate_distributions import dither_magnitudes
from seismostats.utils.binning import binning_test


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


def test_dither_magnitudes():
    delta_m = 0.1
    mags_dith = dither_magnitudes(
        MAGNITUDES,
        delta_m=delta_m,
        b_value=1.0
    )

    # Check if magnitudes are within expected range
    assert min(MAGNITUDES) - delta_m / 2 <= min(mags_dith)
    assert max(MAGNITUDES) + delta_m / 2 >= max(mags_dith)

    # Check that length is preserved
    assert len(MAGNITUDES) == len(mags_dith)

    # Check that magnitudes are not binned anymore
    assert binning_test(mags_dith, delta_x=0.1) is False
    assert binning_test(mags_dith, delta_x=0.01) is False
